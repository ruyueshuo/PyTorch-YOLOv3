#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/13 16:23
# @Author  : FengDa
# @File    : train_fire.py
# @Software: PyCharm
from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-safety-helmet.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/safety-helmet.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str,
                        default="/home/ubuntu/code/fengda/PyTorch-YOLOv3/checkpoints/safety-helmet/20200819-113615/yolov3_safety_helmet_det_ckpt_26_0.77371.pth",
                        # default=None,
                        # default="/home/ubuntu/code/fengda/PyTorch-YOLOv3/weights/yolov3.weights",
                        # default="/home/ubuntu/code/fengda/PyTorch-YOLOv3/checkpoints/fire_det/latest.pth",
                        help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--lr", type=float, default=0.000025, help="learning rate")
    parser.add_argument("--seed", type=int, default=41, help="learning rate")
    parser.add_argument("--output_folder", type=str, default='checkpoints/safety-helmet/', help="learning rate")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    opt.output_folder = os.path.join(opt.output_folder, now)
    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # device = "cuda:0"

    set_random_seed(opt.seed)

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    # class_names = load_classes(data_config["names"])
    class_names = ['safety-helmet', 'person']

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    # model = nn.DataParallel(model, device_ids=[0,1]).modules()

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = FireDataset(train_path, data_config['data'], augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
                mode='fire',
                data_config=data_config
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        scheduler.step()

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(opt.output_folder, "yolov3_safety_helmet_det_ckpt_%d_%.5f.pth" % (epoch, AP.mean())))
