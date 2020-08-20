from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from  test_fire import *
CLASSES = ['hat', 'person']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-safety-helmet.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/safety-helmet.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str,
                        # default="weights/yolov3-tiny.weights",
                        default="/home/ubuntu/code/fengda/PyTorch-YOLOv3/checkpoints/safety-helmet/20200819-113615/yolov3_safety_helmet_det_ckpt_24_0.80629.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.05, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    opt.class_names = class_names

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknettest_fire.py weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    img_path = "/home/ubuntu/datasets/VOC2028-安全帽/VOC2028/JPEGImages/000104.jpg"
    # frame = cv2.imread(img_path)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # resize data
    # frame = cv2.resize(frame, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    #
    test_image(model, img_path, opt, pred_score_thr=0.8, show_name=None)
    # web_video_test(model, url=0, scale=0.75, frame_rate=10)
    # web_video_test(model)
    # web_video_test(model, opt, url='/home/ubuntu/datasets/VOC2028-安全帽/VOC2028/test_safet_helmet_video1.mp4',
    #                pred_score_thr=0.92, scale=0.5, frame_rate=2,
    #                save_video='/home/ubuntu/datasets/VOC2028-安全帽/VOC2028/test_safet_helmet_video1_result.avi')


