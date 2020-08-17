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

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
           'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
           'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
           'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')


def web_video_test(model, url='rtsp://dlxh:dlxh@2017@10.10.77.17/Streaming/Channels/301', scale=0.5, frame_rate=5, pred_score_thr=0.6,video_name='current frame'):
    # settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_n = 0

    # capture video stream from url
    cap = cv2.VideoCapture(url)
    print(cap)

    # read frame data
    ret, frame = cap.read()

    while ret:
        t0 = time.time()
        ret, frame = cap.read()

        if frame_n % (frame_rate) == 0:
            # resize data
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            with torch.no_grad():
                # Extract image as PyTorch tensor
                img = transforms.ToTensor()(frame)

                # Handle images with less than three channels
                if len(img.shape) != 3:
                    img = img.unsqueeze(0)
                    img = img.expand((3, img.shape[1:]))

                # Pad to square resolution
                img, pad = pad_to_square(img, 0)
                img = img.unsqueeze(0).cuda()

                # inference
                t1 = time.time()
                outputs = model(img)
                t2 = time.time()
                outputs = non_max_suppression(outputs, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
                t3 = time.time()

                output = outputs[0].numpy()
                if output.any():
                    for i in range(output.shape[0]):
                        d = output[i,:]
                        pred_boxes = d[0 :4].astype(int)
                        pred_scores = d[4]
                        pred_labels = d[-1].astype(int)
                        if pred_scores > pred_score_thr:
                            # add annotations
                            frame = cv2.rectangle(frame, (pred_boxes[0], pred_boxes[1]-pad[3]), (pred_boxes[2], pred_boxes[3]-pad[3]), (0, 255, 0), 2)
                            frame = cv2.putText(frame, CLASSES[pred_labels], (pred_boxes[0], pred_boxes[1]-pad[3]-10), font, 0.5, (0, 255, 0), 2)

                            print("bbox:{}\tclass:{}".format(pred_boxes, CLASSES[pred_labels]))
            # show frame
            # cv2.imshow(video_name, frame)

            # time cost
            print("Inference time cost:{}s".format(t2 - t1))
            print("Inference + nms time cost:{}s".format(t3 - t1))
            print("Total time cost:{}s".format(time.time() - t0))
            # cv2.imwrite('frame.jpg', frame)
        frame_n += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3-tiny.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    web_video_test(model, url=0, scale=0.75, frame_rate=10)
    # web_video_test(model)


