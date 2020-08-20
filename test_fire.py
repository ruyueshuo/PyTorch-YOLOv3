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

CLASSES = ['fire']


def web_video_test(model, opt,
                   url='rtsp://dlxh:dlxh@2017@10.10.77.17/Streaming/Channels/301',
                   scale=0.5, frame_rate=1, pred_score_thr=0.6, video_name=None, save_video=None,
                   ):
    # settings
    frame_n = 0

    # capture video stream from url
    cap = cv2.VideoCapture(url)
    # 获得码率及尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale))
    print(cap)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if save_video is not None:
        # 指定写视频的格式, I420-avi, MJPG-mp4
        videoWriter = cv2.VideoWriter(save_video, fourcc, fps, size)

    # read frame data
    ret, frame = cap.read()

    while ret:
        ret, frame = cap.read()

        if ret and (frame_n % (frame_rate) == 0):
            # resize data
            # frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)

            # out_file = os.path.join(save_folder, '')
            # inference
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = test_single_img(model, frame, opt, pred_score_thr=pred_score_thr, show_name=video_name)

            if save_video is not None:
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                videoWriter.write(frame)  # 写视频帧

        frame_n += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


def test_image(model, img_path, opt, pred_score_thr=0.5, show_name=None):
    # load image
    frame = Image.open(img_path).convert('RGB')

    # resize to image size in option
    scale = min(opt.img_size / frame.size[0], opt.img_size / frame.size[1])
    frame = frame.resize((int(frame.size[0] * scale), int(frame.size[1] * scale)), Image.BILINEAR)

    frame_np = np.array(frame)

    # test
    test_single_img(model, frame_np, opt, pred_score_thr=pred_score_thr, show_name=show_name, save_file='output/test.jpg')


def test_single_img(model, frame, opt, pred_score_thr=0.5, show_name=None, save_file=None, min_box=128):
    # settings
    font = cv2.FONT_HERSHEY_SIMPLEX

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
        print(img.shape)
        outputs = model(img)
        t2 = time.time()
        outputs = non_max_suppression(outputs, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
        t3 = time.time()

        output = outputs[0].numpy()
        if output.any():
            num = 0
            for i in range(output.shape[0]):
                d = output[i, :]
                pred_boxes = d[0:4].astype(int)
                pred_scores = d[4]
                pred_labels = d[-1].astype(int)
                box = (pred_boxes[2] - pred_boxes[0]) * (pred_boxes[3] - pred_boxes[1])
                if (pred_scores > pred_score_thr) and (box >= min_box):
                    # add annotations
                    frame = cv2.rectangle(frame, (pred_boxes[0] - pad[0], pred_boxes[1] - pad[3]),
                                          (pred_boxes[2] - pad[0], pred_boxes[3] - pad[3]), (0, 255, 0), 2)
                    frame = cv2.putText(frame, opt.class_names[pred_labels],
                                        (pred_boxes[0] - pad[0], pred_boxes[1] - pad[3] - 10), font, 0.5, (0, 255, 0), 2)
                    # if num == 0:
                    #     frame = cv2.putText(frame, "Fire Warning!",
                    #                         (20, 20), font, 1, (255, 0, 0), 2)
                    num += 1
                    print("bbox:{}\tclass:{}".format(pred_boxes, opt.class_names[pred_labels]))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # show frame
        if show_name is not None:
            cv2.imshow(show_name, frame)

        # time cost
        print("Inference time cost:{}s".format(t2 - t1))
        print("Inference + nms time cost:{}s".format(t3 - t1))
        print("Total time cost:{}s".format(time.time() - t1))
        if save_file is not None:
            cv2.imwrite(save_file, frame)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-fire.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/fire.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str,
                        # default="weights/yolov3-tiny.weights",
                        # default="/home/ubuntu/code/fengda/PyTorch-YOLOv3/checkpoints/fire_det/20200818-111706/yolov3_fire_det_ckpt_34_0.39812.pth",
                        default="/home/ubuntu/code/fengda/PyTorch-YOLOv3/checkpoints/fire_det/20200818-104346/yolov3_fire_det_ckpt_14_0.44713.pth",
                        # default="/home/ubuntu/code/fengda/PyTorch-YOLOv3/checkpoints/fire_det/yolov3_fire_det_ckpt_60_0.42646.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.05, help="object confidence threshold")
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
        # Load darknettest_fire.py weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    img_path = "/home/ubuntu/datasets/fire_detection/VOC2020/JPEGImages/396f209c-9f89-4871-9127-bd64e14b13e5.jpg"
    # frame = cv2.imread(img_path)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # resize data
    # frame = cv2.resize(frame, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)

    # test_image(model, img_path)
    # web_video_test(model, url=0, scale=0.75, frame_rate=10)
    # web_video_test(model)
    web_video_test(model, url='/home/ubuntu/datasets/fire_detection/VOC2020/test_fire_video_train.mp4', pred_score_thr=0.95, scale=0.25)


