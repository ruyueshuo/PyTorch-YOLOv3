#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/13 16:26
# @Author  : FengDa
# @File    : train_val_split.py
# @Software: PyCharm
import os
import random


def split_dataset(data_path, save_path, ratios=None, seed=41):
    """
    split dataset to train and validation sets.
    :param data_path: str or Path. raw data path
    :param save_path: str or Path. train and validation text file saving path.
    :param ratios: list. rations of train and validation.
    :param seed: int. random seed.
    :return:
    """

    if ratios is None:
        ratios = [0.8, 0.2]

    # set random seed
    random.seed(seed)

    # get raw image list
    file_list = os.listdir(data_path)
    file_list = [file.split('.')[0] for file in file_list]
    file_num = len(file_list)

    # split dataset
    train_list = random.sample(file_list, int(file_num*ratios[0]))
    valid_list = list(set(file_list).difference(set(train_list)))

    # save results
    save_txt(train_list, os.path.join(save_path, 'train.txt'))
    save_txt(valid_list, os.path.join(save_path, 'valid.txt'))


def save_txt(data, file):
    """save data to text file."""
    with open(file, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')
        # f.write(str(data))
        f.close()


def add_annotation(img_path, ann_path):
    img_list = os.listdir(img_path)
    img_list = [img[:-4] for img in img_list]
    ann_list = os.listdir(ann_path)
    ann_list = [ann[:-4] for ann in ann_list]

    subtraction = list(set(img_list).difference(set(ann_list)))
    for img in subtraction:
        ann_file = os.path.join(ann_path, img + ".txt")
        with open(ann_file, 'w') as f:
            f.write('')
        f.close()


if __name__ == '__main__':
    path = '/home/ubuntu/datasets/fire_detection/VOC2020/JPEGImages'
    save_path = '/home/ubuntu/datasets/fire_detection/VOC2020/ImageSets/Main'
    split_dataset(path, save_path)

    # img_path = "/home/ubuntu/datasets/fire_detection/VOC2020/JPEGImages"
    # ann_path = "/home/ubuntu/datasets/fire_detection/VOC2020/labels"
    # add_annotation(img_path, ann_path)