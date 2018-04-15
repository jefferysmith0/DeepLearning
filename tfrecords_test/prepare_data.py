# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

dir = "imgs/"  # 加载jpg， testShape = (333, 500, 3)


def data_augmentation(data):
    """
    数据增强处理
    :param data:
    :return:
    """
    return data


def get_img_data(file_dir):
    """
    获取图片数据， 返回类型是 list
    :param file_dir: 图片所在目录
    :return: 返回类型是 list
    """
    files = [os.path.join('imgs', x) for x in os.listdir(file_dir)]
    raw_data = [cv2.imread(img) for img in files]
    raw_data = data_augmentation(raw_data)
    return raw_data


if __name__ == "__main__":
    get_img_data(dir)
