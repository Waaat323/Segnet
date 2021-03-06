# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:39:27 2017

@author: kawalab
"""

import numpy as np
import matplotlib.pyplot as plt


from chainercv.datasets import CamVidDataset
from chainer.datasets import TransformDataset


import configparser

cp = configparser.ConfigParser()
cp.read('config')
root_dir = cp.get('dataset_dir', 'dir_path')


def transform(in_data):
    img, label = in_data
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1]
        label = label[:, ::-1]
    return img, label


def CamVid_loader(dataset_dir=root_dir):

    # Dataset
    train = CamVidDataset(dataset_dir, split='train')
    train = TransformDataset(train, transform)
    test = CamVidDataset(dataset_dir, split='val')

"""
    train_images = np.array(train[:, 0])
    train_classes = np.array(train[:, 1])
    test_images = np.array(test[:, 0])
    test_classes = np.array(test[:, 1])
"""

    train_images = train[:, 0]
    train_classes = train[:, 1]
    test_images = test[:, 0]
    test_classes = test[:, 1]

    train_images /= 255.0
    train_classes /= 255.0
    test_images /= 255.0
    test_classes /= 255.0

    return train_images, test_images, train_labels, test_labels


if __name__ == '__main__':
    (train_images, test_images,
     train_labels, test_labels) = np.(CamVid_loader())

    a = train_images.transpose(train_images[0](1, 2, 0))
    b = train_labels[0]
    c = test_images.transepose(test_images[0](1, 2, 0))
    d = train_labels[0]

    plt.imshow(a)
    plt.imshow(b)

    plt.imshow(c)
    plt.imshow(d)