# -*- coding: utf-8 -*-

""" AlexNet.

Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

dataset_file = r'/home/ubuntu/cs249_final_project/image_files/train'

from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset(dataset_file, image_shape=(300, 300), mode='folder', output_path='dataset.h5', categorical_labels=True, normalize=True)

import h5py
h5f = h5py.File('dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']

net = tflearn.input_data(shape=[None, 300, 300, 3])
net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
# Residual blocks
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
# Training
model = tflearn.DNN(net, checkpoint_path='RNN',
                    max_checkpoints=10, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=None,
          show_metric=True, batch_size=64, snapshot_step=50,
          snapshot_epoch=False, run_id='CPU')
