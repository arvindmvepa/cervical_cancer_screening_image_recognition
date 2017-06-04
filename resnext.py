""" Aggregated Residual Transformations for Deep Neural Network.

Applying a 'ResNeXT' to CIFAR-10 Dataset classification task.

References:
    - S. Xie, R. Girshick, P. Dollar, Z. Tu and K. He. Aggregated Residual
        Transformations for Deep Neural Networks, 2016.

Links:
    - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

dataset_file = r'/home/ubuntu/cs249_final_project/image_files/train'

from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset(dataset_file, image_shape=(100, 100), mode='folder', output_path='dataset.h5', categorical_labels=True, normalize=True)

import h5py
h5f = h5py.File('dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']

# Real-time data preprocessing
#img_prep = tflearn.ImagePreprocessing()
#img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
#img_aug = tflearn.ImageAugmentation()
#img_aug.add_random_flip_leftright()
#img_aug.add_random_crop([100, 100], padding=4)

n=5

# Building Residual Network
net = tflearn.input_data(shape=[None, 100, 100, 3])
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.resnext_block(net, n, 16, 32)
net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 32, 32)
net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 64, 32)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 3, activation='softmax')
opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=opt,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_cifar10',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=200, validation_set=.1,
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnext')
