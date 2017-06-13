from sklearn import svm
import pandas as pd
from sklearn.decomposition import PCA
import time
import os
import numpy as np

#dataset_file = r'/Users/watermelon/Desktop/Courses/CS249Sun/prj/data/small'
dataset_file = '/home/ubuntu/small'
from tflearn.data_utils import image_preloader
data = []
label = []
print("Loading images...")
X,Y = image_preloader(dataset_file, image_shape=(300, 300), mode='folder', categorical_labels=True, normalize=True)
from scipy import sqrt, pi, arctan2, cos, sin
qqq=sqrt(X)

for i in range(X.__len__()):
    items = X.__getitem__(i).reshape(90000,3)
    #data.append(rgb2hex(items))
    data.append(items.reshape(270000,1))
    label.append((Y.__getitem__(i)).tolist().index(1))

data = np.array(data)
data=data.reshape(data.shape[0],data.shape[1])
label = np.array(label)

from random import random
# Randomly shuffle data
np.random.seed(131)
shuffle_indices = np.random.permutation(np.arange(len(label)))
data_shuffled = np.array(np.asarray(data))[shuffle_indices]
label_shuffled = label[shuffle_indices]

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
data_shuffled = scaler.fit_transform(data_shuffled)
#save train_data
# print("Saving data...")
# np.save('data.npy',data)
# np.save('label.npy',label)

# LOAD train_data
# data=np.load('data.npy')
# label=np.load('label.npy')

train_data=data_shuffled[0:200,0:270000]
train_label=label_shuffled[0:200]
test_data=data_shuffled[200:237,0:270000]
test_label=label_shuffled[200:237]

print("Initializing SVM...")
# Stochastic Gradient Descent Classfier
from sklearn import linear_model
clf = linear_model.SGDClassifier(loss='log')
clf.fit(data_shuffled, label_shuffled)
print(clf.score(data_shuffled,label_shuffled))
from sklearn import metrics
print(metrics.classification_report(clf.predict(data_shuffled), label_shuffled))
from sklearn.linear_model import SGDClassifier

print("Doing cross validation...")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, data_shuffled,label_shuffled,cv=5)
print(scores)

import sklearn.cross_validation
scores=cross_validation.cross_val_score(clf,data,label,cv=5)
print (clf.score(data,label))
