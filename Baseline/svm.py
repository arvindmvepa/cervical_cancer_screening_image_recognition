
from sklearn import svm
import pandas as pd
from sklearn.decomposition import PCA
import time
import os
import numpy as np

dataset_file = r'/Users/watermelon/Desktop/Courses/CS249Sun/prj/data/train'
#dataset_file = '/home/ubuntu/small'
from tflearn.data_utils import image_preloader
data = []
label = []
print("Loading images...")
X,Y = image_preloader(dataset_file, image_shape=(300, 300), mode='folder', categorical_labels=True, normalize=True)
for i in range(X.__len__()):
    data.append(X.__getitem__(i).reshape(270000,1))
    label.append((Y.__getitem__(i)).tolist().index(1))

data = np.array(data)
data=data.reshape(data.shape[0],data.shape[1])
label = np.array(label)

#save train_data
# print("Saving data...")
# np.save('data.npy',data)
# np.save('label.npy',label)

# LOAD train_data
# data=np.load('data.npy')
# label=np.load('label.npy')

print("Initializing SVM...")
from sklearn.svm import SVC
#TODO different kernel
#clf = SVC(kernel='precomputed')
clf = SVC()
clf.fit(data, label)

print("Doing cross validation...")
from sklearn import metrics

from sklearn.model_selection import cross_val_score
# 5-fold cross validation
scores = cross_val_score(clf, data,label,cv=5)
print(scores)
