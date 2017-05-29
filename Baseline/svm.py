
from sklearn import svm
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
import time
import os
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC

dataset_file = r'/Users/watermelon/Desktop/Courses/CS249Sun/prj/data/small'
#dataset_file = '/home/ubuntu/small'
from tflearn.data_utils import image_preloader
data = []
label = []

X,Y = image_preloader(dataset_file, image_shape=(300, 300), mode='folder', categorical_labels=True, normalize=True)
for i in range(X.__len__()):
    data.append(X.__getitem__(i).reshape(270000,1))
    label.append((Y.__getitem__(i)).tolist().index(1))

data = np.array(data)
data=data.reshape(data.shape[0],data.shape[1])
label = np.array(label)


#TODO different kernel
#clf = SVC(kernel='precomputed')
clf = SVC()
clf.fit(data, label)

# 5-fold cross validation
cross_validation.cross_val_score(clf,data,label,cv=5)
print (clf.score(data,label))
