#!/usr/bin/env python
# coding: utf-8

# In[276]:


import os
import glob
import json
import pandas as pd
import random

import sktime.datatypes
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


# In[295]:


label_dict = {}
with open('collection_scripts/top-1k-curated') as label_file:
    lines = label_file.readlines()
    for i, line in enumerate(lines):
        label_dict[line.strip()] = i


# In[296]:


def load_data(data_dir, label_dict):
    data = []
    labels = []
    path = os.path.join(data_dir, '**/*.json')
    for filename in glob.glob(path, recursive=True):
        with open(filename, 'r') as infile:
            file_data = json.load(infile)

            label = file_data['label']
            times = file_data['time']
            sizes = file_data['size']
            
            series = dict(zip(times, sizes))
            data.append(series)
            labels.append(label_dict[label])  # index of url in list

    return data, labels


# In[297]:


def make_dataframe(data):
    cols = ['instances', 'timepoints', 'packet_size']
    
    # make a list of dataframes where each frame has rows (index, time, size) for row index in data
    Xlist = [
        pd.DataFrame(
            [ [i, time, size] for time, size in series.items() ],
            columns=cols
        ) for i, series in enumerate(data)
    ]
    
    # convert to sktime panel mtype
    X = datatypes.convert_to(Xlist, to_type='pd-multiindex')
    
    return X


# In[298]:


data, labels = load_data('collection_scripts/data/processed/', label_dict)


# In[301]:


random.seed(10)
zipped = list(zip(data, labels))
random.shuffle(zipped)
data, labels = zip(*zipped)


# In[302]:


X_train, y_train = make_dataframe(data[0:9000]), pd.Series(labels[0:9000])
X_test, y_test = make_dataframe(data[9000:]), pd.Series(labels[9000:])


# In[303]:


x = plt.hist([y_train, y_test], bins=[i for i in range(50)])
plt.show()


# In[ ]:


clf = PaddingTransformer() * RocketClassifier()
clf.fit(X_train, y_train)


# In[294]:


pred = clf.predict(X_test)
accuracy_score(pred, y_test)


# In[304]:


y_test


# In[ ]:




