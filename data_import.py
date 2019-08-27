#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""SPEECH RECOGNITION KAGGLE COMPETITION"""
"""
LABELS:
0 - NO
1 - YES
2 - ON
3 - OFF
4 - DOWN
5 - UP
6 - LEFT
7 - RIGHT
8 - GO
9 - WOW
"""


# In[2]:


import numpy as np
from scipy.io import wavfile
import os
from tqdm import tqdm
import functions as f

# In[3]:


label_names = ['no', 'yes', 'on', 'off', 'down', 'up', 'left', 'right', 'go', 'wow']



# In[5]:


import numpy as np
import functions as f


# In[6]:


X_train, y_train, X_test, y_test, fs = f.import_data(100, 10, label_names=label_names)


# In[7]:


print('X_train: ', X_train.shape[0], X_train.shape[1])
print('y_train', y_train.shape[0])
print('X_test', X_test.shape[0], X_test.shape[1])
print('y_test', y_test.shape[0])


# In[8]:


"""execute this code!!!"""
np.savez_compressed('data100-10.npz',
                    X_train = X_train,
                    y_train = y_train,
                    X_test = X_test,
                    y_test = y_test)

