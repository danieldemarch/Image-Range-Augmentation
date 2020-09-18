import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

def gen_dataset_cifar():

    cifar10tr, cifar10te = tfds.load('cifar100', split=['train', 'test'])
    
    npcifartr = tfds.as_numpy(cifar10tr)
    npcifarte = tfds.as_numpy(cifar10te)
    
    cifarlisttr = list(npcifartr)
    print(len(cifarlisttr))
    cifarlistte = list(npcifarte)
    print(len(cifarlistte))
    
    images_train = []
    labels_train = []
    coarselabels_train = []
    for item in cifarlisttr:
      images_train.append(item["image"])
      labels_train.append(item["label"])
    
    images_test = []
    labels_test = []
    coarselabels_test = []
    for item in cifarlistte:
      images_test.append(item["image"])
      labels_test.append(item["label"])
    
    train_images = np.array(images_train)
    train_labels = np.array(labels_train)
    
    truelabels = np.zeros((train_labels.size, 100))
    truelabels[np.arange(train_labels.size),train_labels] = 1
    
    train_labels = truelabels
    
    test_images = np.array(images_test)
    test_labels = np.array(labels_test)
    
    truelabels = np.zeros((test_labels.size, 100))
    truelabels[np.arange(test_labels.size),test_labels] = 1
    
    test_labels = truelabels
    
    train_labels = train_labels.astype("float64")
    test_labels = test_labels.astype("float64")
    
    return train_images, train_labels, test_images, test_labels








