import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

import gc

from image_utils import *
from data_loading import *
from modeling_agency import *
from model_utils import *

train_images, train_labels, test_images, test_labels = gen_dataset_cifar()

train_images, train_labels, test_images, test_labels = train_images.astype("float64"), train_labels.astype("float64"), test_images.astype("float64"), test_labels.astype("float64")
    
camera_model = create_camera_model()
batch_size = 500
amount_to_corrupt = 0
epochs = 60

rightlist = []

for i in range(1,3):
    amount_to_corrupt = i
    camera_model = create_camera_model()
    for j in range(epochs):
        dataset = gen_corrupted_data(train_images, amount_to_corrupt, 2)
        
        camera_model.fit(dataset, train_labels, batch_size = 1000)
    
    predicts = camera_model.predict(test_images)
    right = 0
    for j in range(len(test_labels)):
        if(np.argmax(test_labels[i]) == np.argmax(predicts[i])):            
            right+=1
    rightlist.append(right)
    
print(rightlist)








































