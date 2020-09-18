import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os


def create_camera_model():
    loss_func = tf.keras.losses.BinaryCrossentropy()
    optimizer_visual = tf.keras.optimizers.Adam(lr=0.001)

    class VisualAnalyzer(tf.keras.Model):

        def __init__(self):
            super(VisualAnalyzer, self).__init__()
            self.c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
            self.m1 = tf.keras.layers.MaxPooling2D((2, 2))
            self.c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
            self.m2 = tf.keras.layers.MaxPooling2D((2, 2))
            self.c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
            self.f1 = tf.keras.layers.Flatten()
            self.d1 = tf.keras.layers.Dense(128, activation="relu")
            self.classification = tf.keras.layers.Dense(100, activation="softmax")
    
        def call(self, inputs):
            x = self.c1(inputs)
            x = self.m1(x)
            x = self.c2(x)
            x = self.m2(x)
            x = self.c3(x)
            x = self.f1(x)
            x = self.d1(x)
            return self.classification(x)

    visual_analyzer = VisualAnalyzer()
    visual_analyzer.compile(optimizer = optimizer_visual, loss = loss_func, metrics=["accuracy"])
    
    return visual_analyzer

def create_photographer_model(outdims):
    optimizer_data = tf.keras.optimizers.Adam(lr=0.001)
    loss_func = tf.keras.losses.MeanSquaredError()
    
    class DataDecider(tf.keras.Model):
    
        def __init__(self):
            super(DataDecider, self).__init__()
            self.mnet = tf.keras.applications.InceptionV3(input_shape=(32,32,3), include_top=True, weights='imagenet', classes=outdims)
    
        def call(self, inputs):
            return self.mnet(inputs)
    
    data_decider = DataDecider()
    data_decider.compile(optimizer = optimizer_data, loss = loss_func, metrics=["accuracy"])
    
    return data_decider

def get_losses():
    return tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.MeanSquaredError()


