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

def fit_camera_model_range(epochs, camera_model, pure_images, pure_labels, amount_to_corrupt, batch_size):
      for i in range(epochs):
          gc.collect()
          print("epoch", i)
          camera_in = gen_corrupted_data(pure_images, amount_to_corrupt, batch_size)
          camera_model.fit(x = camera_in, y = pure_labels, batch_size=batch_size, epochs=1)
  
def fit_photographer_model_range(epochs, photographer_model, photographer_in, photographer_target, batch_size):
  for i in range(epochs): 
      gc.collect()
      print("epoch", i)
                  
      photographer_model.fit(x = photographer_in, y = photographer_target, batch_size=batch_size, epochs=1)
      
      
def fit_both_range(epochs, photographer_model, batch_size, camera_model, pure_images, pure_labels, amount_to_corrupt):
    for k in range(epochs):
        corruption = gen_corrupted_data(pure_images, amount_to_corrupt, len(pure_images)//batch_size)
        
        blight = tf.reshape(corruption, (50000, 32, 32, 3))
        
        to_correct = photographer_model.predict(blight)
            
        restored = np.zeros((corruption.shape))
        
        for i in range(len(to_correct)):
            restored[i] = purify_images(corruption[i], pure_images[i], [np.argmin(to_correct[i])])
                
        restored = restored.astype("float64")
        
        camera_model.fit(x = restored, y = pure_labels, batch_size=batch_size, epochs=1)
    
        return blight, restored
      
def test_range(test_images, test_labels, photographer_model, camera_model, amount_to_corrupt):
    range_data = np.zeros((test_images.shape))
    
    for i in range(len(test_images)):
        range_data[i] = gen_corrupted_data(test_images[i], 2, 1)
            
    photo_finish = photographer_model.predict(range_data)
        
    min_losses = np.zeros((len(test_images)))
    
    for i in range(len(test_images)):
        min_losses[i] = np.argmax(photo_finish[i])
                        
    restore_data = np.zeros((test_images.shape))
    
    for i in range(len(test_images)):
        
        restore_data[i] = purify_images(range_data[i], test_images[i], [min_losses[i]])
                
    predict_nomod = camera_model.predict(range_data)
        
    predict_restore = camera_model.predict(restore_data)
        
    return predict_nomod, predict_restore

def fit_camera_model_block(epochs, camera_model, train_images, train_labels, blockdim, num_blocks, batch_size):
      for i in range(epochs):
          gc.collect()
          print("epoch", i)
          camera_in, positionlist = gen_block_data(train_images, num_blocks, blockdim, batch_size)
          camera_model.fit(x = camera_in, y = train_labels, batch_size=batch_size, epochs=1)
  
def fit_photographer_model_block(epochs, photographer_model, photographer_in, photographer_target, batch_size):
  for i in range(epochs): 
      gc.collect()
      print("epoch", i)
            
      photographer_model.fit(x = photographer_in, y = photographer_target, batch_size=batch_size, epochs=1)
      
def fit_both_block(epochs, photographer_model, batch_size, camera_model, pure_images, pure_labels, num_blocks, blockdim):
    for k in range(epochs):
        blockhead, indices = gen_block_data(pure_images, num_blocks, blockdim, batch_size)
        
        blocked = tf.reshape(blockhead, (50000, 32, 32, 3))
        
        to_correct = photographer_model.predict(blocked)
        
        print(to_correct[0:10])
        
        print("made predicts")
        
        restored = np.zeros((blockhead.shape))
        
        for i in range(len(to_correct)):
            
            currentblock = i//batch_size
            
            currentindex = indices[currentblock]
            
            restored[i] = restore_block(pure_images[i], blockhead[i], currentindex[0, np.argmin(to_correct[i])], currentindex[1, np.argmin(to_correct[i])], blockdim)
                
        restored = restored.astype("float64")
        
        camera_model.fit(x = restored, y = pure_labels, batch_size=batch_size, epochs=1)
    
        return blockhead, restored
        
def test_block(test_images, test_labels, photographer_model, camera_model, num_blocks, blockdim):
    
    block_data, indices = gen_block_data(test_images, num_blocks, blockdim, 1)
            
    photo_finish = photographer_model.predict(block_data)
        
    min_losses = np.zeros((len(test_images))).astype(int)
    
    for i in range(len(test_images)):
        min_losses[i] = np.argmax(photo_finish[i])
                
    restore_data = np.zeros((test_images.shape))
    
    for i in range(len(test_images)):
        
        restore_data[i] = restore_block(test_images[i], block_data[i], indices[i][0,min_losses[i]], indices[i][1,min_losses[i]], blockdim)
                
    predict_nomod = camera_model.predict(block_data)
        
    predict_restore = camera_model.predict(restore_data)
        
    return predict_nomod, predict_restore

def fit_camera_model_sides(epochs, camera_model, pure_images, pure_labels, percent, batch_size):
      for i in range(epochs):
          gc.collect()
          print("epoch", i)
          camera_in, positionlist = gen_lr_data(pure_images, percent, batch_size)
          camera_model.fit(x = camera_in, y = pure_labels, batch_size=batch_size, epochs=1)

def fit_photographer_model_sides(epochs, photographer_model, photographer_in, photographer_target, batch_size):
  for i in range(epochs): 
      gc.collect()
      print("epoch", i)
                  
      photographer_model.fit(x = photographer_in, y = photographer_target, batch_size=batch_size, epochs=1)
        
def fit_both_sides(epochs, photographer_model, batch_size, camera_model, pure_images, pure_labels, percent, percentrestore):
    for k in range(epochs):
        corruption = gen_corrupted_data(pure_images, amount_to_corrupt, len(pure_images)//batch_size)
        
        blight = tf.reshape(corruption, (50000, 32, 32, 3))
        
        to_correct = photographer_model.predict(blight)
            
        restored = np.zeros((corruption.shape))
        
        for i in range(len(to_correct)):
            restored[i] = purify_images(corruption[i], pure_images[i], [np.argmin(to_correct[i])])
                
        restored = restored.astype("float64")
        
        camera_model.fit(x = restored, y = pure_labels, batch_size=batch_size, epochs=1)
    
        return blight, restored
    
def test_sides(test_images, test_labels, photographer_model, camera_model, percent, percentrestore):
    
    restoreamount = int(32*percentrestore)
    
    side_data, indices = gen_lr_data(test_images, percent, 1)
            
    photo_finish = photographer_model.predict(side_data)
        
    min_losses = np.zeros((len(test_images))).astype(int)
    
    for i in range(len(test_images)):
        min_losses[i] = np.argmax(photo_finish[i])
                
    restore_data = np.zeros((test_images.shape))
    
    for i in range(len(test_images)):
        
        restore_data[i] = restore_left_to_right(test_images[i].reshape(1,32,32,3), side_data[i].reshape(1,32,32,3), indices[i][0], indices[i][1], min_losses[i], restoreamount-min_losses[i])
                
    predict_nomod = camera_model.predict(side_data)
        
    predict_restore = camera_model.predict(restore_data)
        
    return predict_nomod, predict_restore
        
        
        
        
        
        
        
        
        
    
    
      
      
      
      
      
      