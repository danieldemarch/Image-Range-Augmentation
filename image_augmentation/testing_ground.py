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







