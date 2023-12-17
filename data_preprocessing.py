import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow.keras
import pickle
import random
import keras
import requests
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from PIL import Image
from keras import models, layers
from keras.datasets import cifar10, cifar100

# Load data from CIFAR10 and CIFAR100
(cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()
(cifar100_x_train, cifar100_y_train), (cifar100_x_test, cifar100_y_test) = cifar100.load_data()

# filter out classes from CIFAR10 and CIFAR100
# 1, 2, 3, 4, 5, 7, 9                                                               = automobile, bird, cat, deer, dog, horse, and truck
# 2, 8, 11, 13, 19, 34, 35, 41, 46, 47, 48, 52, 56, 58, 59, 65, 80, 89, 90, 96, 98  = cattle, fox, baby, boy, girl, man, woman, rabbit, squirrel, trees(superclass), bicycle, bus, motorcycle, pickup truck, train, lawn-mower and tractor
cifar10_classes = [1, 2, 3, 4, 5, 7, 9]
cifar100_classes = [2, 8, 11, 13, 19, 34, 35, 41, 46, 47, 48, 52, 56, 58, 59, 65, 80, 89, 90, 96, 98]

# Gray scale filter
def grayscale_filter(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

# Gaussian filter with size 3x3 and sigma 0
def gaussian_filter(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img