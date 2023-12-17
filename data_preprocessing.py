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

# Scale down image
def scale_down_image(img):

    return img / 255

# Reshape the image to 32x32x1 for RGB 
def reshape(img):

    return img.reshape(img.shape[0], 32, 32, 1)

# Filter CIFAR10 and CIFA100 data with the classes we want
def filter_class_cifar10(cifar10_x_train, cifar10_y_train, cifar10_x_test, cifar10_y_test):

    # Flatten the array
    cifar10_x_train_filtered = cifar10_x_train[np.isin(cifar10_y_train, cifar10_classes).flatten()]
    cifar10_y_train_filtered = cifar10_y_train[np.isin(cifar10_y_train, cifar10_classes).flatten()]
    cifar10_x_test_filtered = cifar10_x_test[np.isin(cifar10_y_test, cifar10_classes).flatten()]
    cifar10_y_test_filtered = cifar10_y_test[np.isin(cifar10_y_test, cifar10_classes).flatten()]
    
    return cifar10_x_train_filtered, cifar10_y_train_filtered, cifar10_x_test_filtered, cifar10_y_test_filtered

def filter_class_cifar100(cifar100_x_train, cifar100_y_train, cifar100_x_test, cifar100_y_test):

    # Flatten the array
    cifar100_x_train_filtered = cifar100_x_train[np.isin(cifar100_y_train, cifar100_classes).flatten()]
    cifar100_y_train_filtered = cifar100_y_train[np.isin(cifar100_y_train, cifar100_classes).flatten()]
    cifar100_x_test_filtered = cifar100_x_test[np.isin(cifar100_y_test, cifar100_classes).flatten()]
    cifar100_y_test_filtered = cifar100_y_test[np.isin(cifar100_y_test, cifar100_classes).flatten()]
    
    return cifar100_x_train_filtered, cifar100_y_train_filtered, cifar100_x_test_filtered, cifar100_y_test_filtered