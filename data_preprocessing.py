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

# Plotting CIFAR10 and CIFAR100 data
def plot_for_cifar(x_train, y_train, num_of_img):

    fig, axes = plt.subplots(1, num_of_img, figsize=(10, 10))

    for i in range(num_of_img):
        #get random image from training set
        index = np.random.randint(0, x_train.shape[0])
        axes[i].set_title("Class: " + str(y_train[index][0]))
        axes[i].axis('off')
        axes[i].imshow(x_train[index])
    plt.show()


# Filter out the plot from CIFAR10 and CIFAR100
def plot_filtered_for_cifar(x, y, class_labels, num_of_img):

    # Set the seed to get consistent results
    np.random.seed(0)

    # Get the number of classes
    num_of_class = len(class_labels)

    # Make a subplot with a grid of size with the number of classes and number of images
    fig, axes = plt.subplots(num_of_class, num_of_img, figsize=(2 * num_of_img, 2 * num_of_class))
    
    # Adjust spacing between subplots
    plt.tight_layout(pad=3.0, h_pad=1.0, w_pad=0.5)
    
    # Loop through each class
    for i, class_label in enumerate(class_labels):
        # Get the indices of the current class
        indices = np.where(y == class_label)[0]
        # get random image from training set
        random_indices = np.random.choice(indices, num_of_img, replace=False)
        
        # Loop through each image in the current class
        for j, idx in enumerate(random_indices):
            # Plot the image
            axes[i][j].imshow(x[idx])
            axes[i][j].axis('off')
            axes[i][j].set_title(f'Class: {class_label}' if j == 0 else '', size='large')

    plt.show()

# Combine the two CIFAR-10 and CIFAR100 data
def combine_cifar(cifar10_x_train, cifar10_y_train, cifar10_x_test, cifar10_y_test, cifar100_x_train, cifar100_y_train, cifar100_x_test, cifar100_y_test):
    
    x_train = np.concatenate((cifar10_x_train, cifar100_x_train), axis=0)
    y_train = np.concatenate((cifar10_y_train, cifar100_y_train), axis=0)
    x_test = np.concatenate((cifar10_x_test, cifar100_x_test), axis=0)
    y_test = np.concatenate((cifar10_y_test, cifar100_y_test), axis=0)
    
    return x_train, y_train, x_test, y_test

# Show the combined CIFAR10 and CIFAR100 data
def show_combined_cifar(x, y, class_labels, num_of_img):

    num_of_class = len(class_labels)
    num_of_data = []

    # Make a subplot with a grid of size with the number of classes and number of images
    fig, axes = plt.subplots(num_of_class, num_of_img, figsize=(2 * num_of_img, 2 * num_of_class))
    plt.tight_layout(pad=3.0, h_pad=1.0, w_pad=0.5)

    # Check if there is only one class
    if num_of_class == 1:

        # If there is only one class then add an extra dimension to the axes
        axes = np.array([axes])

    # Loop
    for j, class_label in enumerate(class_labels):

        # Find indices where the label matches the current class
        indices = np.where(y.flatten() == class_label)[0]

        # Select random indices from the indices array with the number of images 
        selected_indices = np.random.choice(indices, num_of_img, replace=False)

        # Loop
        for k, index in enumerate(selected_indices):
            # Display the image on the subplot at the current row and column index 
            axes[j, k].imshow(x[index], interpolation='nearest')
            axes[j, k].axis('off')
            num_of_data.append(len(indices))

        # Set the title of the subplot to the current class
        axes[j, -1].set_title(f'Class: {class_label}', size='large')

    # show the plot
    plt.show()

    # Return the number of data points in each class
    return num_of_data


# preprocess the data
def preprocess(img):

    img = grayscale_filter(img)

    # Apply equalization filter
    img = cv2.equalizeHist(img)

    img = gaussian_filter(img)
    
    # Scale down the image to 0-1 range to improve the performance of the model
    img = scale_down_image(img)
    
    return img

# # Show random image from CIFAR10 and grayscale it
# img = random.choice(cifar10_x_train)
# grayscale_random_img = grayscale_filter(img)
# plt.imshow(grayscale_random_img)
# plt.title("Random Image (Grayscale)")
# plt.axis("off")
# plt.show()

# # Show the image in shades of gray
# plt.imshow(grayscale_random_img, cmap='gray')
# plt.title("Random Image with Colormap (Grayscale)")
# plt.axis("off")
# plt.show()

# # Show Plot equalized image
# img_gray_eq = cv2.equalizeHist(grayscale_random_img)
# plt.imshow(img_gray_eq)
# plt.title("Random Image (Equalized)")
# plt.axis("off")
# plt.show()

# # Show the image in shades of gray
# plt.imshow(img_gray_eq, cmap='gray')
# plt.title("Random Equalized Image (Colormap)")
# plt.axis("off")
# plt.show()

# # Show Plot gaussian image
# img_gaussian = gaussian_filter(img)
# plt.imshow(img_gaussian)
# plt.title("Random Image (Gaussian)")
# plt.axis("off")
# plt.show()

# # Show Plot preprocessed image
# img_preprocessed = preprocess(img)
# plt.imshow(img_preprocessed)
# plt.title("Random Image (Preprocessed)")
# plt.axis("off")
# plt.show()

# # Show the image in shades of gray
# plt.imshow(img_preprocessed, cmap='gray')
# plt.title("Random Preprocessed Image (Colormap)")
# plt.axis("off")
# plt.show()

def leNet_model():
    model = Sequential()
    # Add convolutional layer with 100 filters, 3x3 kernel, relu activation function and input shape of 32x32x3
    model.add(Conv2D(100, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    # Add convolutional layer with 100 filters, 3x3 kernel, relu activation function
    model.add(Conv2D(70, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # Add convolutional layer with 100 filters, 3x3 kernel, relu activation function
    model.add(Conv2D(100, (3, 3), activation='relu'))
    model.add(Flatten())
    # Add dense layer with 1000 nodes and relu activation function
    model.add(Dense(1000, activation='relu'))
    # Add dense layer with 500 nodes and relu activation function
    model.add(Dense(10, activation='softmax'))
    # Compile the model with Adam optimizer, categorical crossentropy loss function and accuracy metric with learning rate of 0.001
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print(leNet_model().summary())

# Output of the model
# Show the summary of the model
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 30, 30, 100)       2800

#  max_pooling2d (MaxPooling2  (None, 15, 15, 100)       0
#  D)

#  conv2d_1 (Conv2D)           (None, 13, 13, 70)        63070

#  max_pooling2d_1 (MaxPoolin  (None, 6, 6, 70)          0
#  g2D)

#  conv2d_2 (Conv2D)           (None, 4, 4, 100)         63100

#  flatten (Flatten)           (None, 1600)              0

#  dense (Dense)               (None, 1000)              1601000

#  dense_1 (Dense)             (None, 10)                10010

# =================================================================
# Total params: 1739980 (6.64 MB)
# Trainable params: 1739980 (6.64 MB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________