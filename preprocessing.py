import numpy as np
import matplotlib.pyplot as plt
import cv2
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

#load the data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#convert the images to grayscale
def grayscale(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

#resize the images to 32x32
def filter_classes(cifar, data_dict, classes_to_keep, label_names):
    """
    Filter the data to keep only the classes in classes_to_keep
    """
    data = data_dict[b'data']

    #reshape the data to the format (num_samples, depth, height, width)
    if cifar == 10:
        labels = np.array(data_dict[b'labels'])
    elif cifar == 100:
        labels = np.array(data_dict[b'fine_labels'])

    # here we convert the labels to the indices of the classes we want to keep
    classes_to_keep = [label_names.index(class_name) for class_name in classes_to_keep]

    # We keep only the label indices that are in classes_to_keep
    mask = np.isin(labels, classes_to_keep)
    filtered_data = data[mask]
    filtered_labels = labels[mask]
    return filtered_data, filtered_labels

#normalize the data to values between 0 and 1
def normalize_data(data):

    return data.astype('float32') / 255

#plot the images
def plot_samples(data, labels, label_names):
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        k = np.random.randint(0, data.shape[0]) 
        image = data[k]
        image = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))
        plt.imshow(image)
        plt.title(label_names[labels[k]])
    plt.show()


# Load CIFAR-10 data
cifar_10 = unpickle('cifar-10-batches-py/data_batch_1')
cifar_10_labels = [ "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" ]

cifar10_classes_to_keep = ['automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck']

x_train_cifar_10, y_train_cifar_10 = filter_classes(10, cifar_10, cifar10_classes_to_keep, cifar_10_labels)

print(x_train_cifar_10.shape)

print(y_train_cifar_10.shape)


print("sample of the CIFAR-10 dataset")
plot_samples(x_train_cifar_10, y_train_cifar_10, cifar_10_labels)


# Load CIFAR-100 data
cifar_100 = unpickle('cifar-100-python/train')

# List of classes in CIFAR-100
cifar_100_labels = [ "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm" ]

# Classes to keep
cifar100_classes_to_keep = ["cattle", "fox", "baby", "boy", "girl", "man", "woman", "rabbit", "squirrel", "maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree", "bicycle", "bus", "motorcycle", "pickup_truck", "train", "lawn_mower", "tractor"]

x_train_cifar_100, y_train_cifar_100 = filter_classes(100, cifar_100, cifar100_classes_to_keep, cifar_100_labels)

print("sample of the CIFAR-100 dataset")
plot_samples(x_train_cifar_100, y_train_cifar_100, cifar_100_labels)

# add 10 to the labels to avoid confusion with the labels from CIFAR-10
y_train_cifar_100 = y_train_cifar_100 + 10

# Combining the two datasets
x_train = np.concatenate((x_train_cifar_10, x_train_cifar_100))
y_train = np.concatenate((y_train_cifar_10, y_train_cifar_100))

print(y_train[random.randint(0, y_train.shape[0])])

combined_all_labels = cifar_10_labels + cifar_100_labels
combined_labels = cifar10_classes_to_keep + cifar100_classes_to_keep

print("Sample of the combined dataset")
plot_samples(x_train, y_train, combined_all_labels)

#make it similar scale
x_train = normalize_data(x_train)
y_train = normalize_data(y_train)

#convert the labels to one-hot encoding
y_train = to_categorical(y_train)

#print the shapes of the data
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

#32x32 images with RGB
print("Image size:", x_train.shape[1:])
#Number of classes: 10
print("Number of classes:", y_train.shape[0])

#here we split the data into training and validation sets by using cifar10 data
train_data = unpickle('cifar-10-batches-py/data_batch_1') 
validation_data = unpickle('cifar-10-batches-py/data_batch_2')
test_data = unpickle('cifar-10-batches-py/test_batch')

#features for training, validation and testing data
X_train, y_train = train_data[b'data'], train_data[b'labels']
X_val, y_val = validation_data[b'data'], validation_data[b'labels']
X_test, y_test = test_data[b'data'], test_data[b'labels']

#normalize the data for x_train, x_val and x_test
X_train = normalize_data(X_train)
X_val = normalize_data(X_val)
X_test = normalize_data(X_train) 

# Reshape the data to (num_samples, depth, height, width)
X_train = X_train.reshape(-1, 32, 32, 3)
X_val = X_val.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)

#convert the labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Build a CNN model using activation function relu
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Train the model for 5 epochs (will do with more epochs later)
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))