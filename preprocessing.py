import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar_10 = unpickle('cifar-10-batches-py/data_batch_1')
cifar_100 = unpickle('cifar-100-python/train')

def plot_sample(x, y, axis):
    for i in range(10):
        axis[i].imshow(x[i])
        axis[i].set_title(y[i])
        axis[i].axis('off')
        plt.tight_layout()
    plt.show()

# Plot a sample of 10 images of cifar-10
x = cifar_10[b'data'].reshape(10000, 3, 32, 32).transpose(0,2,3,1)
y = np.array(cifar_10[b'labels'])
axis = plt.subplots(2,5,figsize=(10,5))[1].reshape(-1)
plot_sample(x, y, axis)

# Plot a sample of 10 images of cifar-100
x = cifar_100[b'data'].reshape(50000, 3, 32, 32).transpose(0,2,3,1)
y = np.array(cifar_100[b'fine_labels'])
axis = plt.subplots(2,5,figsize=(10,5))[1].reshape(-1)
plot_sample(x, y, axis)

