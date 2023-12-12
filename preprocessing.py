import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras
import pickle
import random

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def grayscale(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

def filter_classes(cifar, data_dict, classes_to_keep, label_names):
    """
    Filter the data to keep only the classes in classes_to_keep
    """
    data = data_dict[b'data']

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




