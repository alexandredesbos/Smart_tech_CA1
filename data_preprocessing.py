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
# 19, 34, 2, 11, 19, 35, 46, 98, 46, 65, 80, 47, 52, 56, 8, 13, 48, 89, 90, 41, 58  = cattle, fox, baby, boy, girl, man, woman, rabbit, squirrel, trees(superclass), bicycle, bus, motorcycle, pickup truck, train, lawn-mower and tractor
cifar10_classes = [1, 2, 3, 4, 5, 7, 9]
cifar100_classes = [19, 34, 2, 11, 19, 35, 46, 98, 46, 65, 80, 47, 52, 56, 8, 13, 48, 89, 90, 41, 58]

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

# Build the model
def leNet_model():
  model = Sequential()
  # Convolutional layer with filters, kernel size (5, 5), and ReLU activation
  model.add(Conv2D(64, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  # Max pooling layer with pool size (2, 2)
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Convolutional layer with filters, kernel size (3, 3), and ReLU activation
  model.add(Conv2D(64, (3, 3), activation='relu'))
  # Max pooling layer with pool size (2, 2)
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Flatten the output for dense layers
  model.add(Flatten())
  # Dense layer with neurons and ReLU activation
  model.add(Dense(50, activation='relu'))
  # Dropout layer with a dropout rate of 0.5
  model.add(Dropout(0.5))
  # Output layer with 'num_classes' neurons and softmax activation
  model.add(Dense(num_of_data, activation='softmax'))
  # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
  model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model


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

        # Set the title of the subplot to the current class at middle of the row
        axes[j, 1].set_title(f'Class: {class_label}', size='small')

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

def evaluate_model(model, x_test, y_test):

    # Evaluate the model and print the test loss and test accuracy
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Model Score:", score[0])
    print("Model Accuracy:", score[1])

def analyze_model(history):

    # Plot the training accuracy and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # Set the legend of the plot to the upper left corner
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_loss(history):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def display_first_image_size(file_name, x_train):
    print(f"The size of the first image in {file_name} is:")
    print(x_train[0].shape)

def count_label_per_image(file_name, y_train):
    label_counts = np.sum(y_train, axis=1)
    print(f"{file_name} Counts of the label:", label_counts)
    return label_counts

def display_unique_classes(file_name, y_train):
    unique_classes = np.unique(y_train)
    print(f"{file_name} Unique Classes:", unique_classes)
    return unique_classes

# Display shapes of CIFAR10 training and testing data
def display_data_shapes(file_name, x_train, y_train, x_test, y_test):
    print(f"\n{file_name} Training Data: X Shape - {x_train.shape}, Y Shape - {y_train.shape}")
    print(f"{file_name} Testing Data: X Shape - {x_test.shape}, Y Shape - {y_test.shape}")


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

display_first_image_size("CIFAR-10", cifar10_x_train)
# Output
# The size of the first image in CIFAR-10 is:
# (32, 32, 3)

display_first_image_size("CIFAR-100", cifar100_x_train)
# Output
# The size of the first image in CIFAR-100 is:
# (32, 32, 3)


cifar10_label_counts = count_label_per_image("CIFAR-10", cifar10_y_train)
# Output
# CIFAR-10 Counts of the label: [6 9 9 ... 9 1 1]

cifar100_label_counts = count_label_per_image("CIFAR-100", cifar100_y_train)
# Output
# CIFAR-100 Counts of the label: [19 29  0 ...  3  7 73]

cifar10_unique_classes = display_unique_classes("CIFAR-10", cifar10_y_train)
# Output
# CIFAR-10 Unique Classes: [0 1 2 3 4 5 6 7 8 9]

cifar100_unique_classes = display_unique_classes("CIFAR-100", cifar100_y_train)
# Output
# CIFAR-100 Unique Classes: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#                             24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
#                             48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
#                             72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
#                             96 97 98 99]


display_data_shapes("CIFAR-10", cifar10_x_train, cifar10_y_train, cifar10_x_test, cifar10_y_test)
# Output
# CIFAR-10 Training Data: X Shape - (50000, 32, 32, 3), Y Shape - (50000, 1)
# CIFAR-10 Testing Data: X Shape - (10000, 32, 32, 3), Y Shape - (10000, 1)

display_data_shapes("CIFAR-100", cifar100_x_train, cifar100_y_train, cifar100_x_test, cifar100_y_test)
# Output
# CIFAR-100 Training Data: X Shape - (50000, 32, 32, 3), Y Shape - (50000, 1)
# CIFAR-100 Testing Data: X Shape - (10000, 32, 32, 3), Y Shape - (10000, 1)

# show 10 random images from CIFAR10
plot_for_cifar(cifar10_x_train, cifar10_y_train, 10)
# show 10 random images from CIFAR100
plot_for_cifar(cifar100_x_train, cifar100_y_train, 10)

cifar10_x_train_filtered, cifar10_y_train_filtered, cifar10_x_test_filtered, cifar10_y_test_filtered = filter_class_cifar10(cifar10_x_train, cifar10_y_train, cifar10_x_test, cifar10_y_test)
cifar100_x_train_filtered, cifar100_y_train_filtered, cifar100_x_test_filtered, cifar100_y_test_filtered = filter_class_cifar100(cifar100_x_train, cifar100_y_train, cifar100_x_test, cifar100_y_test)

# Plot filtered images from CIFAR10 (1.automobile, 2.bird, 3.cat, 4.deer, 5.dog, 7.horse, and 9.truck)
plot_filtered_for_cifar(cifar10_x_train_filtered, cifar10_y_train_filtered, cifar10_classes, 5)
# Plot filtered images from CIFAR100 19, 34, 2, 11, 19, 35, 46, 98, 46, 65, 80, 47, 52, 56, 8, 13, 48, 89, 90, 41, 58  = cattle, fox, baby, boy, girl, man, woman, rabbit, squirrel, trees(superclass), bicycle, bus, motorcycle, pickup truck, train, lawn-mower and tractor
plot_filtered_for_cifar(cifar100_x_train_filtered, cifar100_y_train_filtered, cifar100_classes, 5)

# Combine CIFAR10 and CIFAR100 data using the combine_cifar function
x_train, y_train, x_test, y_test = combine_cifar(
    cifar10_x_train_filtered, cifar10_y_train_filtered,
    cifar10_x_test_filtered, cifar10_y_test_filtered,
    cifar100_x_train_filtered, cifar100_y_train_filtered,
    cifar100_x_test_filtered, cifar100_y_test_filtered
)

# Get the unique classes in the combined dataset (Data Exploration)
combined_classes = np.unique(np.concatenate((y_train, y_test)))

# Display the shape of the combined dataset
print("\nCombined Train Shape:", x_train.shape, y_train.shape)
print("Combined Test Shape:", x_test.shape, y_test.shape)

# Display the unique combined classes
print("Combined Classes:", combined_classes)

# Display images from combined dataset
combined_cifar = show_combined_cifar(x_train, y_train, combined_classes, 5)


# Plot the combined classes (Data Exploration)
print("\nClasses:", combined_cifar)
num_of_data = len(combined_cifar)
plt.figure(figsize=(12, 4))
plt.bar(range(num_of_data), combined_cifar)
plt.title("Distribution of Images Across Combined Classes")
plt.xlabel("Classes")
plt.ylabel("Number of Data")
plt.show()


# Plot preprocessed image
x_train_preprocessed = np.array(list(map(preprocess, x_train)))
x_test_preprocessed = np.array(list(map(preprocess, x_test)))

# Randomly select an image from the training set
random_index = np.random.choice(len(x_train))
# Display the preprocessed image
plt.imshow(x_train[random_index])
plt.title("Preprocessed Image x_train")
plt.axis("off")


# Reshape data 
x_train = reshape(x_train_preprocessed)
x_test = reshape(x_test_preprocessed)

print("\nX Train shape: ", x_train.shape)
print("X Test shape: ", x_test.shape)


# One hot encoding
y_train = to_categorical(y_train, num_of_data)
y_test = to_categorical(y_test, num_of_data)

# Create the LeNet model with the default parameters
model = leNet_model()
print(model.summary())

history = model.fit(x_train, y_train, epochs=10, validation_split=0.1, batch_size=40, verbose=1, shuffle=1)

# Evaluate the model on the test set
evaluate_model(model, x_test, y_test)

# Plot the training accuracy and validation accuracy
analyze_model(history)

# Plot the training loss and validation loss
plot_loss(history)