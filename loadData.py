import time
import numpy as np
import os
import cv2
# TensorFlow and tf.keras
import tensorflow as tf
import torch

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_dir):
    images = []
    labels = []

    # Iterating over the directory of training/testing images
    for image_file in os.listdir(os.path.join(data_dir, "images")):
        # Load the image
        image_path = os.path.join(data_dir, "images", image_file)
        image = cv2.imread(image_path)
        

        # Load the labels
        label_file = os.path.join(data_dir, "labels", os.path.splitext(image_file)[0] + ".txt")
        with open(label_file, 'r') as f:
            lines = f.readlines()

        label = None
        for line in lines:
            data = line.strip().split(' ')
            label = int(data[0])
        # If the label is empty, skip the iteration
        if label is None:
            print("empty", label, image_file)
            continue

        labels.append(label)  # Append the not empty label to the labels list
        images.append(image)  # Append the loaded image to the images list
    
    # Convert lists to TensorFlow Tensors
    images_tensor = tf.convert_to_tensor(np.array(images))
    labels_tensor = tf.convert_to_tensor(np.array(labels))
    
    return images_tensor, labels_tensor

# Directory of training data
train_data_dir = "train"
test_data_dir = "test"

# Load training data
train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)


# Print the shape of the loaded data
print("Number of training images:", train_images.shape)
print("Number of training labels:", train_labels.shape)
print("Number of testing images:", test_images.shape)
print("Number of testing labels:", test_labels.shape)

print("train_labels", train_labels)


class_names = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold', 'Target Spot', 'Black Spot']


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# Add a color dimension to the images
train_images = tf.reshape(train_images, [train_images.shape[0], 640, 640, 3])
test_images = tf.reshape(test_images, [test_images.shape[0], 640, 640, 3])

# Normalize pixel values to be between 0 and 1
train_images = tf.cast(train_images, tf.float32) / 255.0
test_images = tf.cast(test_images, tf.float32) / 255.0


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

time.sleep(2)


# train model

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

model = Sequential([
    Flatten(input_shape=(640,640,3)),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
])

"""
    compile the model using:
    for optimization use adam optimizer  'adam'
    for loss use sparse categorcil crossentropy with logits use the object from tensorflow
    tf.keras.losses.SparseCategoricalCrossentropy pass logits as true
    for metrics use accuracy
"""
model.compile(optimizer= 'adam',loss= 'sparse_categorical_crossentropy'   ,metrics = ['accuracy'] )


model.fit(train_images, train_labels, batch_size=32, epochs=8)
