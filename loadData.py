import time
import numpy as np
import os
import cv2
# TensorFlow and tf.keras
import tensorflow as tf

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
            # print("empty", label, image_file)
            continue

        labels.append(label)  # Append the not empty label to the labels list
        images.append(image)  # Append the loaded image to the images list
    
    # Convert lists to TensorFlow Tensors
    images_tensor = tf.convert_to_tensor(np.array(images))
    labels_tensor = tf.convert_to_tensor(np.array(labels))
    
    return images_tensor, labels_tensor

def reduce_reflection(image):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    L, A, B = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_L = clahe.apply(L)
    
    # Merge enhanced L channel with original A and B channels
    enhanced_lab = cv2.merge([enhanced_L, A, B])
    
    # Convert back to BGR color space
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return result

def remove_shadows(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to uint8 type
    gray = np.uint8(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute the Otsu's threshold
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Invert the thresholded image
    thresholded = 255 - thresholded
    
    # Use morphology to remove small noise and fill in gaps
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    
    # Smooth the edges using dilation
    dilated = cv2.dilate(opened, kernel, iterations=2)
    
    # Replace shadow pixels with non-shadow pixels from the original image
    result = image.copy()
    result[dilated == 255] = image[dilated == 255]
    
    return result

def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        # Convert image to numpy array
        image = np.array(image)

        #reduce reflection
        image = reduce_reflection(image)

        image = remove_shadows(image)

        # # Apply Gaussian blur to reduce noise
        # blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # # Convert image to grayscale
        # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # enhanced = clahe.apply(gray)

        # # Convert image back to BGR (if necessary)
        # enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Normalize pixel values to be between 0 and 1
        enhanced = image.astype(np.float32) / 255.0

        # Convert numpy array to TensorFlow tensor
        enhanced_tensor = tf.convert_to_tensor(enhanced)

        preprocessed_images.append(enhanced_tensor)

    return preprocessed_images



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

# Define class names
class_names = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold', 'Target Spot', 'Black Spot']




p_train_i = preprocess_images(train_images)
p_test_i = preprocess_images(test_images)

# Visualize training images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(p_train_i[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Segment training and testing images
segmented_train_images = segment_images(p_train_i)
segmented_test_images = segment_images(p_test_i)


# Visualize segmented images
plt.figure(figsize=(10, 10))
for i in range(min(len(segmented_train_images), 25)):
    plt.subplot(5, 5, i+1)
    plt.imshow(segmented_train_images[i])
    plt.axis('off')
plt.show()



# # train model

# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(640, 640, 3)),
#     MaxPooling2D((2, 2)),
#     Flatten(input_shape=(640,640,3)),
#     Dense(128, activation='relu'),
#     Dense(7, activation='softmax')
# ])

# """
#     compile the model using:
#     for optimization use adam optimizer  'adam'
#     for loss use sparse categorcil crossentropy with logits use the object from tensorflow
#     tf.keras.losses.SparseCategoricalCrossentropy pass logits as true
#     for metrics use accuracy
# """
# model.compile(optimizer= 'adam',loss= 'sparse_categorical_crossentropy'   ,metrics = ['accuracy'] )


# model.fit(train_images, train_labels, batch_size=20, epochs=8)
