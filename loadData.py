#load data.py
import numpy as np
import os
import cv2
# TensorFlow and tf.keras
import tensorflow as tf


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

def load_data2(data_dir, class_names):
    images = []
    labels = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        # Read the image
        img = cv2.imread(img_path)
        # Preprocess the image (if needed)
        # Add the image to the list
        images.append(img)
        # Extract label from the image name
        label = img_name.split('_')[0]  # Assuming the class name is before the first underscore
        # Append the index of the class name in class_names
        if label in class_names:
            labels.append(class_names.index(label))
        else:
            print(f"Warning: '{label}' not found in class_names.")

    return images, labels  

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

def preprocess_images(images, new_width, new_height):
    preprocessed_images = []
    for image in images:
        # Convert image to numpy array
        image = np.array(image)

        # Reduce reflection and remove shadows
        image = reduce_reflection(image)
        image = remove_shadows(image)

        # Resize image
        image_resized = cv2.resize(image, (new_width, new_height))

        # Normalize pixel values to be between 0 and 1
        image_resized = image_resized.astype(np.float32) / 255.0

        # Convert numpy array to TensorFlow tensor
        image_tensor = tf.convert_to_tensor(image_resized)

        preprocessed_images.append(image_tensor)

    return preprocessed_images