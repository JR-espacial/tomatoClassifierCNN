import numpy as np
import os
import cv2

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
        label = []
        for line in lines:
            data = line.strip().split(' ')
            label.append({
                'class': int(data[0]),
                'x_center': float(data[1]),
                'y_center': float(data[2]),
                'width': float(data[3]),
                'height': float(data[4])
            })
        # If the label is empty, skip the iteration
        if not(len(label) > 0):
            print("empty", label, image_file)
            continue

        labels.append(label)  # Append the not empty label to the labels list
        images.append(image)  # Append the loaded image to the images list
    
    return np.array(images), np.array(labels, dtype=object)

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


# notEmptytrain = 0
# notEmptytest = 0
# for label in train_labels:
#     if len(label) > 0:
#         # print( label[0]['class'])
#         notEmptytrain  += 1
#     else:
#         print("empty", label)
# for label in test_labels:
#     if len(label) > 0:
#         # print( label[0]['class'])
#         notEmptytest  += 1



# print(notEmptytrain)
# print(notEmptytest )