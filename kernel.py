#kernel
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from loadData import load_data
from loadData import load_data2
from loadData import preprocess_images
from model import train_model
from tests import test_model
import random

def shuffle_together(arr1, arr2):
    combined = list(zip(arr1, arr2))
    random.shuffle(combined)
    arr1[:], arr2[:] = zip(*combined)
    return arr1, arr2

def label_distribution(labels, class_names):
    #print(labels)
    #print how many labels of each class
    for i in range(7):
        count = 0
        for label in labels:
            if label == i:
                count += 1
        print("Number of labels of class", i, class_names[i] , ":", count)
            

        

def rebalance_datasets(train_images, train_labels, test_images, test_labels, val_images, val_labels, out_percentages, classes):
    
    #Join all images and labels
    all_images = np.concatenate((train_images, test_images, val_images))
    all_labels = np.concatenate((train_labels, test_labels, val_labels))


    #print how many labels of each class
    for i in range(7):
        print("Number of labels of class", i, class_names[i] , ":", np.sum(all_labels == i))

    #redestructure the data according to the out_percentages
    new_train_images = []
    new_train_labels = []
    new_test_images = []
    new_test_labels = []
    new_val_images = []
    new_val_labels = []

    #out_percentages ex. = [0.5, 0.2, 0.3]
    for i, percentage in  enumerate(out_percentages):
        #train data and labels
        if  i == 0:
            train_limit = int(len(all_images)*percentage)
            new_train_images = all_images[:train_limit]
            new_train_labels = all_labels[:train_limit]
        #val data and labels
        elif i == 1:
            val_limit = train_limit + int(len(all_images)*percentage)
            new_val_images = all_images[train_limit:val_limit]
            new_val_labels = all_labels[train_limit:val_limit]
        #test data and labels
        else:
            new_test_images = all_images[val_limit:]
            new_test_labels = all_labels[val_limit:]
            
    return new_train_images, new_train_labels, new_test_images, new_test_labels, new_val_images, new_val_labels
    

def show_dataset(images,labes,class_names,title):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.suptitle(title)
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labes[i]])
    plt.show()

def execute(function_name, function, *args):
    #prompt the use if he wants to ecexute the function name
    opt = input("Do you want to " + function_name + " ? (y/n)")
    if opt == "y":
        function(*args)
    else:
        print("Function " + function_name + " was not executed")
    

# Directory of training data
train_data_dir = "./SplitData/train"
test_data_dir = "./SplitData/test"
val_data_dir = "./SplitData/val"

# Load training data
# Define class names (7)
class_names = [ 'BacterialSpot', 'EarlyBlight', 
                'Healthy', 'LateBlight', 'LeafMold', 
                'TargetSpot', 'BlackSpot']

print("Loading data...")
train_images, train_labels = load_data2(train_data_dir,class_names)
test_images, test_labels = load_data2(test_data_dir,class_names)
val_images, val_labels = load_data2(val_data_dir,class_names)

#Shuffle the data
train_images, train_labels = shuffle_together(train_images, train_labels)
test_images, test_labels = shuffle_together(test_images, test_labels)
val_images, val_labels = shuffle_together(val_images, val_labels)


#Rebalance the data
#percentage of the data to be used in the training, validation and testing
# [train, val, test]
# new_data_shape = [0.6, 0.1, 0.3]
# train_images, train_labels, test_images, test_labels, val_images, val_labels = rebalance_datasets(train_images, train_labels, test_images, test_labels, val_images, val_labels, new_data_shape, class_names)

#Show the distribution of the labels
print ("Training data distribution")
label_distribution(train_labels,class_names)
print("---------------------------------")
print ("Validation data distribution")
label_distribution(val_labels,class_names)
print("---------------------------------")
print ("Testing data distribution")
label_distribution(test_labels,class_names)
print("---------------------------------")


#Preprocess the images
p_train_i = preprocess_images(train_images,128,128)
p_test_i = preprocess_images(test_images,128,128)
p_val_i = preprocess_images(val_images,128,128)

#if the user chooses to show the images
execute("show original data",show_dataset, train_images,train_labels,class_names,"Training Data")
execute("show proceseed data",show_dataset, p_train_i,train_labels,class_names,"Training Data")


# prepare data to train
images_tensor = tf.convert_to_tensor(np.array(p_train_i))
val_images = tf.convert_to_tensor(np.array(p_val_i))
label_tensor = tf.convert_to_tensor(np.array(train_labels))
val_labels = tf.convert_to_tensor(np.array(val_labels))

#show shape of the data
print("Shape of the data")
print(images_tensor.shape)
print(label_tensor.shape)
print(val_images.shape)
print(val_labels.shape)

# Convert the lists to NumPy arrays
images_tensor = np.array(images_tensor)
train_labels = np.array(train_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)


#Train the model
execute("train the model",train_model,images_tensor,train_labels,val_images,val_labels)

# Load the model
model = tf.keras.models.load_model('jorge.h5')



# Convert test_images and test_labels to NumPy arrays
test_images = np.array(p_test_i)
test_labels = np.array(test_labels)
#Testing the model
execute("test the model",test_model, test_images, test_labels,model)
