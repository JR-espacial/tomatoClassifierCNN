#kernel
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from loadData import load_data
from loadData import preprocess_images
from model import train_model
from tests import test_model

def rebalance_datasets(train_images, train_labels, test_images, test_labels, val_images, val_labels, out_percentages):
    
    #Join all images and labels
    all_images = np.concatenate((train_images, test_images, val_images))
    all_labels = np.concatenate((train_labels, test_labels, val_labels))


    #print how many labels of each class
    for i in range(7):
        print("Number of labels of class", i, ":", np.sum(all_labels == i))

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
train_data_dir = "train"
test_data_dir = "test"
val_data_dir = "valid"

# Load training data
print("Loading data...")
train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)
val_images, val_labels = load_data(val_data_dir)

#Rebalance the data
#percentage of the data to be used in the training, validation and testing
# [train, val, test]
new_data_shape = [0.6, 0.1, 0.3]
train_images, train_labels, test_images, test_labels, val_images, val_labels = rebalance_datasets(train_images, train_labels, test_images, test_labels, val_images, val_labels, new_data_shape)


#Shape of the data
print("Shape of training images:", train_images.shape)
print("Shape of training labels:", train_labels.shape)
print("Shape of validation images:", val_images.shape)
print("Shape of validation labels:", val_labels.shape)
print("Shape of testing images:", test_images.shape)
print("Shape of testing labels:", test_labels.shape)

print("---------------------------------")



# Define class names (7)
class_names = [ 'Bacterial Spot', 'Early Blight', 
                'Healthy', 'Late Blight', 'Leaf Mold', 
                'Target Spot', 'Black Spot']


p_train_i = preprocess_images(train_images,128,128)
p_test_i = preprocess_images(test_images,128,128)
p_val_i = preprocess_images(val_images,128,128)



#if the user chooses to show the images
execute("show original data",show_dataset, train_images,train_labels,class_names,"Training Data")
execute("show proceseed data",show_dataset, p_train_i,train_labels,class_names,"Training Data")




# prepare data to train
images_tensor = tf.convert_to_tensor(np.array(p_train_i))
val_images = tf.convert_to_tensor(np.array(p_val_i))

#Train the model
execute("train the model",train_model,images_tensor,train_labels,val_images,val_labels)



# Load the model
model = tf.keras.models.load_model('jorge.h5')
#Testing the model
images_tensor = tf.convert_to_tensor(np.array(p_test_i))

execute("test the model",test_model,images_tensor,test_labels,model)
