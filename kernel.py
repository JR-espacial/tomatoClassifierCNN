#kernel
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from loadData import load_data
from loadData import preprocess_images
from model import train_model
from tests import test_model

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

# Load training data
print("Loading data...")
train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)


#Shape of the data
print("Shape of training images:", train_images.shape)
print("Shape of training labels:", train_labels.shape)
print("Shape of testing images:", test_images.shape)
print("Shape of testing labels:", test_labels.shape)

print("---------------------------------")



# Define class names (7)
class_names = [ 'Bacterial Spot', 'Early Blight', 
                'Healthy', 'Late Blight', 'Leaf Mold', 
                'Target Spot', 'Black Spot']


p_train_i = preprocess_images(train_images,128,128)
p_test_i = preprocess_images(test_images,128,128)



#if the user chooses to show the images
execute("show original data",show_dataset, train_images,train_labels,class_names,"Training Data")
execute("show proceseed data",show_dataset, p_train_i,train_labels,class_names,"Training Data")




# prepare data to train
images_tensor = tf.convert_to_tensor(np.array(p_train_i))

#Train the model
execute("train the model",train_model,images_tensor,train_labels)



# Load the model
model = tf.keras.models.load_model('jorge.h5')
#Testing the model
images_tensor = tf.convert_to_tensor(np.array(p_test_i))

execute("test the model",test_model,images_tensor,test_labels,model)
