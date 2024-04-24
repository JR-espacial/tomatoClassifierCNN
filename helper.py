import os
import cv2

def create_folders_class(images,class_names,dataset):
    print("Creating folders for ", dataset)
    # Create folders for each class
    folder = "Jointdata"
    # Create a folder for the joint data
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Create a folder for each class in the joint data folder
    for i in range(7):
        print("Creating folder for class ", class_names[i])
        subfolder = folder + "/" + class_names[i]
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        for j, image in  enumerate(images[i]):
            imgname = subfolder + "/" + str(j)+ dataset + ".jpg"
            print("Saving image ", imgname)
            # Save the image in the corresponding class folder
            cv2.imwrite( imgname, image)


def load_data(data_dir, class_names, dataset):
    # Initialize lists to store images and labels separate by class
    images = [[], [], [], [], [], [], []]
    labels = [[], [], [], [], [], [], []]

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

        labels[label].append(label)  # Append the not empty label to the labels list
        images[label].append(image)  # Append the loaded image to the images list

      
    
    create_folders_class(images,class_names, dataset)
    
    return images, labels




# Directory of training data
train_data_dir = "train"
test_data_dir = "test"
val_data_dir = "valid"

# Load training data
print("Helper...")
class_names = [ 'BacterialSpot', 'EarlyBlight', 
                'Healthy', 'LateBlight', 'LeafMold', 
                'TargetSpot', 'BlackSpot']
print("---------------------------------")
train_images, train_labels = load_data(train_data_dir, class_names, "train")
print("---------------------------------")
test_images, test_labels = load_data(test_data_dir, class_names, "test")
print("---------------------------------")
val_images, val_labels = load_data(val_data_dir, class_names, "valid")

