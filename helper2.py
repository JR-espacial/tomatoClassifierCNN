import os
import shutil
import random

def split_data(input_dir, output_dir, class_names, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits data into train, validation, and test sets with mixed classes in each set.

    Args:
    - input_dir: Directory containing class folders.
    - output_dir: Directory to save the split data.
    - class_names: List of class names.
    - train_ratio: Ratio of data to allocate for training (default: 0.7).
    - val_ratio: Ratio of data to allocate for validation (default: 0.15).
    - test_ratio: Ratio of data to allocate for testing (default: 0.15).
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Initialize ID counter
    id_counter = 1

    # Iterate over each class
    for class_name in class_names:
        class_dir = os.path.join(input_dir, class_name)
        images = os.listdir(class_dir)
        random.shuffle(images)

        # Calculate split indices
        num_images = len(images)
        num_train = int(train_ratio * num_images)
        num_val = int(val_ratio * num_images)

        # Split data
        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        # Copy images to respective directories with new names
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_dir, f'{class_name}_{id_counter}.jpg')
            shutil.copy(src, dst)
            id_counter += 1

        for img in val_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_dir, f'{class_name}_{id_counter}.jpg')
            shutil.copy(src, dst)
            id_counter += 1

        for img in test_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(test_dir, f'{class_name}_{id_counter}.jpg')
            shutil.copy(src, dst)
            id_counter += 1

# Usage example:
input_dir = './Jointdata/'
output_dir = 'SplitData'
class_names = ['BacterialSpot', 'EarlyBlight', 'Healthy', 'LateBlight', 'LeafMold', 'TargetSpot', 'BlackSpot']
split_data(input_dir, output_dir, class_names)
