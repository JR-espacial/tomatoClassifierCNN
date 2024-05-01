import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
# TensorFlow and tf.keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout # type: ignore

# model definition
def create_model():
    
    #model v3
    # model = Sequential([
    #   Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    #   MaxPooling2D((2, 2)),
    #   Flatten(input_shape=(128,128,3)),
    #   Dense(128, activation='relu'),
    #   Dense(7, activation='softmax')
    # ])
    
    #Model 4

    model = Sequential([
        Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.10),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(input_shape=(128,128,3)),
        Dense(300, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    return model

# Train the model
def train_model(images_tensor, train_labels, val_images, val_labels):
    model = create_model()
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(images_tensor, train_labels, batch_size=32, epochs=22, validation_data=(val_images, val_labels))
    model.summary()
    
    history = model.history.history

    # Plot accuracy and loss
    plot_metrics(history)

    # Save the model
    model.save('jorge.h5')

    # Define class names (7)
    class_names = [ 'BacterialSpot', 'EarlyBlight', 
                'Healthy', 'LateBlight', 'LeafMold', 
                'TargetSpot', 'BlackSpot']

    # Plot confusion matrix
    predictions = np.argmax(model.predict(val_images), axis=-1)
    plot_confusion_matrix(val_labels, predictions, class_names)

# Plot training metrics
def plot_metrics(history):
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


    