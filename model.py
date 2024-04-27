import tensorflow as tf
import matplotlib.pyplot as plt

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
        Conv2D(100, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.22),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.22),
        Flatten(input_shape=(128,128,3)),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')
    ])

    return model

def train_model(images_tensor, train_labels, val_images, val_labels):
    model = create_model()
    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001) ,loss= 'sparse_categorical_crossentropy',metrics = ['accuracy'] )
    model.fit(images_tensor, train_labels, batch_size=32, epochs=20, validation_data=(val_images, val_labels))
    model.summary()
    
    # #graph the accuracy and loss
    history = model.history.history
    # Graph the accuracy and loss
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



    model.save('jorge.h5')

    