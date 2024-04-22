import tensorflow as tf

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
        Dropout(0.20),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.20),
        Flatten(input_shape=(128,128,3)),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')
    ])

    return model

def train_model(images_tensor, train_labels, val_images, val_labels):
    model = create_model()
    model.compile(optimizer= 'adam',loss= 'sparse_categorical_crossentropy',metrics = ['accuracy'] )
    model.fit(images_tensor, train_labels, batch_size=32, epochs=10, validation_data=(val_images, val_labels))
    model.summary()
    model.save('jorge.h5')
    