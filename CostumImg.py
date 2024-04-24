from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TARGET_SIZE = (128, 128)
labels = ['BacterialSpot', 'EarlyBlight', 'Healthy', 'LateBlight', 'LeafMold', 'TargetSpot', 'BlackSpot']

# Load the model
model = tf.keras.models.load_model('jorge.h5')

# Load and preprocess the image
img_path = 'queries/test.jpg'
img = image.load_img(img_path, target_size=TARGET_SIZE)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# Make predictions using the model
confidence = model.predict(img_tensor)

print(confidence)
predicted_class_index = np.argmax(confidence)
predicted_class = labels[predicted_class_index]
highest_confidence = np.max(confidence)

# Display the image along with the predicted class and confidence score
plt.figure()
plt.imshow(img)
plt.title(f'{predicted_class} - Confidence: {highest_confidence*100:.2f}%')
plt.show()
