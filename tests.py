import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the model
def test_model(images,labels, model):
    # Evaluate the model on the test data


    test_loss, test_accuracy = model.evaluate(images, labels)
    
    # Print the test loss and accuracy
    print("Test Accuracy:", test_accuracy)
    print("Test Loss:", test_loss)
    plot_predict(model, images, labels)

    


# Plot the accuracy and loss
def plot_predict(model, test_images, test_labels):

    # Make predictions on the test data
    predictions = model.predict(test_images)

    # Get the predicted classes for each image
    predicted_classes = np.argmax(predictions, axis=1)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_classes)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
