import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

# Load the model
def test_model(images,labels, model):
    # Evaluate the model on the test data


    test_loss, test_accuracy = model.evaluate(images, labels)
    
    # Print the test loss and accuracy
    print("Test Accuracy:", test_accuracy)
    print("Test Loss:", test_loss)

    # Make predictions on the test data
    predictions = model.predict(images)

    # Get the predicted classes for each image
    predicted_classes = np.argmax(predictions, axis=1)


    plot_confusion(predicted_classes, labels)

    #f1 test
    f1 = f1_score(labels, predicted_classes, average='weighted')


    print("F1 Score:", f1)
    


# Plot the confusion matrix
def plot_confusion(predicted_classes, test_labels):

    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_classes)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
