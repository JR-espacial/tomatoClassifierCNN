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

    # Make predictions on the test data
    predictions = model.predict(images)

    # Get the predicted classes for each image
    predicted_classes = np.argmax(predictions, axis=1)
    # Plot confidence distribution
    plot_confidence_distribution(predictions, labels)

    # Define class names (7)
    class_names = [ 'BacterialSpot', 'EarlyBlight', 
                'Healthy', 'LateBlight', 'LeafMold', 
                'TargetSpot', 'BlackSpot']
    
    plot_predictedImages(images,predictions, class_names)

    plot_confusion(predicted_classes, labels)

    #f1 test
    f1 = f1_score(labels, predicted_classes, average='weighted')


    print("F1 Score:", f1)
    
# Plot the distribution of confidence scores
def plot_confidence_distribution(predictions, test_labels):
    # Get the maximum confidence score for each prediction
    max_confidences = np.max(predictions, axis=1)

    # Plot histogram of confidence scores
    plt.figure(figsize=(8, 6))
    plt.hist(max_confidences, bins=30, alpha=0.75)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.show()

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


def get_prediction_label(probabilities, class_names):
	return class_names[np.argmax(probabilities)]

def plot_predictedImages(images,predictions, class_names):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(get_prediction_label(predictions[i], class_names))
