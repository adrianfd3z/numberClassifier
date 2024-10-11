import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
import time

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to be 2D instead of 3D (28x28)
x_train = x_train.reshape(len(x_train), -1) / 255.0
x_test = x_test.reshape(len(x_test), -1) / 255.0

# Reduce the size of the dataset
subset_size = 5000  # Number of samples you want to use
x_train_small = x_train[:subset_size]
y_train_small = y_train[:subset_size]

# Initialize the SVM classifier (with a linear kernel for speed)
classifier = svm.SVC(kernel='linear')

# Measure the training time
start_time = time.time()

# Train the model with the reduced subset
classifier.fit(x_train_small, y_train_small)

# Print the training time
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Make predictions
predicted = classifier.predict(x_test)

# Display the results
print("Linear SVM - Classification with reduced dataset")
print("Classification report:\n", metrics.classification_report(y_test, predicted))
print("Confusion matrix:\n", metrics.confusion_matrix(y_test, predicted))
