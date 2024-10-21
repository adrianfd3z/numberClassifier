import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
import time
import json
import sys
import os
import seaborn as sns

def load_config():
    global use, algorithms, train_size  # Aseguramos que 'algorithms' sea global
    print("Configuration loaded")
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python main.py [config_file.json]")
            exit()
        else:
            config_file = sys.argv[1]
            with open(config_file) as f:
                config = json.load(f)

        if 'USE' in config:                     # Can be ALL, ONLY or WITHOUT
            use = config['USE']
        else:
            use = "ALL"
        if 'ALGORITHMS' in config:              # Name of the algorithms to use
            algorithms = config['ALGORITHMS']   
        else:
            algorithms = ["LINEAR SVM"]
        if 'TRAIN_SIZE' in config:      
            train_size = config['TRAIN_SIZE']
        else:
            train_size = 5000


def linear_svm(x_train, y_train, x_test, y_test):
    # Initialize the SVM classifier (with a linear kernel for speed)
    clf = svm.SVC(kernel='linear')
    # Train the model with the reduced subset
    clf.fit(x_train, y_train)

    # Make predictions
    predicted = clf.predict(x_test)

    # Print the classification report and confusion matrix
    
    save_results("LINEAR SVM", y_test, predicted)




def save_results(algorithm, y_test, predicted):
    # Define the base output directory
    base_output_dir = 'output'
    
    # Create the base directory only if it doesn't exist
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Create the algorithm-specific directory only if it doesn't exist
    algorithm_dir = os.path.join(base_output_dir, algorithm)
    if not os.path.exists(algorithm_dir):
        os.makedirs(algorithm_dir)
    
    # Save the classification report as text
    classification_report = metrics.classification_report(y_test, predicted)
    report_path = os.path.join(algorithm_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Classification report for {algorithm}:\n")
        f.write(classification_report)
    
    # Create and save the confusion matrix as an image
    confusion_matrix = metrics.confusion_matrix(y_test, predicted)
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix for {algorithm}")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    confusion_matrix_img_path = os.path.join(algorithm_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_img_path)
    plt.close()

    print(f"Results saved for {algorithm} in {algorithm_dir}")

#######################################################################################################################
                                        #   M A I N     F U N C T I O N   #
#######################################################################################################################
def main():
    global algorithms, use, train_size
    load_config()
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape the data to be 2D instead of 3D (28x28)
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0

    # Reduce the size of the dataset
    global train_size
    train_size = 5000  # Number of samples you want to use
    x_train_small = x_train[:train_size]
    y_train_small = y_train[:train_size]
    global use, algorithms
    if use == "ALL":
        algorithms = ["LINEAR SVM"]
    elif use == "ONLY":
        if len(algorithms) == 0:
            print("No algorithms specified in config file")
            exit()
    elif use == "WITHOUT":
        all = ["LINEAR SVM"]
        algorithms = all - algorithms
        if len(algorithms) == 0:
            print("Please do not not remove all algorithms from the list")
            exit()
    else:
        print("Invalid value for USE in config file")
        exit()

    print("Using the following algorithms: ", algorithms)
    print("-----------------------------------------------")

    for algorithm in algorithms:
        if algorithm == "LINEAR SVM":
            print(f"Running {algorithm}...")
            start_time = time.time()
            linear_svm(x_train_small, y_train_small, x_test, y_test)
            print(f"Training completed for {algorithm} in {time.time() - start_time:.2f} seconds")

    

if __name__ == "__main__":
    main()