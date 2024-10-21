import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import mnist
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
import time
import json
import sys
import os
import seaborn as sns

def load_config():
    global use, algorithms, train_size, hyperparameters  # Aseguramos que 'algorithms' sea global
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
        if 'HYPERPARAMETERS' in config:
            hyperparameters = config['HYPERPARAMETERS']


def linear_svm(x_train, y_train, x_test, y_test):
    global hyperparameters
    
    # Default hyperparameters for Linear SVM
    C = 1.0
    loss = 'hinge'
    penalty = 'l2'
    dual = True
    tol = 1e-3
    max_iter = 1000

    # Check if hyperparameters are provided in the config file
    if 'LINEAR SVM' in hyperparameters:
        print("Using custom hyperparameters for LINEAR SVM.")
        if 'C' in hyperparameters['LINEAR SVM']:
            C = hyperparameters['LINEAR SVM']['C']
        if 'LOSS' in hyperparameters['LINEAR SVM']:
            loss = hyperparameters['LINEAR SVM']['LOSS'].lower()  # Lowercase to match valid input
        if 'PENALTY' in hyperparameters['LINEAR SVM']:
            penalty = hyperparameters['LINEAR SVM']['PENALTY'].lower()  # Lowercase to match valid input
        if 'DUAL' in hyperparameters['LINEAR SVM']:
            dual = hyperparameters['LINEAR SVM']['DUAL'].lower() == "true"  # Convert to boolean
        if 'TOL' in hyperparameters['LINEAR SVM']:
            tol = hyperparameters['LINEAR SVM']['TOL']
        if 'MAX_ITER' in hyperparameters['LINEAR SVM']:
            max_iter = hyperparameters['LINEAR SVM']['MAX_ITER']
    else:
        print("Using default hyperparameters for LINEAR SVM.")
    
    # Initialize the SVM classifier with the specified hyperparameters
    clf = svm.LinearSVC(C=C, loss=loss, penalty=penalty, dual=dual, tol=tol, max_iter=max_iter)

    # Train the model with the reduced subset
    clf.fit(x_train, y_train)

    # Make predictions
    predicted = clf.predict(x_test)

    # Collect the hyperparameters used
    hp = {
        "C": C,
        "LOSS": loss,
        "PENALTY": penalty,
        "DUAL": dual,
        "TOL": tol,
        "MAX_ITER": max_iter
    }

    # Save results and hyperparameters
    save_results("LINEAR SVM", y_test, predicted, hp)

def knn(x_train, y_train, x_test, y_test):
    global hyperparameters
    #Default hyperparameters
    n_neighbors = 5
    p = 2
    algorithm = 'auto'
    weights = 'uniform'
    metric = 'minkowski'

    # Check if hyperparameters are provided in the config file
    if 'KNN' in hyperparameters:
        if 'N_NEIGHBORS' in hyperparameters['KNN']:
            n_neighbors = hyperparameters['KNN']['N_NEIGHBORS']
        if 'P' in hyperparameters['KNN']:
            p = hyperparameters['KNN']['P']
        if 'ALGORITHM' in hyperparameters['KNN']:
            algorithm = hyperparameters['KNN']['ALGORITHM']
        if 'WEIGHTS' in hyperparameters['KNN']:
            weights = hyperparameters['KNN']['WEIGHTS']
        if 'METRIC' in hyperparameters['KNN']:
            metric = hyperparameters['KNN']['METRIC']
    
    # Initialize the KNN classifier
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, algorithm=algorithm, weights=weights, metric=metric)
    # Train the model with the reduced subset
    clf.fit(x_train, y_train)

    # Make predictions
    predicted = clf.predict(x_test)

    hp = {"N_NEIGHBORS": n_neighbors, "P": p, "ALGORITHM": algorithm, "WEIGHTS": weights, "METRIC": metric}

    save_results("KNN", y_test, predicted, hp)


def save_results(algorithm, y_test, predicted, hyperparameters):
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

    # Save the hyperparameters as txt
    hyperparameters_path = os.path.join(algorithm_dir, 'hyperparameters.txt')
    with open(hyperparameters_path, 'w') as f:
        f.write(f"Hyperparameters for {algorithm}:\n")
        f.write(json.dumps(hyperparameters, indent=4))

    print(f"Results saved for {algorithm} in {algorithm_dir}")

#######################################################################################################################
                                        #   M A I N     F U N C T I O N   #
#######################################################################################################################
def main():
    global algorithms, use, train_size, hyperparameters
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
        if algorithm == "KNN":
            print(f"Running {algorithm}...")
            start_time = time.time()
            knn(x_train_small, y_train_small, x_test, y_test)
            print(f"Training completed for {algorithm} in {time.time() - start_time:.2f} seconds")
    


    

if __name__ == "__main__":
    main()