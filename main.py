import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from tensorflow.keras.datasets import mnist
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
import time
import json
import sys
import os
import seaborn as sns

def load_config():
    global use, algorithms, train_size, hyperparameters, use_full_train  # Aseguramos que 'algorithms' sea global
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
        if 'USE_FULL_TRAIN' in config:
            use_full_train = config['USE_FULL_TRAIN']
        else:
            use_full_train = False


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

def RBF_SVM(x_train, y_train, x_test, y_test):
    global hyperparameters
    #Default hyperparameters
    C = 1.0
    kernel = 'rbf'
    gamma = 'scale'
    tol = 1e-3
    max_iter = -1

    # Check if hyperparameters are provided in the config file
    if 'RBF SVM' in hyperparameters:
        if 'C' in hyperparameters['RBF SVM']:
            C = hyperparameters['RBF SVM']['C']
        if 'KERNEL' in hyperparameters['RBF SVM']:
            kernel = hyperparameters['RBF SVM']['KERNEL']
        if 'GAMMA' in hyperparameters['RBF SVM']:
            gamma = hyperparameters['RBF SVM']['GAMMA']
        if 'TOL' in hyperparameters['RBF SVM']:
            tol = hyperparameters['RBF SVM']['TOL']
        if 'MAX_ITER' in hyperparameters['RBF SVM']:
            max_iter = hyperparameters['RBF SVM']['MAX_ITER']
    
    # Initialize the SVM classifier
    clf = svm.SVC(C=C, kernel=kernel, gamma=gamma, tol=tol, max_iter=max_iter)

    # Train the model with the reduced subset
    clf.fit(x_train, y_train)

    # Make predictions
    predicted = clf.predict(x_test)

    hp = {"C": C, "KERNEL": kernel, "GAMMA": gamma, "TOL": tol, "MAX_ITER": max_iter}

    save_results("RBF SVM", y_test, predicted, hp)

def random_forest(x_train, y_train, x_test, y_test):
    global hyperparameters
    # Default hyperparameters for Random Forest
    n_estimators = 100
    criterion = 'gini'
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 1
    max_features = 'auto'
    bootstrap = True

    # Check if hyperparameters are provided in the config file
    if 'RANDOM FOREST' in hyperparameters:
        if 'N_ESTIMATORS' in hyperparameters['RANDOM FOREST']:
            n_estimators = hyperparameters['RANDOM FOREST']['N_ESTIMATORS']
        if 'CRITERION' in hyperparameters['RANDOM FOREST']:
            criterion = hyperparameters['RANDOM FOREST']['CRITERION']
        if 'MAX_DEPTH' in hyperparameters['RANDOM FOREST']:
            max_depth = hyperparameters['RANDOM FOREST']['MAX_DEPTH']
        if 'MIN_SAMPLES_SPLIT' in hyperparameters['RANDOM FOREST']:
            min_samples_split = hyperparameters['RANDOM FOREST']['MIN_SAMPLES_SPLIT']
        if 'MIN_SAMPLES_LEAF' in hyperparameters['RANDOM FOREST']:
            min_samples_leaf = hyperparameters['RANDOM FOREST']['MIN_SAMPLES_LEAF']
        if 'MAX_FEATURES' in hyperparameters['RANDOM FOREST']:
            max_features = hyperparameters['RANDOM FOREST']['MAX_FEATURES']
        if 'BOOTSTRAP' in hyperparameters['RANDOM FOREST']:
            bootstrap = hyperparameters['RANDOM FOREST']['BOOTSTRAP']
    
    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                max_features=max_features, bootstrap=bootstrap)

    # Train the model with the reduced subset
    clf.fit(x_train, y_train)

    # Make predictions
    predicted = clf.predict(x_test)

    hp = {
        "N_ESTIMATORS": n_estimators,
        "CRITERION": criterion,
        "MAX_DEPTH": max_depth,
        "MIN_SAMPLES_SPLIT": min_samples_split,
        "MIN_SAMPLES_LEAF": min_samples_leaf,
        "MAX_FEATURES": max_features,
        "BOOTSTRAP": bootstrap
    }

    save_results("RANDOM FOREST", y_test, predicted, hp)
def AdaBoost(x_train, y_train, x_test, y_test):
    global hyperparameters
    # Default hyperparameters for AdaBoost
    n_estimators = 50
    learning_rate = 1.0
    algorithm = 'SAMME'
    random_state = 42

    # Check if hyperparameters are provided in the config file
    if 'ADABOOST' in hyperparameters:
        if 'N_ESTIMATORS' in hyperparameters['ADABOOST']:
            n_estimators = hyperparameters['ADABOOST']['N_ESTIMATORS']
        if 'LEARNING_RATE' in hyperparameters['ADABOOST']:
            learning_rate = hyperparameters['ADABOOST']['LEARNING_RATE']
        if 'ALGORITHM' in hyperparameters['ADABOOST']:
            algorithm = hyperparameters['ADABOOST']['ALGORITHM']
        if 'random_state' in hyperparameters['ADABOOST']:
            random_state = hyperparameters['ADABOOST']['random_state']
    
    # Initialize the AdaBoost classifier
    clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm, random_state=random_state)

    # Train the model with the reduced subset
    clf.fit(x_train, y_train)

    # Make predictions
    predicted = clf.predict(x_test)

    hp = {
        "N_ESTIMATORS": n_estimators,
        "LEARNING_RATE": learning_rate,
        "ALGORITHM": algorithm
    }

    save_results("ADABOOST", y_test, predicted, hp)


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
    global algorithms, use, train_size, hyperparameters, use_full_train
    load_config()
    
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape the data to be 2D instead of 3D (28x28)
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0

    # Configuration to use the full training set or a subset
    
    
    if use_full_train:
        x_train_selected = x_train
        y_train_selected = y_train
        print("Using the full training dataset.")
    else:
        x_train_selected = x_train[:train_size]
        y_train_selected = y_train[:train_size]
        print(f"Using a subset of the training dataset with size {train_size}.")

    # Select algorithms to use
    if use == "ALL":
        algorithms = ["LINEAR SVM", "KNN", "RBF SVM", "RANDOM FOREST", "AdaBoost"]
    elif use == "ONLY":
        if len(algorithms) == 0:
            print("No algorithms specified in config file")
            exit()
    elif use == "WITHOUT":
        all_algorithms = ["LINEAR SVM", "KNN", "RBF SVM", "RANDOM FOREST", "AdaBoost"]
        algorithms = list(set(all_algorithms) - set(algorithms))
        if len(algorithms) == 0:
            print("Please do not remove all algorithms from the list")
            exit()
    else:
        print("Invalid value for USE in config file")
        exit()

    print("Using the following algorithms:", algorithms)
    print("-----------------------------------------------")

    # Run selected algorithms
    for algorithm in algorithms:
        print(f"Running {algorithm}...")
        start_time = time.time()
        if algorithm == "LINEAR SVM":
            linear_svm(x_train_selected, y_train_selected, x_test, y_test)
        elif algorithm == "KNN":
            knn(x_train_selected, y_train_selected, x_test, y_test)
        elif algorithm == "RBF SVM":
            RBF_SVM(x_train_selected, y_train_selected, x_test, y_test)
        elif algorithm == "RANDOM FOREST":
            random_forest(x_train_selected, y_train_selected, x_test, y_test)
        elif algorithm == "AdaBoost":
            AdaBoost(x_train_selected, y_train_selected, x_test, y_test)
        
        print(f"Training completed for {algorithm} in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
