import pandas as pd
import numpy as np
import sys
import json

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, Y, learningRate, maxIterations):
    n_samples, n_features = X.shape
    # print("n_samples :", n_samples, "n_features :", n_features, "\n")
    theta = np.zeros(n_features)
    for i in range(maxIterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        h = np.clip(h, 1e-10, 1-1e-10)
        error = h - Y
        gradient = np.dot(X.T, error) / n_samples
        print("gradient : ", gradient)
        theta -= learningRate * gradient
    return theta

def train_one_vs_all(X, Y, classes, learningRate, maxIterations):
    thetas = []
    for clasS in range(classes):
        y_binary = (Y == clasS).astype(int)
        print("y_binary : ", y_binary)
        theta_class = logistic_regression(X, y_binary, learningRate, maxIterations)
        thetas.append(theta_class)
    return thetas

def train_logistic_regression_model(dataset_path):
    # prepare dataset
    X, Y = [], []
    dataFrame = pd.read_csv(dataset_path)
    ignored_columns = ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    X = dataFrame.drop(ignored_columns, axis=1)
    X = X.select_dtypes(include='number')
    X = X.fillna(X.mean(numeric_only=True))
    X = X.values
    X_max, X_min = X.max(axis=0), X.min(axis=0)
    denom = (X_max - X_min)
    denom[denom == 0] = 1
    X = (X - X_min) / denom
    Y= dataFrame['Hogwarts House']
    Y = Y.map({'Gryffindor': 0, 'Slytherin': 1, 'Ravenclaw': 2, 'Hufflepuff': 3})
    Y = Y.values

    #training
    # print("before bias :", X)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # print("bias : ", X)
    thetaS = train_one_vs_all(X, Y, 4, 0.1, 10000)
    thetaS_fixed = {
        "Gryffindor": thetaS[0].tolist(),
        "Slytherin": thetaS[1].tolist(),
        "Ravenclaw": thetaS[2].tolist(),
        "Hufflepuff": thetaS[3].tolist()
    }
    with open('weights.json', 'w') as f:
        json.dump(thetaS_fixed, f)
    with open('minmax.json', 'w') as f:
        json.dump({"X_min": X_min.tolist(), "X_max": X_max.tolist()}, f)
    print("Training finished ✅")

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Usage: python logreg_train.py <dataset_train.csv>")
        sys.exit(1)
    else:
        train_logistic_regression_model(sys.argv[1])