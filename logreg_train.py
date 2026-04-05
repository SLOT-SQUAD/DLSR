import pandas as pd
import numpy as np
import sys
X_max, X_min = None, None

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_logistic(X, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    return (h >= 0.5).astype(int)

def predict_one_vs_all(X, thetas):
    z = np.dot(X, thetas.T)
    h = sigmoid(z)
    return np.argmax(h, axis=1)

def logistic_regression(X, Y, learningRate, maxIterations):
    n_samples, n_features = X.shape
    print("n_samples :", n_samples, "n_features :", n_features, "\n")
    theta = np.zeros(n_features)
    for i in range(maxIterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        h = np.clip(h, 1e-10, 1-1e-10)
        error = h - Y
        gradient = np.dot(X.T, error) / n_samples
        theta -= learningRate * gradient
        # if i % 100 == 0:
        #     loss = -np.mean(Y*np.log(h) + (1-Y)*np.log(1-h))
        #     print(f"Iteration {i}, Loss: {loss:.4f}")
    return theta

def train_one_vs_all(X, Y, classes, learningRate, maxIterations):
    thetas = []
    for clasS in range(classes):
        y_binary = (Y == clasS).astype(int)
        theta_class = logistic_regression(X, y_binary, learningRate, maxIterations)
        thetas.append(theta_class)
    return thetas

def train_logistic_regression_model(dataset_path):
    # prepare dataset
    X, Y = [], []
    dataFrame = pd.read_csv(dataset_path)
    ignored_columns = ['Index', 'Hogwarts House', 'First Name', 'Last Name']
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

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # print("X :", X, "\n")
    #training
    thetaS = train_one_vs_all(X, Y, 4, 0.1, 10000)
    prediction = predict_one_vs_all(X, np.array(thetaS))
    np.save("weights.npy", np.array(thetaS))
    accuracy = np.mean(prediction == Y)
    print(f"Training accuracy: {accuracy*100:.2f}%")






    



if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Usage: python logreg_predict.py <dataset_test.csv>")
        sys.exit(1)
    else:
        train_logistic_regression_model(sys.argv[1])