import pandas as pd
import numpy as np
import sys
import json
import matplotlib.pyplot as plt

def get_accuracy(X, Y, thetas):
    probs = np.array([sigmoid(np.dot(X, t)) for t in thetas]).T
    predictions = np.argmax(probs, axis=1)
    return np.mean(predictions == Y)
    
    

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def logistic_regression(X, Y, learningRate, maxIterations, house_name, fig_loss_num):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    local_costs = []
    accuracies = []

    for i in range(maxIterations):        
        for j in range(n_samples):
            x_j = X[j]
            y_j = Y[j]

            z = np.dot(x_j, theta)
            h = sigmoid(z)
            h = np.clip(h, 1e-10, 1 - 1e-10)

            error = h - y_j
            gradient = x_j * error

            theta -= learningRate * gradient
        if i % 10 == 0:
            z_all = np.dot(X, theta)
            h_all = sigmoid(z_all)
            h_all = np.clip(h_all, 1e-10, 1 - 1e-10)
            cost = -np.mean(Y * np.log(h_all) + (1 - Y) * np.log(1 - h_all))
            local_costs.append(cost)
            preds = (h_all >= 0.5).astype(int)
            accuracies.append(np.mean(preds == Y))
    plt.figure(fig_loss_num)
    plt.plot(local_costs, label=house_name)
    return theta, accuracies

def train_one_vs_all(X, Y, classes, learningRate, maxIterations):
    thetas = []
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    fig_loss = plt.figure(figsize=(10, 5))
    fig_acc = plt.figure(figsize=(10, 5))
    for clasS in range(classes):
        y_binary = (Y == clasS).astype(int)
        theta_class, house_accuracies  = logistic_regression(X, y_binary, learningRate, maxIterations, houses[clasS], fig_loss.number)
        thetas.append(theta_class)
        plt.figure(fig_acc.number)
        plt.plot(house_accuracies, label=f"{houses[clasS]} Accuracy")

    plt.figure(fig_loss.number)
    plt.title("Training Loss per House")
    plt.xlabel("Iterations (x10)")
    plt.ylabel("Cost")
    plt.legend()

    plt.figure(fig_acc.number)
    plt.title("Binary Accuracy per House")
    plt.xlabel("Iterations (x10)")
    plt.ylabel("Accuracy %")
    plt.ylim(0, 1.1)
    plt.legend()

    plt.show()
    return thetas

def train_logistic_regression_model(dataset_path):
    X, Y = [], []
    dataFrame = pd.read_csv(dataset_path)
    with open("selected_features.json", "r") as f:
        selected_features = json.load(f)["selected_features"]
    X = dataFrame[selected_features]
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

    X = np.hstack((np.ones((X.shape[0], 1)), X))

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