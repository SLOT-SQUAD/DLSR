import pandas as pd
import numpy as np
import sys
import json

from logreg_train import sigmoid

def test_logistic_regression_model(dataset_path):

    dataFrame = pd.read_csv(dataset_path)
    indices = dataFrame["Index"].values
    with open('weights.json', 'r') as f:
        models = json.load(f)

    with open('minmax.json', 'r') as f:
        minmax = json.load(f)
    X_min = np.array(minmax["X_min"])
    X_max = np.array(minmax["X_max"])
    with open("selected_features.json", "r") as f:
        selected_features = json.load(f)["selected_features"]
    numeric_columns = [col for col in dataFrame.columns if col in selected_features]

    X = dataFrame[numeric_columns].fillna(0).values.astype(float)

    denom = X_max - X_min
    denom[denom == 0] = 1
    X = (X - X_min) / denom

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    predictions = []

    # Predict
    for x in X:
        probs = {}
        for house, theta in models.items():
            theta = np.array(theta, dtype=float)
            x_trim = x[:len(theta)]
            prob = sigmoid(np.dot(x_trim, theta))
            probs[house] = prob
        best_house = max(probs, key=probs.get)
        predictions.append(best_house)

    with open("houses.csv", "w") as f:
        f.write("Index,Hogwarts House\n")
        for i, house in zip(indices, predictions):
            f.write(f"{i},{house}\n")
    print("Prediction complete ✅")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py <dataset_test.csv>")
        sys.exit(1)
    dataset_test = sys.argv[1]
    test_logistic_regression_model(dataset_test)
