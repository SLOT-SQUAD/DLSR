import pandas as pd
import numpy as np
import sys

from logreg_train import sigmoid, predict_logistic, predict_one_vs_all, X_max, X_min

def preprocess_test_data(df, X_min, X_max):
    X = df.select_dtypes(include='number')
    X = X.fillna(X.mean(numeric_only=True))  # fill missing values

    # normalize using training min/max
    denom = X_max - X_min
    denom[denom == 0] = 1
    X = (X - X_min) / denom

    # add bias column
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X

def test_logistic_regression_model(dataset_path, model_path, X_min, X_max):
    df_test = pd.read_csv(dataset_path)

    # preprocess features
    X_test = preprocess_test_data(df_test, X_min, X_max)

    # load trained weights
    thetas = np.load(model_path)

    # predict
    predictions = predict_one_vs_all(X_test, thetas)

    # map class numbers to Hogwarts houses
    class_map = {0: 'Gryffindor', 1: 'Slytherin', 2: 'Ravenclaw', 3: 'Hufflepuff'}
    houses = [class_map[p] for p in predictions]

    # save predictions
    df_output = pd.DataFrame({'Index': df_test['Index'], 'Hogwarts House': houses})
    df_output.to_csv('houses.csv', index=False)
    print("Predictions saved to houses.csv")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python logreg_predict.py <dataset_test.csv> <model.npy> <X_minmax.npz>")
        sys.exit(1)

    dataset_test = sys.argv[1]
    model_file = sys.argv[2]
    minmax_file = sys.argv[3]

    # load saved min/max from training
    minmax = np.load(minmax_file)
    X_min, X_max = minmax['X_min'], minmax['X_max']

    test_logistic_regression_model(dataset_test, model_file, X_min, X_max)
