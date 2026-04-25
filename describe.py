import sys
import os
import pandas as pd 
import csv

def x_max(dataFrame):
    max_value = dataFrame[0]
    for value in dataFrame:
        if value > max_value:
            max_value = value
    return max_value

def x_min(dataFrame):
    min_value = dataFrame[0]
    for value in dataFrame:
        if value < min_value:
            min_value = value
    return min_value

def x_mean(dataFrame):
    tot = 0
    for value in dataFrame:
        tot += value
    return tot / len(dataFrame)

def x_std(dataFrame):
    mean = x_mean(dataFrame)
    total = 0
    for value in dataFrame:
        total += (value - mean) ** 2
    variance = total / len(dataFrame)
    return variance ** 0.5

def x_percentile(dataFrame, p):
    sorted_values = sorted(dataFrame)
    n = len(sorted_values)
    index = int(p * (n - 1))
    return sorted_values[index]

def save_to_csv(results, columns, filename="describe_output.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([""] + list(columns))

        for stat, values in results.items():
            writer.writerow([stat] + values)

def describe_fun(data_path):
    dataFrame = pd.read_csv(data_path)
    ignored_columns = ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    X = dataFrame.drop(ignored_columns, axis=1)
    X = X.select_dtypes(include='number')
    X = X.fillna(X.mean(numeric_only=True))
    print(f"{'':<10}", end="")
    for col in X.columns:
        print(f"{col:<15}", end="")
    print()

    stats = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    results = {stat: [] for stat in stats}

    for col in X.columns:
        values = X[col].dropna().tolist()
        
        results["Count"].append(len(values))
        results["Mean"].append(x_mean(values))
        results["Std"].append(x_std(values))
        results["Min"].append(x_min(values))
        results["25%"].append(x_percentile(values, 0.25))
        results["50%"].append(x_percentile(values, 0.5))
        results["75%"].append(x_percentile(values, 0.75))
        results["Max"].append(x_max(values))
    for stat in stats:
        print(f"{stat:<10}", end="")
        for val in results[stat]:
            print(f"{val:<15.6f}", end="")
        print()
    save_to_csv(results, X.columns)

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Usage: python3 describe.py <dataset_test.csv>")
        sys.exit(1)
    describe_fun(sys.argv[1])