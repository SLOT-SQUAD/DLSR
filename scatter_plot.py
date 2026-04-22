import sys
import csv
import matplotlib.pyplot as plt


def load_dataset(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    return rows


def is_float(value):
    if value is None or value == "":
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_numeric_features(rows):
    if not rows:
        return []

    headers = rows[0].keys()
    excluded_features = {"Index"}
    numeric_features = []

    for header in headers:
        if header in excluded_features:
            continue

        numeric_count = 0
        non_empty_count = 0

        for row in rows:
            value = row[header]
            if value != "":
                non_empty_count += 1
                if is_float(value):
                    numeric_count += 1

        if non_empty_count > 0 and numeric_count == non_empty_count:
            numeric_features.append(header)

    return numeric_features


def get_feature_pairs(rows, feature_x, feature_y):
    x_values = []
    y_values = []

    for row in rows:
        x = row[feature_x]
        y = row[feature_y]

        if is_float(x) and is_float(y):
            x_values.append(float(x))
            y_values.append(float(y))

    return x_values, y_values


def plot_scatter(x_values, y_values, feature_x, feature_y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.6)
    plt.title(f"{feature_x} vs {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def calculate_mean(values):
    if not values:
        return 0

    total = 0
    for value in values:
        total += value

    return total / len(values)


def calculate_correlation(x_values, y_values):
    if len(x_values) != len(y_values) or len(x_values) == 0:
        return 0

    mean_x = calculate_mean(x_values)
    mean_y = calculate_mean(y_values)

    numerator = 0
    sum_x_squared = 0
    sum_y_squared = 0

    for i in range(len(x_values)):
        x_diff = x_values[i] - mean_x
        y_diff = y_values[i] - mean_y

        numerator += x_diff * y_diff
        sum_x_squared += x_diff ** 2
        sum_y_squared += y_diff ** 2

    denominator = (sum_x_squared ** 0.5) * (sum_y_squared ** 0.5)

    if denominator == 0:
        return 0

    return numerator / denominator

def get_feature_pairs_by_house(rows, feature_x, feature_y):
    grouped_points = {
        "Gryffindor": {"x": [], "y": []},
        "Hufflepuff": {"x": [], "y": []},
        "Ravenclaw": {"x": [], "y": []},
        "Slytherin": {"x": [], "y": []}
    }

    for row in rows:
        house = row["Hogwarts House"]
        x = row[feature_x]
        y = row[feature_y]

        if house in grouped_points and is_float(x) and is_float(y):
            grouped_points[house]["x"].append(float(x))
            grouped_points[house]["y"].append(float(y))

    return grouped_points

def plot_scatter_by_house(grouped_points, feature_x, feature_y):
    house_colors = {
        "Gryffindor": "red",
        "Hufflepuff": "gold",
        "Ravenclaw": "blue",
        "Slytherin": "green"
    }

    plt.figure(figsize=(10, 6))

    for house, points in grouped_points.items():
        plt.scatter(
            points["x"],
            points["y"],
            alpha=0.6,
            label=house,
            color=house_colors[house]
        )

    plt.title(f"{feature_x} vs {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def find_most_similar_features(rows, numeric_features):
    best_feature_x = None
    best_feature_y = None
    best_correlation = 0

    for i in range(len(numeric_features)):
        for j in range(i + 1, len(numeric_features)):
            feature_x = numeric_features[i]
            feature_y = numeric_features[j]

            x_values, y_values = get_feature_pairs(rows, feature_x, feature_y)
            correlation = calculate_correlation(x_values, y_values)

            if abs(correlation) > abs(best_correlation):
                best_feature_x = feature_x
                best_feature_y = feature_y
                best_correlation = correlation

    return best_feature_x, best_feature_y, best_correlation

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scatter_plot.py dataset_train.csv")
        sys.exit(1)

    path = sys.argv[1]
    rows = load_dataset(path)
    numeric_features = get_numeric_features(rows)

    feature_x, feature_y, correlation = find_most_similar_features(rows, numeric_features)

    print("Deux features les plus similaires :")
    print(f"- {feature_x}")
    print(f"- {feature_y}")
    print(f"Corrélation : {correlation:.4f}")

    grouped_points = get_feature_pairs_by_house(rows, feature_x, feature_y)
    plot_scatter_by_house(grouped_points, feature_x, feature_y)


if __name__ == "__main__":
    main()