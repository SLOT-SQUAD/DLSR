import sys
import csv
import matplotlib.pyplot as plt
import json
import os

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

def calculate_mean(values):
    if not values:
        return 0

    total = 0
    for value in values:
        total += value

    return total / len(values)

def calculate_separation_score(grouped_scores):
    means = []

    for house, scores in grouped_scores.items():
        means.append(calculate_mean(scores))

    return max(means) - min(means)

def select_best_features(rows, numeric_features, top_n=4):
    scored_features = []

    for feature in numeric_features:
        grouped_scores = get_single_feature_by_house(rows, feature)
        score = calculate_separation_score(grouped_scores)

        scored_features.append((feature, score))

    scored_features.sort(key=lambda item: item[1], reverse=True)

    selected_features = []
    for feature, score in scored_features[:top_n]:
        selected_features.append(feature)

    return selected_features

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

def get_single_feature_by_house(rows, feature_name):
    grouped_scores = {
        "Gryffindor": [],
        "Hufflepuff": [],
        "Ravenclaw": [],
        "Slytherin": []
    }

    for row in rows:
        house = row["Hogwarts House"]
        value = row[feature_name]

        if house in grouped_scores and is_float(value):
            grouped_scores[house].append(float(value))

    return grouped_scores

def plot_pair_matrix(rows, selected_features):
    house_colors = {
        "Gryffindor": "red",
        "Hufflepuff": "gold",
        "Ravenclaw": "blue",
        "Slytherin": "green"
    }

    n = len(selected_features)
    fig, axes = plt.subplots(n, n, figsize=(14, 14))

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            feature_y = selected_features[i]
            feature_x = selected_features[j]

            if i == j:
                grouped_scores = get_single_feature_by_house(rows, feature_x)

                for house, scores in grouped_scores.items():
                    ax.hist(scores, bins=20, alpha=0.5, color=house_colors[house])

            else:
                grouped_points = get_feature_pairs_by_house(rows, feature_x, feature_y)

                for house, points in grouped_points.items():
                    ax.scatter(
                        points["x"],
                        points["y"],
                        s=8,
                        alpha=0.5,
                        color=house_colors[house]
                    )

            if i == n - 1:
                ax.set_xlabel(feature_x, fontsize=8)
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(feature_y, fontsize=8)
            else:
                ax.set_yticks([])

    plt.tight_layout()
    plt.show()

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

def select_features_for_logreg(rows, numeric_features, top_n=4, correlation_threshold=0.9):
    scored_features = []

    for feature in numeric_features:
        grouped_scores = get_single_feature_by_house(rows, feature)
        score = calculate_separation_score(grouped_scores)
        scored_features.append((feature, score))

    scored_features.sort(key=lambda item: item[1], reverse=True)

    selected_features = []

    for feature, score in scored_features:
        should_add = True

        for selected_feature in selected_features:
            x_values, y_values = get_feature_pairs(rows, feature, selected_feature)
            correlation = calculate_correlation(x_values, y_values)

            if abs(correlation) >= correlation_threshold:
                should_add = False
                break

        if should_add:
            selected_features.append(feature)

        if len(selected_features) == top_n:
            break

    return selected_features

def save_selected_features_to_json(selected_features, filename="selected_features.json"):
    if os.path.exists(filename):
        print(f"Le fichier {filename} existe déjà. Création ignorée.")
        return

    data = {
        "selected_features": selected_features
    }

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"Fichier JSON créé : {filename}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 pair_plot.py dataset_train.csv")
        sys.exit(1)

    path = sys.argv[1]
    rows = load_dataset(path)
    numeric_features = get_numeric_features(rows)

    selected_features = select_features_for_logreg(
        rows,
        numeric_features,
        top_n=4,
        correlation_threshold=0.9
    )

    print("Features recommandées pour la logistic regression :")
    for feature in selected_features:
        print("-", feature)

    save_selected_features_to_json(selected_features)

    plot_pair_matrix(rows, selected_features)


if __name__ == "__main__":
    main()