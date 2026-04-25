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

def group_scores_by_house(rows, feature_name):
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

def plot_histogram(grouped_scores, feature_name):
    plt.figure(figsize=(10, 6))

    for house, scores in grouped_scores.items():
        plt.hist(scores, bins=30, alpha=0.5, label=house)

    plt.title(f"Histogram of {feature_name} by Hogwarts House")
    plt.xlabel(feature_name)
    plt.ylabel("Number of students")
    plt.legend()
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

def calculate_homogeneity_score(grouped_scores):
    means = {}

    for house, scores in grouped_scores.items():
        means[house] = calculate_mean(scores)

    min_mean = min(means.values())
    max_mean = max(means.values())

    score = max_mean - min_mean
    return score, means

def analyze_homogeneity(rows, numeric_features):
    results = []

    for feature in numeric_features:
        grouped_scores = group_scores_by_house(rows, feature)
        score, means = calculate_homogeneity_score(grouped_scores)

        results.append({
            "feature": feature,
            "score": score,
            "means": means
        })

    results.sort(key=lambda item: item["score"])
    return results

def plot_homogeneity_bar_chart(results, title, selected_results):
    feature_names = []
    scores = []

    for item in selected_results:
        feature_names.append(item["feature"])
        scores.append(item["score"])

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, scores)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Homogeneity score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 histogram.py dataset_train.csv summary")
        print("  python3 histogram.py dataset_train.csv hist")
        print("  python3 histogram.py dataset_train.csv feature <feature_name>")
        sys.exit(1)

    path = sys.argv[1]
    mode = sys.argv[2]

    rows = load_dataset(path)
    numeric_features = get_numeric_features(rows)
    results = analyze_homogeneity(rows, numeric_features)

    if mode == "summary":
        print("\nMatières les plus homogènes :")
        for item in results[:5]:
            print(f"- {item['feature']} -> score = {item['score']:.2f}")

        print("\nMatières les moins homogènes :")
        for item in results[-5:]:
            print(f"- {item['feature']} -> score = {item['score']:.2f}")

        plot_homogeneity_bar_chart(
            results,
            "Top 5 des matières les plus homogènes",
            results[:5]
        )

        plot_homogeneity_bar_chart(
            results,
            "Top 5 des matières les moins homogènes",
            results[-5:]
        )

    elif mode == "hist":
        for feature in numeric_features:
            grouped_scores = group_scores_by_house(rows, feature)
            plot_histogram(grouped_scores, feature)

    elif mode == "feature":
        if len(sys.argv) < 4:
            print("Usage: python3 histogram.py dataset_train.csv feature <feature_name>")
            sys.exit(1)

        feature_name = sys.argv[3]

        if feature_name not in numeric_features:
            print(f"Feature invalide : {feature_name}")
            sys.exit(1)

        grouped_scores = group_scores_by_house(rows, feature_name)
        plot_histogram(grouped_scores, feature_name)

    else:
        print("Mode invalide. Utilise 'summary', 'hist' ou 'feature'.")
        sys.exit(1)


if __name__ == "__main__":
    main()