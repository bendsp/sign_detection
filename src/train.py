"""
Training module for the ASL classifier.
Trains a Random Forest classifier on extracted hand landmarks.
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os


def load_features(features_path):
    """Load extracted features from pickle file."""
    with open(features_path, "rb") as f:
        data = pickle.load(f)
    return data["features"], data["labels"], data["classes"]


def train_classifier(features_path, model_path, test_size=0.2, random_state=42):
    """
    Train a Random Forest classifier on the extracted features.

    Args:
        features_path: Path to landmarks.pkl
        model_path: Path to save the trained model
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    print("Loading features...")
    features, labels, classes = load_features(features_path)

    print(f"Dataset size: {len(features)} samples")
    print(f"Number of classes: {len(classes)}")
    print(f"Feature dimension: {features.shape[1]}")

    # Filter out classes with too few samples for stratified split
    from collections import Counter

    label_counts = Counter(labels)
    min_samples = max(2, int(1 / test_size) + 1)  # Need at least 1 per split
    rare_classes = [c for c, n in label_counts.items() if n < min_samples]
    if rare_classes:
        print(f"\nRemoving {len(rare_classes)} class(es) with < {min_samples} samples: {rare_classes}")
        mask = np.array([l not in rare_classes for l in labels])
        features = features[mask]
        labels = [l for l in labels if l not in rare_classes]
        classes = [c for c in classes if c not in rare_classes]
        print(f"Filtered dataset: {len(features)} samples, {len(classes)} classes")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )

    clf.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_data = {"classifier": clf, "classes": classes}

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to: {model_path}")

    return clf, accuracy


if __name__ == "__main__":
    features_path = "data/landmarks.pkl"
    model_path = "models/asl_classifier.pkl"

    train_classifier(features_path, model_path)
