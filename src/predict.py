"""
Single image prediction module.
Predicts the ASL letter from a single image.
Uses the new MediaPipe Tasks API.
"""

import pickle
import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extract_features import HandLandmarkExtractor

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def load_model(model_path):
    """Load the trained classifier."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["classifier"], data["classes"]


def predict_image(image_path, model_path="models/asl_classifier.pkl"):
    """
    Predict the ASL letter in an image.

    Args:
        image_path: Path to the image file
        model_path: Path to the trained model

    Returns:
        Tuple of (predicted_letter, confidence, all_probabilities)
    """
    # Load model
    clf, classes = load_model(model_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Extract landmarks
    extractor = HandLandmarkExtractor()
    landmarks = extractor.extract_landmarks(image)
    extractor.close()

    if landmarks is None:
        return None, 0.0, None

    # Predict
    landmarks = landmarks.reshape(1, -1)
    prediction = clf.predict(landmarks)[0]
    probabilities = clf.predict_proba(landmarks)[0]
    confidence = np.max(probabilities)

    return prediction, confidence, dict(zip(clf.classes_, probabilities))


def display_prediction(image_path, model_path="models/asl_classifier.pkl"):
    """Display the image with prediction overlay."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return

    prediction, confidence, probs = predict_image(image_path, model_path)

    if prediction is None:
        text = "No hand detected"
        color = (0, 0, 255)  # Red
    else:
        text = f"Prediction: {prediction} ({confidence * 100:.1f}%)"
        color = (0, 255, 0)  # Green

    # Add text to image
    cv2.putText(
        image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
    )

    # Resize for display if too large
    max_dim = 600
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    cv2.imshow("ASL Prediction", image)
    print(f"\n{text}")
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/asl_classifier.pkl"

    display_prediction(image_path, model_path)
