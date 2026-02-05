"""
Feature extraction module using MediaPipe Hands.
Extracts 21 hand landmarks + engineered features for better gesture discrimination.
Uses the new MediaPipe Tasks API (0.10.x+).

Enhanced features include:
- Normalized (x, y, z) for all 21 landmarks (63 features)
- Pairwise distances between key landmarks (thumb tip to each fingertip, etc.)
- Angles between finger joints
- Relative z-depth comparisons (thumb vs fingers)
"""

import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Model URL for hand landmarker
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "hand_landmarker.task"
)

# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def download_model():
    """Download the hand landmarker model if not present."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    print(f"Downloading hand landmarker model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    return MODEL_PATH


def compute_angle(p1, p2, p3):
    """Compute angle at p2 formed by p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def compute_engineered_features(landmarks):
    """
    Compute additional engineered features to better discriminate gestures.

    Args:
        landmarks: (21, 3) array of normalized landmark coordinates

    Returns:
        1D array of engineered features
    """
    features = []

    # 1. Distances from thumb tip to each other fingertip (4 features)
    thumb_tip = landmarks[THUMB_TIP]
    for tip_idx in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        dist = np.linalg.norm(thumb_tip - landmarks[tip_idx])
        features.append(dist)

    # 2. Distances between adjacent fingertips (4 features)
    for i in range(len(FINGERTIPS) - 1):
        dist = np.linalg.norm(landmarks[FINGERTIPS[i]] - landmarks[FINGERTIPS[i + 1]])
        features.append(dist)

    # 3. Z-depth comparisons: thumb tip vs each fingertip (4 features)
    # Positive = thumb is in front (closer to camera), Negative = thumb behind
    for tip_idx in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        z_diff = landmarks[tip_idx, 2] - thumb_tip[2]
        features.append(z_diff)

    # 4. Z-depth of thumb tip relative to palm center (1 feature)
    palm_center = np.mean(landmarks[FINGER_MCPS], axis=0)
    thumb_palm_z = thumb_tip[2] - palm_center[2]
    features.append(thumb_palm_z)

    # 5. Finger curl angles (flexion at PIP joints) for each finger (5 features)
    finger_joints = [
        (THUMB_MCP, THUMB_IP, THUMB_TIP),
        (INDEX_MCP, INDEX_PIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP),
    ]
    for mcp, pip, tip in finger_joints:
        angle = compute_angle(landmarks[mcp], landmarks[pip], landmarks[tip])
        features.append(angle)

    # 6. Finger spread angles (angles between adjacent fingers at MCP) (4 features)
    for i in range(1, len(FINGER_MCPS) - 1):  # Skip thumb for this
        angle = compute_angle(
            landmarks[FINGERTIPS[i - 1]], landmarks[WRIST], landmarks[FINGERTIPS[i + 1]]
        )
        features.append(angle)

    # 7. Thumb position relative to index finger (3 features: x, y, z difference)
    thumb_index_diff = landmarks[THUMB_TIP] - landmarks[INDEX_MCP]
    features.extend(thumb_index_diff.tolist())

    # 8. Distance from each fingertip to wrist (5 features) - indicates finger extension
    for tip_idx in FINGERTIPS:
        dist = np.linalg.norm(landmarks[tip_idx] - landmarks[WRIST])
        features.append(dist)

    # 9. Cross product z-component for thumb orientation (1 feature)
    # This helps detect if thumb is crossing over palm
    thumb_vec = landmarks[THUMB_TIP] - landmarks[THUMB_MCP]
    index_vec = landmarks[INDEX_TIP] - landmarks[INDEX_MCP]
    cross_z = thumb_vec[0] * index_vec[1] - thumb_vec[1] * index_vec[0]
    features.append(cross_z)

    return np.array(features)


class HandLandmarkExtractor:
    """Extracts and normalizes hand landmarks using MediaPipe Tasks API."""

    def __init__(self, min_detection_confidence=0.5, use_engineered_features=True):
        # Ensure model is downloaded
        model_path = download_model()
        self.use_engineered_features = use_engineered_features

        # Create hand landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def extract_landmarks(self, image):
        """
        Extract hand landmarks from an image.

        Args:
            image: BGR image (numpy array)

        Returns:
            Feature array or None if no hand detected.
            If use_engineered_features=True: 63 base + 35 engineered = 98 features
            If use_engineered_features=False: 63 features
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect hands
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return None

        # Get the first hand's landmarks
        hand_landmarks = result.hand_landmarks[0]

        # Extract coordinates
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # Normalize: center on wrist (landmark 0) and scale
        wrist = landmarks[0]
        landmarks = landmarks - wrist  # Center on wrist

        # Scale by max distance from wrist (using only x, y for scale)
        max_dist = np.max(np.linalg.norm(landmarks[:, :2], axis=1))
        if max_dist > 0:
            landmarks = landmarks / max_dist

        # Base features: flattened landmarks
        base_features = landmarks.flatten()

        if self.use_engineered_features:
            # Add engineered features
            eng_features = compute_engineered_features(landmarks)
            return np.concatenate([base_features, eng_features])
        else:
            return base_features

    def close(self):
        """Release MediaPipe resources."""
        self.detector.close()


def extract_features_from_dataset(data_dir, output_path, sample_per_class=None):
    """
    Extract landmarks from all images in the dataset.

    Args:
        data_dir: Path to asl_alphabet_train directory
        output_path: Path to save the extracted features
        sample_per_class: If set, limit samples per class (for faster testing)
    """
    extractor = HandLandmarkExtractor(use_engineered_features=True)

    features = []
    labels = []
    skipped = 0

    # Get all class directories
    classes = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    print(f"Found {len(classes)} classes: {classes}")

    for class_name in tqdm(classes, desc="Processing classes"):
        class_dir = os.path.join(data_dir, class_name)
        images = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if sample_per_class:
            images = images[:sample_per_class]

        for img_name in tqdm(images, desc=f"  {class_name}", leave=False):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)

            if image is None:
                skipped += 1
                continue

            landmarks = extractor.extract_landmarks(image)

            if landmarks is not None:
                features.append(landmarks)
                labels.append(class_name)
            else:
                skipped += 1

    extractor.close()

    features = np.array(features)
    labels = np.array(labels)

    print(f"\nExtraction complete:")
    print(f"  - Total samples: {len(features)}")
    print(f"  - Skipped (no hand detected): {skipped}")
    print(f"  - Feature shape: {features.shape}")
    print(f"  - Base landmarks: 63, Engineered features: {features.shape[1] - 63}")

    # Save to pickle
    data = {"features": features, "labels": labels, "classes": classes}

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"  - Saved to: {output_path}")

    return features, labels


if __name__ == "__main__":
    import sys

    # Default paths
    data_dir = "data/asl_alphabet_train/asl_alphabet_train"
    output_path = "data/landmarks.pkl"

    # Optional: limit samples for testing
    sample_per_class = None
    if len(sys.argv) > 1:
        sample_per_class = int(sys.argv[1])
        print(f"Limiting to {sample_per_class} samples per class")

    extract_features_from_dataset(data_dir, output_path, sample_per_class)
