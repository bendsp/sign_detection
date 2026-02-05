"""
Real-time webcam ASL recognition module.
Displays hand landmarks and predicted letters in real-time.
Uses the new MediaPipe Tasks API (0.10.x+).
"""

import pickle
import cv2
import numpy as np
import time
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.extract_features import compute_engineered_features


# Model URL for hand landmarker
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "hand_landmarker.task"
)

# Hand connections for drawing (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (5, 9),
    (9, 13),
    (13, 17),  # Palm
]


def download_model():
    """Download the hand landmarker model if not present."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    print(f"Downloading hand landmarker model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    return MODEL_PATH


class RealtimeASLRecognizer:
    """Real-time ASL recognition using webcam."""

    def __init__(self, model_path="models/asl_classifier.pkl"):
        # Load classifier model
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.clf = data["classifier"]
        self.classes = data["classes"]

        # Download and setup MediaPipe hand landmarker
        landmarker_path = download_model()

        # For real-time, we use LIVE_STREAM mode with callback
        self.latest_result = None
        self.result_timestamp = 0

        base_options = python.BaseOptions(model_asset_path=landmarker_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5,
            result_callback=self._result_callback,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Smoothing for predictions
        self.prediction_history = []
        self.history_size = 5

    def _result_callback(self, result, output_image, timestamp_ms):
        """Callback for async hand detection results."""
        self.latest_result = result
        self.result_timestamp = timestamp_ms

    def normalize_landmarks(self, hand_landmarks):
        """Normalize landmarks and compute engineered features for prediction."""
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # Center on wrist
        wrist = landmarks[0]
        landmarks = landmarks - wrist

        # Scale by max distance (using x, y only for scale consistency)
        max_dist = np.max(np.linalg.norm(landmarks[:, :2], axis=1))
        if max_dist > 0:
            landmarks = landmarks / max_dist

        # Base features
        base_features = landmarks.flatten()

        # Add engineered features (same as training)
        eng_features = compute_engineered_features(landmarks)

        return np.concatenate([base_features, eng_features])

    def get_smoothed_prediction(self, prediction, confidence):
        """Smooth predictions using a sliding window."""
        self.prediction_history.append((prediction, confidence))
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)

        # Get most common prediction with confidence weighting
        pred_counts = {}
        for pred, conf in self.prediction_history:
            if pred not in pred_counts:
                pred_counts[pred] = 0
            pred_counts[pred] += conf

        best_pred = max(pred_counts.keys(), key=lambda k: pred_counts[k])
        avg_conf = pred_counts[best_pred] / len(self.prediction_history)

        return best_pred, avg_conf

    def draw_landmarks_on_image(self, frame, hand_landmarks):
        """Draw hand landmarks on the frame manually."""
        h, w = frame.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_point = points[start_idx]
            end_point = points[end_idx]
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Draw landmark points
        for i, point in enumerate(points):
            # Fingertips in different color
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(frame, point, 6, (255, 0, 255), -1)  # Purple for fingertips
            elif i == 0:
                cv2.circle(frame, point, 6, (0, 255, 255), -1)  # Yellow for wrist
            else:
                cv2.circle(frame, point, 4, (0, 0, 255), -1)  # Red for joints

    def run(self, camera_index=0):
        """Run the real-time recognition loop."""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n" + "=" * 50)
        print("ASL Real-Time Recognition")
        print("=" * 50)
        print("Show your hand to the camera to detect ASL letters")
        print("Press 'Q' to quit")
        print("=" * 50 + "\n")

        prev_time = time.time()
        fps = 0
        frame_timestamp = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB and create MediaPipe Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Process frame asynchronously
            frame_timestamp += 33  # ~30fps
            self.detector.detect_async(mp_image, frame_timestamp)

            prediction = None
            confidence = 0.0

            # Use latest available result
            if self.latest_result and self.latest_result.hand_landmarks:
                for hand_landmarks in self.latest_result.hand_landmarks:
                    # Draw landmarks
                    self.draw_landmarks_on_image(frame, hand_landmarks)

                    # Get prediction
                    landmarks = self.normalize_landmarks(hand_landmarks)
                    landmarks = landmarks.reshape(1, -1)

                    pred = self.clf.predict(landmarks)[0]
                    probs = self.clf.predict_proba(landmarks)[0]
                    conf = np.max(probs)

                    prediction, confidence = self.get_smoothed_prediction(pred, conf)
            else:
                # Clear history when no hand detected
                self.prediction_history = []

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time + 0.0001)
            prev_time = current_time

            # Draw UI
            self._draw_ui(frame, prediction, confidence, fps)

            # Show frame
            cv2.imshow("ASL Recognition", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()

    def _draw_ui(self, frame, prediction, confidence, fps):
        """Draw the UI overlay on the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Prediction text
        if prediction:
            # Large letter display
            letter_size = 2.5
            letter_thickness = 4
            text = prediction
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, letter_size, letter_thickness
            )[0]
            text_x = (w - text_size[0]) // 2

            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange

            cv2.putText(
                frame,
                text,
                (text_x, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                letter_size,
                color,
                letter_thickness,
                cv2.LINE_AA,
            )

            # Confidence bar
            bar_width = int(w * 0.6)
            bar_height = 15
            bar_x = (w - bar_width) // 2
            bar_y = h - 25

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (100, 100, 100),
                -1,
            )
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + int(bar_width * confidence), bar_y + bar_height),
                color,
                -1,
            )

            # Confidence percentage
            conf_text = f"{confidence * 100:.1f}%"
            cv2.putText(
                frame,
                conf_text,
                (bar_x + bar_width + 10, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Show your hand",
                ((w - 200) // 2, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (150, 150, 150),
                2,
                cv2.LINE_AA,
            )

        # FPS counter
        cv2.putText(
            frame,
            f"FPS: {fps:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Quit instruction
        cv2.putText(
            frame,
            "Press 'Q' to quit",
            (w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


def main(model_path="models/asl_classifier.pkl", camera_index=0):
    """Main entry point for real-time recognition."""
    recognizer = RealtimeASLRecognizer(model_path)
    recognizer.run(camera_index)


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/asl_classifier.pkl"
    camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    main(model_path, camera_index)
