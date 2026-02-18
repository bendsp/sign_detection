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
from src.motion import MotionDetector


# Model URL for hand landmarker
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "hand_landmarker.task"
)
WORDBANK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "wordbank.txt"
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

    def __init__(self, model_path="models/asl_classifier.pkl", spell_mode=False):
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

        # Prediction state
        self.prediction_history = []
        self.history_size = 5
        self.current_prediction = (None, 0.0)
        self.last_processed_timestamp = -1

        # Motion detection for J and Z
        self.motion_detector = MotionDetector(buffer_size=30)

        # Spell mode state
        self.spell_mode = spell_mode
        self.current_word = ""
        self.word_history = []           # Completed words
        self.locked_letter = None        # Last committed letter
        self.candidate_letter = None     # Letter being evaluated
        self.stable_count = 0            # Frames the candidate has been stable
        self.no_hand_count = 0           # Consecutive frames with no hand
        self.cooldown_remaining = 0      # Frames to wait after locking a letter
        self.letter_flash_frames = 0     # Frames remaining for lock flash effect

        # Spell mode tunable thresholds
        self.STABLE_THRESHOLD = 15       # Frames to lock a letter (~0.5s at 30fps)
        self.NO_HAND_THRESHOLD = 30      # Frames of no-hand to finalize word (~1s)
        self.COOLDOWN_FRAMES = 10        # Ignore period after locking (~0.33s)
        self.REPEAT_THRESHOLD = 25       # Extra frames holding locked letter to repeat (~0.8s)
        self.MIN_LOCK_CONFIDENCE = 0.6   # Minimum confidence to accept a letter
        self.MAX_WORD_LENGTH = 30        # Safety cap
        self.MAX_HISTORY = 5             # Max completed words to display
        self.repeat_hold_count = 0       # Frames held on the same locked letter

        # Autocomplete
        self.wordbank = self._load_wordbank()
        self.autocomplete_suggestion = None  # Current suggestion shown to user
        self.MIN_PREFIX_LEN = 2              # Min letters before suggesting

    def _load_wordbank(self):
        """Load the word bank from file for autocomplete."""
        if not os.path.exists(WORDBANK_PATH):
            print(f"Word bank not found at {WORDBANK_PATH}, autocomplete disabled.")
            return []
        with open(WORDBANK_PATH, "r") as f:
            words = [line.strip().upper() for line in f if line.strip()]
        words.sort()
        print(f"Loaded {len(words)} words for autocomplete.")
        return words

    def _get_autocomplete(self):
        """Get the best autocomplete suggestion for the current word prefix."""
        if (not self.wordbank
                or len(self.current_word) < self.MIN_PREFIX_LEN):
            return None

        prefix = self.current_word.upper()
        # Find all matches
        matches = [w for w in self.wordbank if w.startswith(prefix) and w != prefix]

        if not matches:
            return None

        # Return shortest match (most likely intended word)
        return min(matches, key=len)

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

    def _update_spell_state(self, prediction, confidence):
        """Update the word spelling state machine.

        States:
            - No hand detected: increment no_hand_count, finalize word if threshold hit
            - Hand detected + cooldown active: decrement cooldown, skip processing
            - Hand detected + new candidate: start counting stability
            - Hand detected + stable candidate: lock letter when threshold hit
        """
        if prediction is None or prediction == "nothing":
            # No hand / nothing detected
            self.no_hand_count += 1
            self.candidate_letter = None
            self.stable_count = 0
            self.cooldown_remaining = 0

            if self.no_hand_count >= self.NO_HAND_THRESHOLD:
                self._finalize_word()
                self.autocomplete_suggestion = None
            return

        # Hand is present
        self.no_hand_count = 0

        # Handle special classes
        if prediction == "space":
            if self.cooldown_remaining <= 0:
                # Accept autocomplete suggestion if available
                if self.autocomplete_suggestion and self.current_word:
                    self.current_word = self.autocomplete_suggestion
                    self.letter_flash_frames = 12
                self._finalize_word()
                self.autocomplete_suggestion = None
                self.cooldown_remaining = self.COOLDOWN_FRAMES
            else:
                self.cooldown_remaining -= 1
            return

        if prediction == "del":
            if self.cooldown_remaining <= 0 and self.current_word:
                self.current_word = self.current_word[:-1]
                self.locked_letter = None
                self.cooldown_remaining = self.COOLDOWN_FRAMES
                self.letter_flash_frames = 8
                self.autocomplete_suggestion = self._get_autocomplete()
            else:
                self.cooldown_remaining -= 1
            return

        # Cooldown after locking a letter
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return

        # If this is the same as the already locked letter, count towards repeat
        if prediction == self.locked_letter:
            self.repeat_hold_count += 1
            self.candidate_letter = None
            self.stable_count = 0

            if (self.repeat_hold_count >= self.REPEAT_THRESHOLD
                    and confidence >= self.MIN_LOCK_CONFIDENCE
                    and len(self.current_word) < self.MAX_WORD_LENGTH):
                # Repeat the same letter
                self.current_word += prediction
                self.repeat_hold_count = 0
                self.cooldown_remaining = self.COOLDOWN_FRAMES
                self.letter_flash_frames = 12
                self.autocomplete_suggestion = self._get_autocomplete()
            return

        # Different letter — reset repeat counter
        self.repeat_hold_count = 0

        # Check if this is the same candidate building stability
        if prediction == self.candidate_letter:
            self.stable_count += 1

            if (self.stable_count >= self.STABLE_THRESHOLD
                    and confidence >= self.MIN_LOCK_CONFIDENCE
                    and len(self.current_word) < self.MAX_WORD_LENGTH):
                # Lock this letter
                self.current_word += prediction
                self.locked_letter = prediction
                self.candidate_letter = None
                self.stable_count = 0
                self.repeat_hold_count = 0
                self.cooldown_remaining = self.COOLDOWN_FRAMES
                self.letter_flash_frames = 12  # Visual feedback frames
                self.autocomplete_suggestion = self._get_autocomplete()
        else:
            # New candidate letter
            self.candidate_letter = prediction
            self.stable_count = 1

    def _finalize_word(self):
        """Finalize the current word and add to history."""
        if self.current_word:
            self.word_history.append(self.current_word)
            if len(self.word_history) > self.MAX_HISTORY:
                self.word_history.pop(0)
            self.current_word = ""
        self.locked_letter = None
        self.candidate_letter = None
        self.stable_count = 0
        self.repeat_hold_count = 0
        self.autocomplete_suggestion = None

    def run(self, camera_index=0):
        """Run the real-time recognition loop."""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        mode_label = "Spell Mode" if self.spell_mode else "Letter Mode"
        print("\n" + "=" * 50)
        print(f"ASL Real-Time Recognition ({mode_label})")
        print("=" * 50)
        print("Show your hand to the camera to detect ASL letters")
        if self.spell_mode:
            print("Hold a letter steady to lock it into the word")
            print("Lower your hand to finalize the word")
            print("Sign 'space' to accept autocomplete or finish a word")
            print("Sign 'del' to backspace")
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

            # Resize for MediaPipe processing (optimization)
            # MediaPipe doesn't need high res; 320x240 is usually plenty for landmarks
            process_w, process_h = 320, 240
            frame_small = cv2.resize(frame, (process_w, process_h))
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Process frame asynchronously
            frame_timestamp += 33  # ~30fps
            self.detector.detect_async(mp_image, frame_timestamp)

            # Get current prediction from state
            prediction, confidence = self.current_prediction

            # Use latest available result
            if self.latest_result and self.latest_result.hand_landmarks:
                # Only run classifier if we have a new result
                if self.result_timestamp > self.last_processed_timestamp:
                    # For ASL we only care about the first detected hand
                    hand_landmarks = self.latest_result.hand_landmarks[0]

                    # Feed landmarks to motion detector
                    self.motion_detector.add_frame(hand_landmarks)

                    # Get prediction
                    landmarks = self.normalize_landmarks(hand_landmarks)
                    landmarks = landmarks.reshape(1, -1)

                    pred = self.clf.predict(landmarks)[0]
                    probs = self.clf.predict_proba(landmarks)[0]
                    conf = np.max(probs)

                    self.current_prediction = self.get_smoothed_prediction(pred, conf)
                    self.last_processed_timestamp = self.result_timestamp

                    # Check for motion-based letters (J, Z)
                    motion_letter, motion_conf = self.motion_detector.classify_motion()
                    if motion_letter:
                        self.current_prediction = (motion_letter, motion_conf)
                        # Clear static prediction history to avoid interference
                        self.prediction_history = []

                # Draw landmarks (every frame for smooth visual)
                for hand_landmarks in self.latest_result.hand_landmarks:
                    self.draw_landmarks_on_image(frame, hand_landmarks)

                # Check if motion detection has a recent result to display
                motion_display, motion_disp_conf = self.motion_detector.get_display_detection()
                if motion_display:
                    prediction, confidence = motion_display, motion_disp_conf
                else:
                    prediction, confidence = self.current_prediction
            else:
                # Clear history when no hand detected
                self.prediction_history = []
                self.current_prediction = (None, 0.0)
                self.motion_detector.clear()
                prediction, confidence = None, 0.0

            # Update spell mode state machine
            if self.spell_mode:
                self._update_spell_state(prediction, confidence)
                if self.letter_flash_frames > 0:
                    self.letter_flash_frames -= 1

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
        """Route to the appropriate UI based on mode."""
        # Draw motion trajectory trails (both modes)
        self._draw_trajectory(frame, "index", (255, 200, 0))   # Light blue for index (BGR)
        self._draw_trajectory(frame, "pinky", (255, 0, 200))   # Magenta for pinky (BGR)

        if self.spell_mode:
            self._draw_spell_ui(frame, prediction, confidence, fps)
        else:
            self._draw_letter_ui(frame, prediction, confidence, fps)

    def _draw_letter_ui(self, frame, prediction, confidence, fps):
        """Draw the single-letter UI overlay."""
        h, w = frame.shape[:2]

        # Optimized semi-transparent background (ROI only)
        roi = frame[h - 100 : h, 0:w]
        overlay = np.zeros_like(roi)
        cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, roi)

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

        # Motion detection indicator
        motion_disp, _ = self.motion_detector.get_display_detection()
        if motion_disp:
            cv2.putText(
                frame,
                f"MOTION: {motion_disp}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_spell_ui(self, frame, prediction, confidence, fps):
        """Draw the spell-mode UI with word accumulation display."""
        h, w = frame.shape[:2]

        # --- Top bar: mode label + FPS ---
        top_bar_h = 40
        roi_top = frame[0:top_bar_h, 0:w]
        overlay_top = np.zeros_like(roi_top)
        cv2.addWeighted(overlay_top, 0.6, roi_top, 0.4, 0, roi_top)

        cv2.putText(
            frame, "SPELL MODE", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"FPS: {fps:.0f}", (w - 120, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "Press 'Q' to quit", (w // 2 - 70, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Motion detection indicator
        motion_disp, _ = self.motion_detector.get_display_detection()
        if motion_disp:
            cv2.putText(
                frame, f"MOTION: {motion_disp}", (w - 200, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )

        # --- Word history (completed words) ---
        if self.word_history:
            history_text = " ".join(self.word_history)
            cv2.putText(
                frame, history_text, (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA,
            )

        # --- Current word being built ---
        word_y = 75 if not self.word_history else 110
        display_word = self.current_word if self.current_word else ""

        # Blinking cursor
        cursor = "|" if (int(time.time() * 3) % 2 == 0) else " "
        word_display = display_word + cursor

        # Flash effect when letter just locked
        if self.letter_flash_frames > 0:
            word_color = (0, 255, 128)  # Bright green flash
        elif self.current_word:
            word_color = (255, 255, 255)  # White
        else:
            word_color = (150, 150, 150)  # Grey when empty

        cv2.putText(
            frame, word_display, (15, word_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, word_color, 2, cv2.LINE_AA,
        )

        # --- Autocomplete suggestion ---
        if self.autocomplete_suggestion and self.current_word:
            typed_width = cv2.getTextSize(
                display_word, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )[0][0]
            remaining = self.autocomplete_suggestion[len(self.current_word):]
            cv2.putText(
                frame, remaining, (15 + typed_width, word_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, "(sign 'space' to accept)",
                (15, word_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA,
            )

        # --- Bottom bar: current detection + stability progress ---
        bottom_bar_h = 120
        roi_bottom = frame[h - bottom_bar_h : h, 0:w]
        overlay_bottom = np.zeros_like(roi_bottom)
        cv2.addWeighted(overlay_bottom, 0.6, roi_bottom, 0.4, 0, roi_bottom)

        if prediction and prediction not in ("nothing",):
            letter_size = 1.8
            letter_thickness = 3
            text = prediction
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, letter_size, letter_thickness
            )[0]

            if confidence > 0.8:
                color = (0, 255, 0)
            elif confidence > 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)

            cv2.putText(
                frame, text, (20, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, letter_size, color,
                letter_thickness, cv2.LINE_AA,
            )

            conf_text = f"{confidence * 100:.1f}%"
            cv2.putText(
                frame, conf_text, (20 + text_size[0] + 10, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Stability progress bar
            bar_x = 20 + text_size[0] + 10
            bar_y = h - 40
            bar_width = w - bar_x - 30
            bar_height = 18

            if bar_width > 50:
                cv2.rectangle(
                    frame, (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (60, 60, 60), -1,
                )

                if self.candidate_letter == prediction and self.cooldown_remaining <= 0:
                    progress = min(self.stable_count / self.STABLE_THRESHOLD, 1.0)
                elif prediction == self.locked_letter and self.repeat_hold_count > 0:
                    progress = min(self.repeat_hold_count / self.REPEAT_THRESHOLD, 1.0)
                elif prediction == self.locked_letter:
                    progress = 1.0
                else:
                    progress = 0.0

                fill_color = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
                cv2.rectangle(
                    frame, (bar_x, bar_y),
                    (bar_x + int(bar_width * progress), bar_y + bar_height),
                    fill_color, -1,
                )

                if prediction == self.locked_letter and self.repeat_hold_count > 0:
                    label = f"hold to repeat... {int(progress * 100)}%"
                elif prediction == self.locked_letter:
                    label = "LOCKED (hold to repeat or change sign)"
                elif self.cooldown_remaining > 0:
                    label = "cooldown..."
                else:
                    label = f"hold steady... {int(progress * 100)}%"

                cv2.putText(
                    frame, label, (bar_x + 5, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
                )

            # Confidence bar at the very bottom
            cbar_y = h - 15
            cbar_w = w - 40
            cv2.rectangle(
                frame, (20, cbar_y), (20 + cbar_w, cbar_y + 10),
                (100, 100, 100), -1,
            )
            cv2.rectangle(
                frame, (20, cbar_y),
                (20 + int(cbar_w * confidence), cbar_y + 10),
                color, -1,
            )
        else:
            if self.current_word:
                remaining = max(0, self.NO_HAND_THRESHOLD - self.no_hand_count)
                status = f"No hand - word finalizes in {remaining} frames"
                cv2.putText(
                    frame, status, (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame, "Show your hand to start spelling",
                    (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA,
                )

    def _draw_trajectory(self, frame, which, color):
        """Draw the fingertip trajectory trail on the frame."""
        h, w = frame.shape[:2]
        points = self.motion_detector.get_trajectory_points(w, h, which=which)

        if len(points) < 3:
            return

        # Draw trail with fading opacity (older = thinner/dimmer)
        for i in range(1, len(points)):
            # Thickness increases towards the tip
            thickness = max(1, int(i / len(points) * 4))
            # Fade alpha via color intensity
            alpha = i / len(points)
            faded_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, points[i - 1], points[i], faded_color, thickness)


def main(model_path="models/asl_classifier.pkl", camera_index=0, spell_mode=False):
    """Main entry point for real-time recognition."""
    recognizer = RealtimeASLRecognizer(model_path, spell_mode=spell_mode)
    recognizer.run(camera_index)


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/asl_classifier.pkl"
    camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    spell = "--spell" in sys.argv

    main(model_path, camera_index, spell_mode=spell)
