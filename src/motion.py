"""
Motion detection module for ASL letters that require hand movement (J and Z).

Uses Dynamic Time Warping (DTW) to compare the trajectory of key fingertips
against reference templates. Pure NumPy implementation — no extra dependencies.

J: Pinky traces a "J" hook shape (down then curve left)
Z: Index finger traces a "Z" zigzag shape (right, diagonal-left-down, right)
"""

import numpy as np
from collections import deque


# Landmark indices for tracking
INDEX_TIP = 8
PINKY_TIP = 20
WRIST = 0

# --- Reference Templates ---
# These are idealized 2D trajectories (x, y) normalized to [0, 1] range.
# Designed for a MIRRORED (flipped) camera view — right hand signing.

# J template: pinky traces downward then hooks to the left
# Starting from top, going down, then curving left and slightly up
J_TEMPLATE = np.array([
    [0.50, 0.00],  # Start top-center
    [0.50, 0.15],  # Down
    [0.50, 0.30],  # Down
    [0.48, 0.45],  # Down, slight left
    [0.45, 0.58],  # Curving left
    [0.40, 0.70],  # Curving left
    [0.32, 0.80],  # Hook left
    [0.22, 0.85],  # Hook left
    [0.12, 0.82],  # Curving up-left
    [0.05, 0.72],  # Up
    [0.02, 0.60],  # Up (end of J hook)
], dtype=np.float64)

# Z template: index finger traces a Z shape
# Right, then diagonal down-left, then right again
Z_TEMPLATE = np.array([
    [0.10, 0.10],  # Start top-left
    [0.30, 0.10],  # Right along top
    [0.50, 0.10],  # Right along top
    [0.70, 0.10],  # Right along top (end of top bar)
    [0.60, 0.25],  # Diagonal down-left
    [0.50, 0.40],  # Diagonal
    [0.40, 0.55],  # Diagonal
    [0.30, 0.70],  # Diagonal down-left (end of diagonal)
    [0.45, 0.70],  # Right along bottom
    [0.60, 0.70],  # Right along bottom
    [0.80, 0.70],  # Right along bottom (end of Z)
], dtype=np.float64)


def dtw_distance(seq1, seq2):
    """
    Compute Dynamic Time Warping distance between two 2D trajectories.

    Args:
        seq1: (N, 2) array — first trajectory
        seq2: (M, 2) array — second trajectory

    Returns:
        Normalized DTW distance (lower = more similar)
    """
    n, m = len(seq1), len(seq2)

    # Cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],      # insertion
                dtw[i, j - 1],      # deletion
                dtw[i - 1, j - 1],  # match
            )

    # Normalize by path length
    return dtw[n, m] / (n + m)


def normalize_trajectory(points):
    """
    Normalize a 2D trajectory: center, scale to unit range, resample to fixed length.

    Args:
        points: (N, 2) array of (x, y) positions

    Returns:
        (NUM_RESAMPLE_POINTS, 2) normalized trajectory, or None if invalid
    """
    NUM_RESAMPLE_POINTS = 20

    if len(points) < 5:
        return None

    points = np.array(points, dtype=np.float64)

    # Remove duplicate consecutive points
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mask = np.concatenate([[True], diffs > 1e-6])
    points = points[mask]

    if len(points) < 5:
        return None

    # Center on centroid
    centroid = np.mean(points, axis=0)
    points = points - centroid

    # Scale to [0, 1] range
    span = np.max(np.abs(points))
    if span < 1e-6:
        return None  # No meaningful movement
    points = points / span

    # Resample to fixed number of points using linear interpolation
    # Compute cumulative arc length
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum_dist = np.concatenate([[0], np.cumsum(dists)])
    total_dist = cum_dist[-1]

    if total_dist < 1e-6:
        return None

    # Interpolate at uniform arc-length intervals
    target_dists = np.linspace(0, total_dist, NUM_RESAMPLE_POINTS)
    resampled = np.zeros((NUM_RESAMPLE_POINTS, 2))
    for i, td in enumerate(target_dists):
        idx = np.searchsorted(cum_dist, td, side='right') - 1
        idx = min(idx, len(points) - 2)
        segment_len = cum_dist[idx + 1] - cum_dist[idx]
        if segment_len < 1e-8:
            t = 0.0
        else:
            t = (td - cum_dist[idx]) / segment_len
        resampled[i] = points[idx] + t * (points[idx + 1] - points[idx])

    # Re-normalize to [0, 1]
    resampled -= resampled.min(axis=0)
    span = resampled.max(axis=0) - resampled.min(axis=0)
    span[span < 1e-8] = 1.0
    resampled /= span.max()

    return resampled


class MotionDetector:
    """
    Detects motion-based ASL letters (J and Z) by tracking fingertip
    trajectories and comparing them against reference templates using DTW.
    """

    def __init__(self, buffer_size=30):
        """
        Args:
            buffer_size: Number of frames to keep in the trajectory buffer (~1s at 30fps)
        """
        self.buffer_size = buffer_size

        # Track positions of key landmarks over time
        # Each entry: (index_tip_xy, pinky_tip_xy, wrist_xy)
        self.index_buffer = deque(maxlen=buffer_size)
        self.pinky_buffer = deque(maxlen=buffer_size)
        self.wrist_buffer = deque(maxlen=buffer_size)

        # Movement detection
        self.min_movement_threshold = 0.08  # Min total displacement to consider motion
        self.dtw_threshold = 0.15           # Max DTW distance to accept a match
        self.cooldown_frames = 0            # Prevent repeated detections
        self.COOLDOWN_MAX = 30              # ~1 second cooldown after detection

        # State
        self.last_detection = None          # ("J", confidence) or ("Z", confidence)
        self.detection_display_frames = 0   # How long to keep showing the detection

        # Pre-normalize templates
        self.j_template = normalize_trajectory(J_TEMPLATE)
        self.z_template = normalize_trajectory(Z_TEMPLATE)

    def add_frame(self, hand_landmarks):
        """
        Add a frame's landmarks to the motion buffer.

        Args:
            hand_landmarks: MediaPipe hand landmarks for one hand
        """
        # Extract pixel positions of key landmarks
        index_tip = hand_landmarks[INDEX_TIP]
        pinky_tip = hand_landmarks[PINKY_TIP]
        wrist = hand_landmarks[WRIST]

        # Use normalized coordinates (0-1 range) for scale invariance
        self.index_buffer.append(np.array([index_tip.x, index_tip.y]))
        self.pinky_buffer.append(np.array([pinky_tip.x, pinky_tip.y]))
        self.wrist_buffer.append(np.array([wrist.x, wrist.y]))

    def clear(self):
        """Clear the motion buffer (called when hand disappears)."""
        self.index_buffer.clear()
        self.pinky_buffer.clear()
        self.wrist_buffer.clear()

    def _compute_total_movement(self, buffer):
        """Compute the total displacement of a trajectory."""
        if len(buffer) < 5:
            return 0.0
        points = np.array(buffer)
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.sum(diffs)

    def _compute_displacement(self, buffer):
        """Compute straight-line distance from start to end."""
        if len(buffer) < 5:
            return 0.0
        return np.linalg.norm(np.array(buffer[-1]) - np.array(buffer[0]))

    def classify_motion(self):
        """
        Check if the current trajectory buffer matches J or Z.

        Returns:
            Tuple of (letter, confidence) if detected, or (None, 0.0) if no motion letter.
            Confidence is 0.0-1.0 (higher = better match).
        """
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return None, 0.0

        # Check if there's enough meaningful movement
        index_movement = self._compute_total_movement(self.index_buffer)
        pinky_movement = self._compute_total_movement(self.pinky_buffer)

        # Also check wrist is relatively still (motion should be fingers, not whole hand)
        wrist_movement = self._compute_total_movement(self.wrist_buffer)

        results = []

        # --- Check Z (index finger traces Z) ---
        # Finger must move significantly AND more than the wrist (not whole-hand motion)
        if (index_movement > self.min_movement_threshold
                and (wrist_movement < 1e-6 or index_movement > wrist_movement * 1.5)
                and len(self.index_buffer) >= 15):

            trajectory = normalize_trajectory(list(self.index_buffer))
            if trajectory is not None and self.z_template is not None:
                dist = dtw_distance(trajectory, self.z_template)
                if dist < self.dtw_threshold:
                    confidence = max(0.0, 1.0 - dist / self.dtw_threshold)
                    results.append(("Z", confidence, dist))

        # --- Check J (pinky finger traces J) ---
        # Finger must move significantly AND more than the wrist (not whole-hand motion)
        if (pinky_movement > self.min_movement_threshold
                and (wrist_movement < 1e-6 or pinky_movement > wrist_movement * 1.5)
                and len(self.pinky_buffer) >= 15):

            trajectory = normalize_trajectory(list(self.pinky_buffer))
            if trajectory is not None and self.j_template is not None:
                dist = dtw_distance(trajectory, self.j_template)
                if dist < self.dtw_threshold:
                    confidence = max(0.0, 1.0 - dist / self.dtw_threshold)
                    results.append(("J", confidence, dist))

        if not results:
            return None, 0.0

        # Return the best match
        best = min(results, key=lambda r: r[2])
        letter, confidence, _ = best

        # Trigger cooldown and clear buffer to prevent repeat detection
        self.cooldown_frames = self.COOLDOWN_MAX
        self.last_detection = (letter, confidence)
        self.detection_display_frames = 45  # Show for ~1.5 seconds
        self.clear()

        return letter, confidence

    def get_display_detection(self):
        """
        Get the current motion detection for display purposes.
        Returns (letter, confidence) or (None, 0.0) if nothing to display.
        """
        if self.detection_display_frames > 0:
            self.detection_display_frames -= 1
            return self.last_detection
        return None, 0.0

    def get_trajectory_points(self, frame_w, frame_h, which="index"):
        """
        Get the current trajectory as pixel coordinates for drawing.

        Args:
            frame_w, frame_h: Frame dimensions
            which: "index" or "pinky"

        Returns:
            List of (x, y) tuples, or empty list
        """
        buffer = self.index_buffer if which == "index" else self.pinky_buffer
        if len(buffer) < 3:
            return []

        points = []
        for pt in buffer:
            x = int(pt[0] * frame_w)
            y = int(pt[1] * frame_h)
            points.append((x, y))
        return points
