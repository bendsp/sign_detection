# AGENTS.md - Development Guide

This document provides context for AI agents and developers working on this project.

## Project Overview

**ASL Sign Language Detection** - A real-time American Sign Language alphabet recognition system using computer vision and machine learning.

### Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────┐
│   Webcam    │────▶│  MediaPipe Hands │────▶│  Feature Engine │────▶│ Classifier │
│   (or img)  │     │  (21 landmarks)  │     │  (93 features)  │     │   (RF)     │
└─────────────┘     └──────────────────┘     └─────────────────┘     └────────────┘
                           │                        │                       │
                           ▼                        ▼                       ▼
                    x, y, z coords           Engineered features      Predicted letter
                    for each point           (angles, distances,      + confidence
                                             depth comparisons)
```

### Key Design Decisions

1. **MediaPipe Tasks API** (not legacy `mp.solutions`)
   - We use the new `mediapipe.tasks` API (v0.10+)
   - The legacy `mp.solutions.hands` is deprecated
   - Hand landmarker model is downloaded automatically from Google's model hub

2. **Engineered Features** (not just raw landmarks)
   - Raw 21 landmarks (63 features) lose depth information
   - We add 30 engineered features to capture:
     - Thumb-to-fingertip distances
     - Z-depth comparisons (thumb in front/behind fingers)
     - Finger curl angles
     - Cross-product orientation for thumb crossing
   - This improved accuracy from 92% → 97.5%

3. **Random Forest Classifier**
   - Fast training (seconds, not hours)
   - Good accuracy for this feature set
   - Easy to interpret feature importance

4. **Real-time Architecture**
   - Uses `LIVE_STREAM` mode with async callbacks
   - Prediction smoothing via 5-frame sliding window
   - Manual landmark drawing (no `mp.solutions.drawing_utils`)

## File Guide

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.py` | CLI entry point | Commands: download, extract, train, predict, realtime |
| `src/extract_features.py` | Feature extraction | `HandLandmarkExtractor`, `compute_engineered_features()` |
| `src/train.py` | Model training | `train_classifier()` |
| `src/predict.py` | Single image inference | `predict_image()`, `display_prediction()` |
| `src/realtime.py` | Webcam recognition | `RealtimeASLRecognizer` class |

## Current Limitations

1. **M and N confusion** (~80% accuracy)
   - Both require thumb under fingers
   - Subtle position differences hard to detect from 2D camera

2. **J and Z require motion**
   - These letters involve hand movement
   - Currently detected as static poses only

3. **Lighting sensitivity**
   - MediaPipe needs reasonable lighting
   - Hand detection fails in very dark conditions

4. **Single camera depth**
   - Z-coordinates are estimated, not true depth
   - Stereo camera or depth sensor would help

## Development Workflow

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Get data (requires Kaggle API key)
python main.py download

# Feature extraction (use sample count for quick tests)
python main.py extract 200    # Quick test
python main.py extract        # Full dataset (~15-20 min)

# Train
python main.py train

# Test
python main.py realtime       # Live webcam
python main.py predict <img>  # Single image
```

## Testing Changes

When modifying feature extraction:
1. Update `compute_engineered_features()` in `extract_features.py`
2. Update `normalize_landmarks()` in `realtime.py` to match
3. Re-run `extract` and `train`
4. Test with `realtime`

## Code Style

- Python 3.10+
- Type hints encouraged but not enforced
- Docstrings for public functions
- Keep feature extraction logic consistent between training and inference
