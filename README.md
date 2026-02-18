# ASL Sign Language Detection

Real-time American Sign Language (ASL) alphabet recognition using MediaPipe hand landmarks and machine learning.

## Features

- Detects ASL alphabet letters (A-Z) plus special gestures (delete, space, nothing)
- **Real-time webcam recognition** with hand landmark visualization
- **Spell mode** — build words letter-by-letter with autocomplete suggestions
- **Motion detection** for J and Z using DTW trajectory matching
- **98.7% accuracy** using 117 engineered features trained on the full dataset (63,580 samples)
- Uses MediaPipe Tasks API for hand landmark extraction
- Random Forest classifier for fast inference

## Quick Start

If you just want to run the recognizer, the trained model is included via Git LFS:

```bash
git clone git@github.com:bendsp/sign_detection.git
cd sign_detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py realtime
```

## Installation

```bash
# Clone the repository
git clone git@github.com:bendsp/sign_detection.git
cd sign_detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training From Scratch (Optional)

The trained model is already included in the repo. Only follow these steps if you want to retrain.

### 1. Download the Dataset

You'll need a Kaggle account. Get your API key from https://www.kaggle.com/settings and place `kaggle.json` in `~/.kaggle/`.

```bash
python main.py download
```

### 2. Extract Features

```bash
python main.py extract        # Full dataset (~15-20 min)
python main.py extract 200    # Quick test with 200 samples/class
```

### 3. Train the Model

```bash
python main.py train
```

## Usage

### Real-time Letter Recognition

```bash
python main.py realtime
```

- Show your hand to the camera
- The detected letter and confidence are displayed
- Hand landmarks are overlaid on the video feed
- Trajectory trails show finger movement (for J/Z detection)
- Press 'Q' to quit

### Spell Mode (Word Building)

```bash
python main.py realtime --spell
```

- Hold a letter steady (~0.5s) to lock it into the current word
- Keep holding the same letter to repeat it (e.g. "LL")
- Sign `space` to finalize the word (or accept an autocomplete suggestion)
- Sign `del` to backspace
- Lower your hand for ~1s to finalize the word
- Autocomplete suggestions appear after 2+ letters (from a built-in word bank)

### Single Image Prediction

```bash
python main.py predict path/to/image.jpg
```

## Project Structure

```
sign_detection/
├── data/
│   └── wordbank.txt             # ~850 common English words for autocomplete
├── models/
│   └── asl_classifier.pkl       # Trained Random Forest model (Git LFS)
├── src/
│   ├── extract_features.py      # MediaPipe landmark extraction + 117 engineered features
│   ├── motion.py                # DTW-based motion detection for J and Z
│   ├── train.py                 # Classifier training
│   ├── predict.py               # Single image prediction
│   └── realtime.py              # Webcam recognition (letter mode + spell mode)
├── main.py                      # CLI entry point
├── requirements.txt
└── AGENTS.md                    # Development guide for AI agents
```

## Technical Details

### Features (117 total)

- **Base landmarks**: 21 hand points × 3 coordinates (x, y, z) = 63 features
- **Engineered features** (54):
  - Thumb-to-fingertip distances and z-depth comparisons
  - Finger curl angles (PIP and DIP joints)
  - Thumb crossing orientation (cross-product)
  - Thumb-to-PIP/DIP distances (M/N/E/S/T disambiguation)
  - Thumb vs fingertip y-position (finger drape detection)
  - Thumb position relative to middle/ring MCP (finger valley detection)
  - Finger drape count proxy (M=3, N=2, others=0)
  - Thumb-to-palm-plane signed distance (front vs behind)

### Motion Detection (J and Z)

Letters J and Z require hand movement, not just a static pose. The system uses **Dynamic Time Warping (DTW)** to match fingertip trajectories against reference templates:
- **J**: Pinky finger traces a J-shaped arc
- **Z**: Index finger traces a Z-shaped zigzag

Trajectories are visualized as colored trails on the video feed.

### Accuracy

Trained on the full [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (3,000 images/class, 63,580 total samples after filtering):

| Metric | Value |
|--------|-------|
| **Overall accuracy** | 98.74% |
| **Weakest letter** | N (89% recall) |
| **Previous accuracy** (93 features, 200 samples/class) | 97.53% |

### Why Engineered Features?

Some ASL letters (M, N, S, T, E, A) are all closed-fist variants differing only in thumb position. The 54 engineered features capture:

- Whether the thumb is in front of or behind the palm plane
- How many fingers drape over the thumb (M=3, N=2)
- Which finger valley the thumb sits in
- Fingertip curl tightness at the DIP joint

This improved M from 82% → 98% and N from 80% → 89%.

## Limitations

- **N detection** (~89%) — still occasionally confused with M due to subtle thumb positioning
- **J and Z** — detected via motion templates; require deliberate finger movement
- **Lighting** — MediaPipe needs reasonable lighting for reliable hand detection
- **Single camera depth** — z-coordinates are estimated, not true depth

## License

MIT
