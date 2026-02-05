# ASL Sign Language Detection

Real-time American Sign Language (ASL) alphabet recognition using MediaPipe hand landmarks and machine learning.

## Features

- Detects ASL alphabet letters (A-Z) plus special gestures (delete, space, nothing)
- Real-time webcam recognition with hand landmark visualization
- 97.5% accuracy using engineered features for depth-aware gesture detection
- Uses MediaPipe Tasks API for hand landmark extraction
- Random Forest classifier for fast inference

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

## Setup

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

### Real-time Recognition

```bash
python main.py realtime
```

- Show your hand to the camera
- The detected letter and confidence will be displayed
- Hand landmarks are overlaid on the video feed
- Press 'Q' to quit

### Single Image Prediction

```bash
python main.py predict path/to/image.jpg
```

## Project Structure

```
sign_detection/
├── data/                    # Dataset and extracted features
├── models/                  # Trained models
├── src/
│   ├── extract_features.py  # MediaPipe landmark extraction + engineered features
│   ├── train.py             # Classifier training
│   ├── predict.py           # Single image prediction
│   └── realtime.py          # Webcam real-time recognition
├── main.py                  # CLI entry point
└── requirements.txt
```

## Technical Details

### Features (93 total)

- **Base landmarks**: 21 hand points × 3 coordinates (x, y, z) = 63 features
- **Engineered features** (30): Thumb-finger distances, z-depth comparisons, finger curl angles, spatial relationships

### Why Engineered Features?

Some ASL letters (like M, N, Q, J) require understanding whether the thumb is in front of or behind the fingers. The additional engineered features capture:

- Relative z-depth between thumb and fingertips
- Finger curl/extension angles
- Thumb crossing orientation
- Inter-finger distances

This improves accuracy from ~92% to ~97.5%.

## Limitations

- Letters J and Z require motion (currently detected as static poses)
- M and N can be confused due to subtle thumb positioning
- Requires good lighting for reliable hand detection

## License

MIT
