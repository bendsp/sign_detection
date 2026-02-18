# ASL Sign Language Detection

Real-time American Sign Language (ASL) alphabet recognition using MediaPipe hand landmarks and machine learning.

## Features

- Detects ASL alphabet letters (A-Z) plus special gestures (delete, space, nothing)
- Real-time webcam recognition with hand landmark visualization
- 97.5% accuracy using engineered features for depth-aware gesture detection
- Uses MediaPipe Tasks API for hand landmark extraction
- Random Forest classifier for fast inference
- Spell mode: accumulate letters into words with autocomplete

---

## Quick Start

There are two ways to run this project:
1. **Docker** — recommended for training (reproducible, no dependency hassle)
2. **Local** — required for real-time webcam recognition

You can mix both: train with Docker, then run real-time locally.

---

## Option 1: Docker (Training Pipeline)

Docker handles all dependencies automatically. Use this for downloading data, extracting features, and training the model.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed
- A [Kaggle](https://www.kaggle.com) account (for dataset download only)

### Setup Kaggle Credentials

You need Kaggle credentials to download the dataset. Choose one method:

**Option A — `kaggle.json` file:**
1. Go to https://www.kaggle.com/settings
2. Click **"Create New Token"** — downloads a `kaggle.json` file
3. Place it in your home directory:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Option B — Environment variable (simpler):**
```bash
export KAGGLE_API_TOKEN=your_token_here
```

### Build & Run

```bash
# Build the Docker image
docker compose build

# Step 1: Download the ASL Alphabet dataset (~1GB)
docker compose run --rm download

# Step 2: Extract hand landmarks from images
docker compose run --rm extract          # Full dataset (~15-20 min)
docker compose run --rm extract-quick    # Quick test with 200 samples/class (~2-3 min)

# Step 3: Train the classifier
docker compose run --rm train
```

### How Docker Shares Data with Your Machine

Docker uses **volume mounts** to share the `data/` and `models/` folders between the container and your local machine. This means:

- If you already downloaded or extracted data **locally**, Docker will see it and skip those steps automatically.
- Any models trained **inside Docker** are saved directly to your local `models/` folder.
- You can train with Docker and then run `python main.py realtime` locally — it will use the same model.

---

## Option 2: Local Setup

Use this for **real-time webcam recognition** or if you prefer to run everything locally.

### Prerequisites

- Python 3.10+
- A webcam (for real-time mode)

### Install

```bash
# Clone the repo
git clone git@github.com:bendsp/sign_detection.git
cd sign_detection

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Kaggle Credentials

Same as above — see [Setup Kaggle Credentials](#setup-kaggle-credentials).

### Run the Pipeline

```bash
# Step 1: Download dataset
python main.py download

# Step 2: Extract features
python main.py extract          # Full dataset (~15-20 min)
python main.py extract 200      # Quick test with 200 samples/class

# Step 3: Train
python main.py train
```

---

## Usage

### Real-time Webcam Recognition (Local Only)

```bash
source venv/bin/activate

# Letter-by-letter mode
python main.py realtime

# Spell mode (accumulates letters into words)
python main.py realtime --spell
```

- Show your hand to the camera
- The detected letter and confidence will be displayed
- Hand landmarks are overlaid on the video feed
- Press **Q** to quit

### Single Image Prediction

```bash
python main.py predict path/to/image.jpg
```

---

## Interactive Notebook

For a step-by-step walkthrough with visualizations (dataset exploration, training, evaluation charts), open the Jupyter notebook:

```bash
pip install jupyter
jupyter notebook notebooks/train_and_test.ipynb
```

The notebook covers the full pipeline and includes confusion matrix, per-class F1 scores, and feature importance analysis.

---

## Project Structure

```
sign_detection/
├── data/                    # Dataset and extracted features (gitignored)
├── models/                  # Trained models (gitignored)
├── notebooks/
│   └── train_and_test.ipynb # Interactive training & evaluation guide
├── src/
│   ├── extract_features.py  # MediaPipe landmark extraction + engineered features
│   ├── train.py             # Classifier training
│   ├── predict.py           # Single image prediction
│   ├── realtime.py          # Webcam real-time recognition
│   └── motion.py            # Motion detection for J and Z letters
├── main.py                  # CLI entry point
├── Dockerfile               # Docker image for training
├── docker-compose.yml       # Docker Compose services
└── requirements.txt
```

## Technical Details

### Features (117 total)

- **Base landmarks**: 21 hand points × 3 coordinates (x, y, z) = 63 features
- **Engineered features** (54): Thumb-finger distances, z-depth comparisons, finger curl angles, spatial relationships, drape detection, palm-plane distance

### Why Engineered Features?

Some ASL letters (like M, N, Q, J) require understanding whether the thumb is in front of or behind the fingers. The additional engineered features capture:

- Relative z-depth between thumb and fingertips
- Finger curl/extension angles
- Thumb crossing orientation
- Inter-finger distances
- Thumb-to-PIP/DIP distances for M/N/E/S/T disambiguation

This improves accuracy from ~92% to ~97.5%.

## Limitations

- Letters J and Z require motion (motion detection is implemented but still experimental)
- M and N can be confused due to subtle thumb positioning
- Requires good lighting for reliable hand detection

## License

MIT
