#!/usr/bin/env python3
"""
ASL Sign Language Recognition - Main CLI Entry Point

Usage:
    python main.py download          - Download the ASL Alphabet dataset from Kaggle
    python main.py extract            - Extract hand landmarks from dataset images
    python main.py train              - Train the classifier on extracted features
    python main.py predict <img>      - Predict the ASL letter in an image
    python main.py realtime           - Launch real-time webcam recognition
    python main.py realtime --spell   - Launch real-time with spell/word mode
"""

import sys
import os

# Ensure we're running from the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)


def print_usage():
    """Print usage information."""
    print(__doc__)


def cmd_download():
    """Download the ASL Alphabet dataset from Kaggle."""
    import subprocess

    print("Downloading ASL Alphabet dataset from Kaggle...")
    os.makedirs("data", exist_ok=True)

    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "grassknoted/asl-alphabet",
            "-p",
            "data",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return

    print(result.stdout)

    # Unzip
    print("Extracting dataset...")
    subprocess.run(["unzip", "-q", "-o", "data/asl-alphabet.zip", "-d", "data"])
    print("Done! Dataset extracted to data/asl_alphabet_train/")


def cmd_extract(sample_per_class=None):
    """Extract hand landmarks from dataset images."""
    from src.extract_features import extract_features_from_dataset

    data_dir = "data/asl_alphabet_train/asl_alphabet_train"
    output_path = "data/landmarks.pkl"

    if not os.path.exists(data_dir):
        print(f"Error: Dataset not found at {data_dir}")
        print("Run 'python main.py download' first.")
        return

    extract_features_from_dataset(data_dir, output_path, sample_per_class)


def cmd_train():
    """Train the classifier on extracted features."""
    from src.train import train_classifier

    features_path = "data/landmarks.pkl"
    model_path = "models/asl_classifier.pkl"

    if not os.path.exists(features_path):
        print(f"Error: Features not found at {features_path}")
        print("Run 'python main.py extract' first.")
        return

    train_classifier(features_path, model_path)


def cmd_predict(image_path):
    """Predict the ASL letter in an image."""
    from src.predict import display_prediction

    model_path = "models/asl_classifier.pkl"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run 'python main.py train' first.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    display_prediction(image_path, model_path)


def cmd_realtime(camera_index=0, spell_mode=False):
    """Launch real-time webcam recognition."""
    from src.realtime import main as realtime_main

    model_path = "models/asl_classifier.pkl"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run 'python main.py train' first.")
        return

    realtime_main(model_path, camera_index, spell_mode=spell_mode)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    if command == "download":
        cmd_download()

    elif command == "extract":
        sample_per_class = None
        if len(sys.argv) > 2:
            sample_per_class = int(sys.argv[2])
        cmd_extract(sample_per_class)

    elif command == "train":
        cmd_train()

    elif command == "predict":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path")
            print("Usage: python main.py predict <image_path>")
            return
        cmd_predict(sys.argv[2])

    elif command == "realtime":
        camera_index = 0
        spell_mode = "--spell" in sys.argv
        # Parse camera index from non-flag args
        for arg in sys.argv[2:]:
            if arg != "--spell":
                camera_index = int(arg)
                break
        cmd_realtime(camera_index, spell_mode=spell_mode)

    elif command in ["help", "-h", "--help"]:
        print_usage()

    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
