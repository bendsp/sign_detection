#!/usr/bin/env python3
"""
ASL Sign Language Recognition - Main CLI Entry Point

Usage:
    python main.py download          - Download the ASL Alphabet dataset from Kaggle
    python main.py download --force   - Re-download even if data already exists
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


def cmd_download(force=False):
    """Download the ASL Alphabet dataset from Kaggle."""
    import subprocess
    import shutil
    import zipfile

    dataset_dir = "data/asl_alphabet_train/asl_alphabet_train"
    zip_path = "data/asl-alphabet.zip"

    if os.path.exists(dataset_dir) and not force:
        print(f"Dataset already exists at {dataset_dir}, skipping download.")
        print("Use --force to re-download.")
        return

    if force and os.path.exists(dataset_dir):
        print(f"Removing existing dataset at {dataset_dir}...")
        shutil.rmtree(dataset_dir)
    if force and os.path.exists(zip_path):
        os.remove(zip_path)

    print("Downloading ASL Alphabet dataset from Kaggle...")
    os.makedirs("data", exist_ok=True)

    # Support KAGGLE_API_TOKEN env var (new token format)
    api_token = os.environ.get("KAGGLE_API_TOKEN")
    if api_token:
        os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME", "_")
        os.environ["KAGGLE_KEY"] = api_token

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("Authenticated with Kaggle API.")
        api.dataset_download_files("grassknoted/asl-alphabet", path="data", quiet=False)
    except Exception as e:
        print(f"Kaggle API error: {e}")
        print("\nTo fix this, do ONE of the following:")
        print("  1. Place your kaggle.json in ~/.kaggle/kaggle.json")
        print("     (Download from https://www.kaggle.com/settings → Create New Token)")
        print("  2. Set the KAGGLE_API_TOKEN environment variable:")
        print("     export KAGGLE_API_TOKEN=your_token_here")
        return

    # Unzip
    if os.path.exists(zip_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall("data")
        print("Done! Dataset extracted to data/asl_alphabet_train/")
    else:
        print("Error: Download completed but zip file not found.")


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
        force = "--force" in sys.argv
        cmd_download(force=force)

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
