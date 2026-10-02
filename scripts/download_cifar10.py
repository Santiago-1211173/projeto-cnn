"""
Downloads and extracts the CIFAR-10 dataset into data/CIFAR10/raw/.
Uses the official Python/pickle distribution from https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import sys
import urllib.request
import tarfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "CIFAR10", "raw")

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    archive_path = os.path.join(TARGET_DIR, ARCHIVE_NAME)

    # 1. Download
    if not os.path.exists(archive_path):
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}...")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)
        print(f"Downloaded to {archive_path}")
    else:
        print(f"Archive already exists: {archive_path}")

    # 2. Extract
    print("Extracting...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=TARGET_DIR)
    print(f"Extracted to {TARGET_DIR}")

    # 3. Verify
    batch_dir = os.path.join(TARGET_DIR, "cifar-10-batches-py")
    expected_files = ["data_batch_1", "data_batch_2", "data_batch_3",
                      "data_batch_4", "data_batch_5", "test_batch", "batches.meta"]
    for f in expected_files:
        path = os.path.join(batch_dir, f)
        if os.path.exists(path):
            print(f"  OK: {f}")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nCIFAR-10 dataset ready.")


if __name__ == "__main__":
    main()
