"""
Downloads and extracts the CIFAR-100 dataset into data/CIFAR100/raw/.
Uses the official Python/pickle distribution from https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import sys
import urllib.request
import tarfile
import argparse
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import CIFAR100_DATA_DIR

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
ARCHIVE_NAME = "cifar-100-python.tar.gz"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def download_and_extract_cifar100(target_dir: str, force: bool = False) -> None:
    """
    Downloads and extracts the CIFAR-100 archive into target_dir.

    Args:
        target_dir: Directory where the dataset will be placed.
        force: If True, re-downloads even if the archive or files already exist.
    """
    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)

    # 1. Download archive if missing or forced
    if not os.path.exists(archive_path) or force:
        logger.info(f"Downloading CIFAR-100 from {CIFAR100_URL}...")
        try:
            urllib.request.urlretrieve(CIFAR100_URL, archive_path)
            logger.info(f"Downloaded to {archive_path}")
        except Exception as e:
            logger.error(f"Failed to download CIFAR-100: {e}")
            if os.path.exists(archive_path):
                os.remove(archive_path)
            sys.exit(1)
    else:
        logger.info(f"Archive already exists: {archive_path}")

    # 2. Extract archive
    logger.info("Extracting CIFAR-100 archive...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        logger.info(f"Extracted to {target_dir}")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)

    # 3. Verify extracted contents
    batch_dir = os.path.join(target_dir, "cifar-100-python")
    expected_files = ["train", "test", "meta"]
    missing = []
    for filename in expected_files:
        path_in_sub = os.path.join(batch_dir, filename)
        path_in_root = os.path.join(target_dir, filename)
        if os.path.exists(path_in_sub) or os.path.exists(path_in_root):
            logger.info(f"  OK: {filename}")
        else:
            logger.error(f"  MISSING: {filename}")
            missing.append(filename)

    if missing:
        logger.error(f"Verification failed: missing files: {missing}")
        sys.exit(1)

    logger.info("\nCIFAR-100 dataset ready.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract the CIFAR-100 dataset."
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=CIFAR100_DATA_DIR,
        help=f"Target directory to extract files (default: {CIFAR100_DATA_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if archive already exists.",
    )
    args = parser.parse_args()

    download_and_extract_cifar100(target_dir=args.target_dir, force=args.force)


if __name__ == "__main__":
    main()
