#!/usr/bin/env python3
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from train.train_local_parking import main  # noqa: E402


if __name__ == "__main__":
    main()
