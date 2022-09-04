#!/usr/bin/env python
"""Basic script to run experiments"""

import sys

sys.path.append("../")
from utils import run_basic_experiment  # noqa: E402


def main():
    """Main function to run."""
    run_basic_experiment()


if __name__ == "__main__":
    main()
