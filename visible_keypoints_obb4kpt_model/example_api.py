#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from detector_api import BottleOBB4KPTDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bottle OBB and keypoint detector.")
    parser.add_argument("image", help="Path to one input image")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, 0, 1, ...")
    args = parser.parse_args()

    detector = BottleOBB4KPTDetector(device=args.device)
    result = detector.predict_one(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
