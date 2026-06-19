"""Dataset preparation: extract skeletons from camera fall datasets.

Runs pose estimation over UP-Fall / NTU (train pool) and URFD / Le2i (held-out
zero-shot test) and caches normalized COCO skeleton sequences for training/eval.

Usage:
    python data/prepare.py --dataset up-fall --src data/raw/up-fall --out data/up-fall
"""

from __future__ import annotations

import argparse

DATASETS = ("up-fall", "ntu", "urfd", "le2i", "sisfall", "fallalld")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare skeleton datasets")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--src", required=True, help="raw dataset root")
    p.add_argument("--out", required=True, help="output cache dir")
    p.add_argument("--pose-backend", default="movenet-thunder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: iterate clips -> pose estimate -> normalize_to_coco -> save .npz
    raise NotImplementedError


if __name__ == "__main__":
    main()
