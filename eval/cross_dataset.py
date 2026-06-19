"""Headline eval: cross-dataset zero-shot fall detection.

Train on UP-Fall + NTU; test zero-shot on URFD and Le2i (no fine-tuning).
Reports sensitivity/specificity/F1 and the generalization gap vs in-dataset.

Usage:
    python eval/cross_dataset.py --model models/ctrgcn.onnx --test urfd le2i
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-dataset zero-shot eval")
    p.add_argument("--model", required=True, help="ONNX classifier")
    p.add_argument("--test", nargs="+", default=["urfd", "le2i"])
    p.add_argument("--config", default="configs/ctrgcn.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: run pipeline over held-out sets; compute metrics; write tables
    raise NotImplementedError


if __name__ == "__main__":
    main()
