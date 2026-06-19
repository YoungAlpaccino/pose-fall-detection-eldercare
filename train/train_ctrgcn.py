"""Train the temporal fall classifier (ST-GCN / CTR-GCN / PoseConv3D).

Trains on UP-Fall + an NTU fall/ADL subset in PyTorch. ST-GCN serves as the deep
baseline; CTR-GCN adds channel-wise topology refinement with a confidence-gated
input layer.

Usage:
    python train/train_ctrgcn.py --config configs/ctrgcn.yaml
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train skeleton fall classifier")
    p.add_argument("--config", required=True, help="path to YAML hyperparams")
    p.add_argument("--model", default="ctrgcn", choices=["stgcn", "ctrgcn", "poseconv3d"])
    p.add_argument("--data-root", default="data/", help="processed skeleton data")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: build dataset/dataloader (subject-wise split, no leakage)
    # TODO: build model (confidence-gated input layer)
    # TODO: train loop + checkpointing + experiment tracking
    raise NotImplementedError


if __name__ == "__main__":
    main()
