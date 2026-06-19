"""Privacy eval: reconstruction attack from keypoints.

Trains a decoder keypoints->image and reports SSIM/LPIPS/identity-reID accuracy
to show that the telemetry is non-recoverable (target SSIM < 0.15, LPIPS > 0.6).

Usage:
    python eval/reconstruction_attack.py --keypoints data/urfd --frames data/raw/urfd
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Keypoint->image reconstruction attack")
    p.add_argument("--keypoints", required=True)
    p.add_argument("--frames", required=True, help="held-out ground-truth frames")
    p.add_argument("--epochs", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: train decoder; report SSIM/LPIPS/reID against held-out frames
    raise NotImplementedError


if __name__ == "__main__":
    main()
