"""Export a trained PyTorch classifier to ONNX.

Produces the single weights file served identically on the edge node and the
backend. Includes a PyTorch-vs-ONNX parity check (logits within 1e-3).

Usage:
    python train/export_onnx.py --ckpt runs/ctrgcn/best.pt --out models/ctrgcn.onnx
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export classifier to ONNX")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="models/ctrgcn.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--int8", action="store_true", help="post-training INT8 quant")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: load ckpt, torch.onnx.export with dynamic axes
    # TODO: run onnxruntime, assert max|logit_torch - logit_ort| < 1e-3
    raise NotImplementedError


if __name__ == "__main__":
    main()
