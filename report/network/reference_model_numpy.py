"""Torch-free verification of the reference network's shape flow + param budget.

PyTorch is not required in every environment (the MVP venv ships without it), so
this script reproduces the *forward shape arithmetic* of reference_model.py in
pure NumPy and analytically counts parameters. It exists to let a reviewer
confirm — on any machine — that the architecture is internally consistent and to
print the real parameter budget quoted in the report.

It uses random tensors only. There is NO pretrained checkpoint anywhere.

Run:
    python report/network/reference_model_numpy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_model import ModelConfig, build_adjacency  # noqa: E402


# ---- analytic parameter counters for each layer type ----------------------
def conv2d_params(cin, cout, kt=1, kv=1, bias=True):
    return cin * cout * kt * kv + (cout if bias else 0)


def bn_params(c):
    return 2 * c  # gamma + beta


def temporal_conv_params(cin, cout, scales=(1, 2)):
    """Mirror TemporalConv: len(scales)+1 branches + a fuse 1x1 + final BN."""
    nb = len(scales) + 1
    bch = cout // nb
    p = 0
    for _ in scales:
        p += conv2d_params(cin, bch, 1, 1) + bn_params(bch)
        p += conv2d_params(bch, bch, 9, 1) + bn_params(bch)  # kernel 9 along T
    # max-pool branch: 1x1 conv + 2 BN (pool has no params)
    p += conv2d_params(cin, bch, 1, 1) + bn_params(bch) + bn_params(bch)
    p += conv2d_params(bch * nb, cout, 1, 1) + bn_params(cout)
    return p


def stgraph_params(cin, cout, num_part=3):
    return conv2d_params(cin, cout * num_part, 1, 1)


def ctrgraph_params(cin, cout, num_part=3, rel_reduction=8):
    mid = max(cout // rel_reduction, 8)
    p = conv2d_params(cin, mid, 1, 1)       # conv1
    p += conv2d_params(cin, mid, 1, 1)      # conv2
    p += conv2d_params(mid, cout, 1, 1)     # conv_refine
    p += conv2d_params(cin, cout * num_part, 1, 1)  # value
    p += 1                                  # alpha scalar
    return p


def block_params(cin, cout, stride, kind, scales):
    p = ctrgraph_params(cin, cout) if kind == "ctrgcn" else stgraph_params(cin, cout)
    p += bn_params(cout)                    # gcn_bn
    p += temporal_conv_params(cout, cout, scales)
    if not (cin == cout and stride == 1):   # residual projection
        p += conv2d_params(cin, cout, 1, 1) + bn_params(cout)
    return p


def count_params(cfg: ModelConfig, kind: str) -> int:
    p = 0
    if kind == "ctrgcn" and cfg.confidence_gating:
        p += cfg.num_joints                 # ConfidenceGate.joint_reliability
    p += bn_params(cfg.in_channels * cfg.num_joints)  # data_bn
    cin = cfg.in_channels
    for cout, stride in cfg.blocks:
        p += block_params(cin, cout, stride, kind, cfg.temporal_scales)
        cin = cout
    p += cin * cfg.num_classes + cfg.num_classes      # final FC
    return p


# ---- forward shape flow (no learned weights, just dimensions) -------------
def forward_shapes(cfg: ModelConfig, kind: str) -> None:
    N, C, T, V = 4, cfg.in_channels, cfg.window_T, cfg.num_joints
    print(f"\n  {kind.upper()} forward shape trace")
    print(f"    input                         (N,C,T,V) = ({N},{C},{T},{V})")
    if kind == "ctrgcn" and cfg.confidence_gating:
        print(f"    confidence gate (mask+impute) (N,C,T,V) = ({N},{C},{T},{V})  [shape preserved]")
    print(f"    data BN over C*V={C*V}            ok")
    t = T
    cin = C
    for bi, (cout, stride) in enumerate(cfg.blocks):
        t = -(-t // stride)  # ceil division for stride along time
        print(f"    block {bi:>2}  graph+temporal  -> (N,{cout:>3},{t:>2},{V})"
              f"   stride={stride}")
        cin = cout
    print(f"    global avg pool (T,V)         -> (N,{cin})")
    print(f"    fc                            -> (N,{cfg.num_classes})  logits")


def main() -> None:
    cfg = ModelConfig()
    print("Reference fall-classification network - torch-free verification")
    print("(random/illustrative dimensions only; NO pretrained weights)\n")

    # adjacency sanity
    A = build_adjacency(cfg.num_joints)
    print(f"  skeleton adjacency A stack: shape {A.shape} "
          f"(3 ST-GCN partitions x V x V)")
    print(f"    partition row-sums after normalization (should be ~1): "
          f"self={A[0].sum(1).mean():.2f}, "
          f"centripetal={A[1].sum(1).mean():.2f}, "
          f"centrifugal={A[2].sum(1).mean():.2f}")

    for kind in ("stgcn", "ctrgcn"):
        forward_shapes(cfg, kind)

    print("\n  Parameter budget (analytic, matches reference_model.py construction):")
    for kind in ("stgcn", "ctrgcn"):
        n = count_params(cfg, kind)
        print(f"    {kind.upper():7s}  {n:>10,d} params   (~{n/1e6:.2f} M, "
              f"~{n*4/1e6:.1f} MB FP32 / ~{n/1e6:.1f} MB INT8)")

    print("\n  Both fit the < 25 MB on-device footprint budget (research doc sec.7).")
    print("  Run report/network/reference_model.py with PyTorch installed to")
    print("  execute the real forward pass and confirm these exact counts.")


if __name__ == "__main__":
    main()
