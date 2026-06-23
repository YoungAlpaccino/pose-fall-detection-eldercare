"""Reference implementation of the fall-classification network — FROM SCRATCH.

This is the deep "training network" the report documents. It is written from
first principles: the COCO-17 skeleton graph, the ST-GCN spatial-temporal
convolution, the CTR-GCN channel-wise topology refinement, and the
confidence-gated input layer are all defined here explicitly. **No pretrained
weights are loaded anywhere** — every parameter is freshly initialized. The file
is meant to be read by a reviewer as the canonical description of the model, and
run to verify that tensor shapes flow end-to-end.

Tensor convention (matches the skeleton-action-recognition literature):

    x : (N, C, T, V)
        N = batch         C = channels (3: x, y, score)
        T = frames        V = joints   (17, COCO order)

Two model families are provided behind one interface:

    * ``STGCN``   — the deep baseline (fixed graph, single adjacency partition
                    set, vanilla spatial GC + temporal conv).
    * ``CTRGCN``  — the proposed model (channel-wise topology refinement +
                    confidence-gated input + multi-scale temporal conv).

Both expose ``forward(x) -> logits (N, num_classes)`` and a matching
``num_parameters`` so the report can quote a real parameter count.

Requires PyTorch only to *train/run*; if torch is absent, importing this module
still succeeds for documentation purposes, and ``reference_model_numpy.py``
provides a dependency-free forward pass that verifies the same shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # torch optional — module still imports for doc/inspection
    _HAS_TORCH = False
    torch = None  # type: ignore
    nn = object  # type: ignore


# ---------------------------------------------------------------------------
# The skeleton graph — COCO-17, identical order to core/eldercare/schema.py.
# ---------------------------------------------------------------------------
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
_K = {n: i for i, n in enumerate(COCO_KEYPOINTS)}

# Bone list (undirected). This is the human skeleton topology the GCN respects.
COCO_BONES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
)


def build_adjacency(num_joints: int = 17):
    """Return the normalized adjacency stack A of shape (3, V, V).

    Uses the ST-GCN *spatial configuration* partitioning, which splits each
    node's neighbourhood into three subsets and gives the GCN three learnable
    weight matrices instead of one:

        k=0  self    — the joint itself (identity)
        k=1  centripetal — neighbours closer to the body centre of gravity
        k=2  centrifugal — neighbours farther from the centre

    Each partition is symmetrically normalized:  A_k <- D^-1/2 (A_k) D^-1/2.

    Implemented with NumPy so it is available with or without torch (the torch
    models wrap the result in a buffer).
    """
    import numpy as np

    V = num_joints
    # binary adjacency (no self loops yet)
    A = np.zeros((V, V), dtype=np.float32)
    for a, b in COCO_BONES:
        i, j = _K[a], _K[b]
        A[i, j] = A[j, i] = 1.0

    # centre of gravity = mean joint position on a canonical standing skeleton.
    # We approximate "distance to centre" by graph hop-distance to the hips,
    # which is the standard ST-GCN choice and needs no coordinates.
    centre = _K["left_hip"]
    hop = _hop_distance(A, centre)

    self_mat = np.eye(V, dtype=np.float32)
    centripetal = np.zeros((V, V), dtype=np.float32)
    centrifugal = np.zeros((V, V), dtype=np.float32)
    for i in range(V):
        for j in range(V):
            if A[i, j] <= 0:
                continue
            if hop[j] == hop[i]:
                # same hop level -> fold into self/identity partition's spirit;
                # keep it in centripetal to stay 3-partition (rare for a tree).
                centripetal[i, j] = 1.0
            elif hop[j] < hop[i]:
                centripetal[i, j] = 1.0  # neighbour closer to centre
            else:
                centrifugal[i, j] = 1.0  # neighbour farther from centre

    stack = np.stack([
        _normalize(self_mat),
        _normalize(centripetal),
        _normalize(centrifugal),
    ], axis=0)
    return stack  # (3, V, V) float32


def _hop_distance(A, source, max_hop: int = 8):
    """BFS hop distance from ``source`` to every node (inf if unreachable)."""
    import numpy as np

    V = A.shape[0]
    dist = np.full(V, np.inf)
    dist[source] = 0
    frontier = [source]
    for d in range(1, max_hop + 1):
        nxt = []
        for u in frontier:
            for v in range(V):
                if A[u, v] > 0 and dist[v] == np.inf:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return dist


def _normalize(mat):
    """Symmetric normalization D^-1/2 (A+I) D^-1/2 of a single partition."""
    import numpy as np

    A = mat + np.eye(mat.shape[0], dtype=np.float32) * (mat.diagonal().sum() == 0) * 0
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = deg[nz] ** -0.5
    Dn = np.diag(deg_inv_sqrt)
    return (Dn @ A @ Dn).astype(np.float32)


# ---------------------------------------------------------------------------
# Hyperparameters (mirrors configs/ctrgcn.yaml).
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    num_joints: int = 17
    num_classes: int = 2
    in_channels: int = 3            # x, y, score
    base_channels: int = 64
    window_T: int = 32
    confidence_gating: bool = True
    min_score: float = 0.2          # joints below this are masked + imputed
    temporal_scales: tuple = (1, 2)  # multi-scale temporal conv dilations
    # block channel plan: (out_channels, stride) per CTR-GC / ST-GC block
    blocks: tuple = field(default_factory=lambda: (
        (64, 1), (64, 1), (64, 1),
        (128, 2), (128, 1),
        (256, 2), (256, 1), (256, 1), (256, 1), (256, 1),
    ))


# ===========================================================================
# Everything below this point needs torch. Guarded so the module still imports.
# ===========================================================================
if _HAS_TORCH:

    class ConfidenceGate(nn.Module):
        """Mask low-confidence joints and temporally impute them from context.

        Input/Output: (N, C=3, T, V).  The third channel is the per-joint score.
        For joints whose score < ``min_score`` the (x, y) coordinates are
        replaced by a temporally forward/back-filled value computed from the
        confident frames of the same joint — so a brief occlusion never punches
        a hole in the tensor (see report Fig. 4). A learnable per-joint
        reliability scalar additionally re-weights each joint's contribution.
        """

        def __init__(self, num_joints: int, min_score: float = 0.2) -> None:
            super().__init__()
            self.min_score = min_score
            # learnable per-joint reliability (initialized to 1).
            self.joint_reliability = nn.Parameter(torch.ones(num_joints))

        def forward(self, x):  # (N, C, T, V)
            xy = x[:, :2]                      # (N, 2, T, V)
            score = x[:, 2:3]                  # (N, 1, T, V)
            mask = (score >= self.min_score).float()  # 1 where confident

            # temporal forward-fill then back-fill of xy using only confident frames.
            xy_filled = _temporal_impute(xy, mask)

            # learnable reliability gate, broadcast over (N, C, T).
            rel = torch.sigmoid(self.joint_reliability).view(1, 1, 1, -1)
            xy_gated = xy_filled * rel

            # re-attach a (masked) score channel so downstream BN sees 3 ch.
            return torch.cat([xy_gated, score * mask], dim=1)

    def _temporal_impute(xy, mask):
        """Forward-fill then back-fill (N,2,T,V) along T where mask==0.

        Vectorized: for each (joint, frame) carry the index of the most recent
        confident frame; gather xy from it. Mirrors the NumPy logic in
        core/eldercare/temporal/__init__.py (SlidingWindow.as_tensor).
        """
        N, C, T, V = xy.shape
        device = xy.device
        idx = torch.arange(T, device=device).view(1, T, 1).expand(N, T, V)
        conf = mask[:, 0] > 0                       # (N, T, V)
        # forward fill: cummax of confident indices
        fwd_idx = torch.where(conf, idx, torch.full_like(idx, -1))
        fwd_idx, _ = torch.cummax(fwd_idx, dim=1)
        # back fill: reverse cummax for leading gaps
        rev = torch.where(conf, idx, torch.full_like(idx, T))
        rev = torch.flip(torch.cummin(torch.flip(rev, [1]), dim=1)[0], [1])
        src = torch.where(fwd_idx >= 0, fwd_idx, rev).clamp(0, T - 1)  # (N,T,V)

        src_e = src.view(N, 1, T, V).expand(N, C, T, V)
        return torch.gather(xy, 2, src_e)

    class TemporalConv(nn.Module):
        """Multi-scale temporal convolution (a small subset of MS-TCN).

        Several parallel 1-D temporal convolutions with different dilations,
        concatenated — captures both the fast impact transient and the slower
        "stays down" evidence of a fall within one block.
        """

        def __init__(self, in_ch, out_ch, stride=1, scales=(1, 2)) -> None:
            super().__init__()
            branch_ch = out_ch // (len(scales) + 1)
            self.branches = nn.ModuleList()
            for d in scales:
                pad = ((9 - 1) // 2) * d  # kernel 9 along time, "same" padding
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, branch_ch, 1),
                    nn.BatchNorm2d(branch_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(branch_ch, branch_ch, (9, 1), (stride, 1),
                              (pad, 0), dilation=(d, 1)),
                    nn.BatchNorm2d(branch_ch),
                ))
            # a max-pool branch for robustness to outlier frames
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, 1),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((3, 1), (stride, 1), (1, 0)),
                nn.BatchNorm2d(branch_ch),
            ))
            # 1x1 to restore exactly out_ch (handles non-divisible channels)
            self.fuse = nn.Conv2d(branch_ch * (len(scales) + 1), out_ch, 1)
            self.bn = nn.BatchNorm2d(out_ch)

        def forward(self, x):
            out = torch.cat([b(x) for b in self.branches], dim=1)
            return self.bn(self.fuse(out))

    class STGraphConv(nn.Module):
        """Vanilla ST-GCN spatial graph convolution over a fixed 3-partition A.

        out = sum_k  (A_k  x)  W_k   — the static-topology baseline.
        """

        def __init__(self, in_ch, out_ch, A) -> None:
            super().__init__()
            self.register_buffer("A", A)        # (3, V, V)
            self.num_part = A.shape[0]
            self.theta = nn.Conv2d(in_ch, out_ch * self.num_part, 1)

        def forward(self, x):                   # (N, C, T, V)
            N, _, T, V = x.shape
            feat = self.theta(x)                # (N, out*K, T, V)
            feat = feat.view(N, self.num_part, -1, T, V)
            # einsum: sum over partitions k and source joints w
            out = torch.einsum("nkctv,kvw->nctw", feat, self.A)
            return out.contiguous()

    class CTRGraphConv(nn.Module):
        """CTR-GCN channel-wise topology refinement.

        Starts from the shared static topology A_k, then *refines it per
        channel* using a learned function of the difference between joint
        feature pairs:  A_refined = A_shared + alpha * MLP(x_i - x_j). This lets
        the graph adapt its connectivity to the motion content of each channel
        instead of being frozen to the human skeleton.
        """

        def __init__(self, in_ch, out_ch, A, rel_reduction=8) -> None:
            super().__init__()
            self.register_buffer("A", A)        # (3, V, V) shared topology
            self.num_part = A.shape[0]
            self.out_ch = out_ch
            mid = max(out_ch // rel_reduction, 8)
            # feature transform producing channel features for refinement
            self.conv1 = nn.Conv2d(in_ch, mid, 1)
            self.conv2 = nn.Conv2d(in_ch, mid, 1)
            # expand the pairwise relation back to out_ch channels
            self.conv_refine = nn.Conv2d(mid, out_ch, 1)
            # value transform, one per partition (like ST-GCN's W_k)
            self.value = nn.Conv2d(in_ch, out_ch * self.num_part, 1)
            self.alpha = nn.Parameter(torch.zeros(1))  # start at the static graph
            self.tanh = nn.Tanh()

        def forward(self, x):                   # (N, C, T, V)
            N, _, T, V = x.shape
            # --- channel-wise topology delta ---------------------------------
            # pool over time, build a (N, mid, V, V) pairwise difference tensor
            f1 = self.conv1(x).mean(2)          # (N, mid, V)
            f2 = self.conv2(x).mean(2)          # (N, mid, V)
            diff = f1.unsqueeze(-1) - f2.unsqueeze(-2)   # (N, mid, V, V)
            delta = self.conv_refine(self.tanh(diff))    # (N, out, V, V)

            # --- value, split into partitions --------------------------------
            val = self.value(x).view(N, self.num_part, self.out_ch, T, V)

            # refined adjacency per partition: A_k (broadcast) + alpha*delta
            A = self.A.view(1, self.num_part, 1, V, V)
            refined = A + self.alpha * delta.unsqueeze(1)  # (N,K,out,V,V)

            # graph conv with the per-channel refined topology
            out = torch.einsum("nkctv,nkcvw->nctw", val, refined)
            return out.contiguous()

    class GCNBlock(nn.Module):
        """One spatial-temporal block: graph conv -> temporal conv -> residual."""

        def __init__(self, in_ch, out_ch, A, stride=1, kind="ctrgcn",
                     scales=(1, 2)) -> None:
            super().__init__()
            if kind == "ctrgcn":
                self.gcn = CTRGraphConv(in_ch, out_ch, A)
            else:
                self.gcn = STGraphConv(in_ch, out_ch, A)
            self.gcn_bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.tcn = TemporalConv(out_ch, out_ch, stride=stride, scales=scales)

            if in_ch == out_ch and stride == 1:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, (stride, 1)),
                    nn.BatchNorm2d(out_ch),
                )

        def forward(self, x):
            res = self.residual(x)
            y = self.relu(self.gcn_bn(self.gcn(x)))
            y = self.tcn(y)
            return self.relu(y + res)

    class _Backbone(nn.Module):
        """Shared backbone for both STGCN and CTRGCN (kind switches the block)."""

        def __init__(self, cfg: ModelConfig, kind: str) -> None:
            super().__init__()
            self.cfg = cfg
            A = torch.from_numpy(build_adjacency(cfg.num_joints))
            self.kind = kind
            self.gate = ConfidenceGate(cfg.num_joints, cfg.min_score) \
                if cfg.confidence_gating else None
            self.data_bn = nn.BatchNorm1d(cfg.in_channels * cfg.num_joints)

            blocks = []
            in_ch = cfg.in_channels
            for out_ch, stride in cfg.blocks:
                blocks.append(GCNBlock(in_ch, out_ch, A, stride=stride,
                                       kind=kind, scales=cfg.temporal_scales))
                in_ch = out_ch
            self.blocks = nn.ModuleList(blocks)
            self.fc = nn.Linear(in_ch, cfg.num_classes)
            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):                   # (N, C, T, V)
            if self.gate is not None:
                x = self.gate(x)
            N, C, T, V = x.shape
            # data BN over the (C*V) feature dimension (standard for ST-GCN)
            x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
            x = self.data_bn(x)
            x = x.reshape(N, C, V, T).permute(0, 1, 3, 2).contiguous()

            for blk in self.blocks:
                x = blk(x)                      # (N, C', T', V)

            x = x.mean(dim=(2, 3))              # global average pool -> (N, C')
            return self.fc(x)                  # (N, num_classes)

        @property
        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

    class STGCN(_Backbone):
        """Deep baseline — static-topology spatial-temporal GCN."""

        def __init__(self, cfg: ModelConfig | None = None) -> None:
            super().__init__(cfg or ModelConfig(confidence_gating=False), "stgcn")

    class CTRGCN(_Backbone):
        """Proposed — confidence-gated, channel-wise topology-refined GCN."""

        def __init__(self, cfg: ModelConfig | None = None) -> None:
            super().__init__(cfg or ModelConfig(confidence_gating=True), "ctrgcn")

    def build_model(name: str = "ctrgcn", cfg: ModelConfig | None = None):
        name = name.lower()
        if name == "stgcn":
            return STGCN(cfg)
        if name == "ctrgcn":
            return CTRGCN(cfg)
        raise ValueError(f"unknown model {name!r} (use 'stgcn' or 'ctrgcn')")


def _demo() -> None:
    """Build the model on a synthetic batch and print shapes + parameter count.

    Uses random input only — there is NO pretrained checkpoint. This proves the
    architecture is internally consistent and reports a real parameter budget.
    """
    if not _HAS_TORCH:
        print("PyTorch is not installed in this environment.")
        print("Run report/network/reference_model_numpy.py for a torch-free "
              "forward-shape check, or `pip install torch` to exercise this file.")
        return

    cfg = ModelConfig()
    for name in ("stgcn", "ctrgcn"):
        model = build_model(name, cfg).eval()
        x = torch.randn(4, cfg.in_channels, cfg.window_T, cfg.num_joints)
        x[:, 2] = x[:, 2].sigmoid()  # make channel-3 a valid [0,1] score
        with torch.no_grad():
            logits = model(x)
        print(f"{name.upper():7s}  in {tuple(x.shape)} -> logits {tuple(logits.shape)}"
              f"   params = {model.num_parameters/1e6:.2f} M")


if __name__ == "__main__":
    _demo()
