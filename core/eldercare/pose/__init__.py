"""Pose runners and normalization to the shared 17-joint COCO schema.

Two backends:

* :class:`MediaPipePoseRunner` — BlazePose via Google MediaPipe (CPU, no GPU,
  no ONNX file). This is what the webcam MVP uses.
* :class:`MoveNetRunner` — MoveNet Lightning/Thunder via ONNX Runtime, for the
  on-device Pi build (Phase 3) once the .onnx weights exist.

Both return keypoints already remapped to COCO-17 order (see
:data:`..schema.COCO_KEYPOINTS`) with per-joint confidence retained.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import COCO_KEYPOINTS, Keypoint

# BlazePose (33 landmarks) -> COCO-17 index map. MediaPipe landmark ordering is
# documented and stable across the 0.10.x line.
_BLAZEPOSE_TO_COCO: tuple[int, ...] = (
    0,   # nose
    2,   # left_eye
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
)
assert len(_BLAZEPOSE_TO_COCO) == len(COCO_KEYPOINTS)


class PoseEstimator(ABC):
    """Base for pose backends. Frames go in, COCO keypoints come out."""

    @abstractmethod
    def infer(self, frame) -> list[Keypoint]:  # noqa: ANN001 (np.ndarray)
        """Run pose estimation on a single BGR/RGB frame -> COCO keypoints."""
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources (override if needed)."""


DEFAULT_POSE_MODEL = "models/pose_landmarker_lite.task"


class MediaPipePoseRunner(PoseEstimator):
    """BlazePose via the MediaPipe Tasks API — CPU-only, no GPU.

    Needs the ``pose_landmarker*.task`` model bundle (download once; see README).
    Returns normalized [0, 1] image coordinates (y grows downward), matching the
    project convention. ``visibility`` becomes the per-joint confidence.

    Uses VIDEO running mode, so feed monotonically increasing timestamps via
    :meth:`infer` (handled automatically from a frame counter).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_POSE_MODEL,
        min_confidence: float = 0.5,
    ) -> None:
        import os

        import mediapipe as mp  # imported lazily so core stays dependency-light
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"pose model not found: {model_path}. Download it with:\n"
                "  python -c \"import urllib.request as u; u.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', "
                f"'{model_path}')\""
            )

        self._mp = mp
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_idx = 0

    def infer(self, frame, timestamp_ms: int | None = None) -> list[Keypoint]:  # noqa: ANN001
        import cv2

        # MediaPipe Tasks wants an mp.Image in RGB; webcam frames are BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        if timestamp_ms is None:
            # ~30 fps synthetic clock; only the monotonic increase matters.
            timestamp_ms = self._frame_idx * 33
        self._frame_idx += 1

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            # No person detected this frame -> all-zero-confidence skeleton.
            return [Keypoint(0.0, 0.0, 0.0) for _ in COCO_KEYPOINTS]

        return normalize_to_coco(result.pose_landmarks[0], "blazepose")

    def close(self) -> None:
        self._landmarker.close()


class MoveNetRunner(PoseEstimator):
    """MoveNet Lightning (fast) / Thunder (accurate) via ONNX Runtime."""

    def __init__(self, onnx_path: str, variant: str = "lightning") -> None:
        # TODO: load ort.InferenceSession (CPU/XNNPACK on Pi, CUDA on dev)
        raise NotImplementedError("MoveNet ONNX path is Phase 3 — use MediaPipe")

    def infer(self, frame) -> list[Keypoint]:  # noqa: ANN001
        # TODO: letterbox -> run -> map to COCO order, retain per-joint score
        raise NotImplementedError


def normalize_to_coco(raw_keypoints, source_schema: str) -> list[Keypoint]:
    """Map a backend's native joints onto the shared 17-joint COCO order.

    ``raw_keypoints`` for ``"blazepose"`` is the MediaPipe landmark list, whose
    entries expose ``.x``, ``.y`` (normalized) and ``.visibility``.
    """
    if source_schema == "blazepose":
        out: list[Keypoint] = []
        for src_idx in _BLAZEPOSE_TO_COCO:
            lm = raw_keypoints[src_idx]
            score = float(getattr(lm, "visibility", 1.0))
            out.append(Keypoint(x=float(lm.x), y=float(lm.y), score=score))
        return out
    raise ValueError(f"unknown source schema: {source_schema!r}")
