// Shared keypoint schema + wire decoding — the TS side of the core/ port.
// Mirrors core/eldercare/schema.py (COCO-17, flat [x,y,score] wire triplets).

export const COCO_KEYPOINTS = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
] as const;

const idx = Object.fromEntries(COCO_KEYPOINTS.map((n, i) => [n, i]));

// Bone connections for drawing the skeleton overlay.
export const COCO_EDGES: [number, number][] = (
  [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
    ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
    ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
    ["nose", "left_eye"], ["nose", "right_eye"],
    ["left_eye", "left_ear"], ["right_eye", "right_ear"],
  ] as [string, string][]
).map(([a, b]) => [idx[a], idx[b]] as [number, number]);

export interface Keypoint {
  x: number;
  y: number;
  score: number;
}

export interface PoseFrame {
  node_id: string;
  ts: number;
  keypoints: Keypoint[];
  fall_score: number;
  event: "none" | "fall" | "node_offline";
}

// Inverse of PoseFrame.to_wire() in schema.py.
export function fromWire(msg: any): PoseFrame {
  const flat: number[] = msg.keypoints ?? [];
  const keypoints: Keypoint[] = [];
  for (let i = 0; i < flat.length; i += 3) {
    keypoints.push({ x: flat[i], y: flat[i + 1], score: flat[i + 2] });
  }
  return {
    node_id: msg.node_id,
    ts: msg.ts,
    keypoints,
    fall_score: msg.fall_score ?? 0,
    event: msg.event ?? "none",
  };
}
