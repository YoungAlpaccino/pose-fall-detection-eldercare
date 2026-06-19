import { useEffect, useRef } from "react";
import { COCO_EDGES, type PoseFrame } from "./lib/skeleton";

const MIN_SCORE = 0.2;

/** Live 2D skeleton overlay. Coordinates are normalized [0,1]; we scale to the
 *  canvas. A fallen pose tints the skeleton red. No pixels are ever drawn —
 *  there is no video to draw, by design (docs/PRIVACY.md). */
export function SkeletonCanvas({
  frame,
  alarming,
}: {
  frame: PoseFrame | null;
  alarming: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // subtle floor grid for spatial reference
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 10; g++) {
      const p = (g / 10) * W;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, H); ctx.stroke();
      const q = (g / 10) * H;
      ctx.beginPath(); ctx.moveTo(0, q); ctx.lineTo(W, q); ctx.stroke();
    }

    if (!frame) return;
    const kp = frame.keypoints;
    const color = alarming ? "#ff4d4f" : "#36d399";

    // bones
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    for (const [a, b] of COCO_EDGES) {
      const ka = kp[a], kb = kp[b];
      if (!ka || !kb || ka.score < MIN_SCORE || kb.score < MIN_SCORE) continue;
      ctx.beginPath();
      ctx.moveTo(ka.x * W, ka.y * H);
      ctx.lineTo(kb.x * W, kb.y * H);
      ctx.stroke();
    }

    // joints
    ctx.fillStyle = color;
    for (const k of kp) {
      if (k.score < MIN_SCORE) continue;
      ctx.beginPath();
      ctx.arc(k.x * W, k.y * H, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [frame, alarming]);

  return (
    <canvas
      ref={ref}
      width={480}
      height={480}
      className={`skeleton-canvas${alarming ? " alarming" : ""}`}
    />
  );
}
