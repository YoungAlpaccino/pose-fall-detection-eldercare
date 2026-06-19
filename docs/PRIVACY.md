# Privacy Invariant

> The skeleton is the privacy boundary, enforced by architecture rather than by policy.

## frame-egress = 0

The core invariant of this project: **no raw or reconstructable pixel data ever
leaves the edge node.** Raw frames live only inside the edge node process and are
never written to disk by default. What crosses the WebSocket is a compact record
of:

- 2D keypoints (x, y, confidence) per frame
- the classifier score
- the discrete event
- a timestamp

Even if the network, the cloud account, or a caregiver's phone is compromised,
there is no pixel data to leak.

## Enforcement & verification (TODO)

- [ ] Static audit: assert no frame buffers reach any WS/serialization path.
- [ ] Runtime audit: assert zero pixel bytes on the wire.
- [ ] Reconstruction attack: train a keypoints→image decoder and report
      SSIM < 0.15 / LPIPS > 0.6 / low reID accuracy to show non-recoverability.

## Ethics

Deployment requires informed consent from the resident (and guardian where
appropriate), per-room opt-in, the ability to pause monitoring, and a clear
data-handling notice. Because raw video never leaves the node, the dashboard
cannot "spy" — it can only show skeletons and alerts. This is both a privacy
feature and a dignity feature for the monitored person.
