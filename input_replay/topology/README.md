# Topology Replay Pack

This folder is for review-only side/back topology validation.

Rules:
- Keep one pose/view variant inside one directory.
- Keep full body and feet visible whenever possible.
- Change view or mild gait/stance only; do not intentionally change identity, body structure, outfit class, or lighting.
- Read `Task-63987060-116-1.png` through pose/gait-aware metrics, not as a rigid overlay.
- Do not mix these replay images into front or three-quarter clean lanes.

Variants:
- `side/side_left_profile`: Collect lane-pure left profile full-body frames with feet visible and stable torso depth.
- `side/side_right_profile`: Collect lane-pure right profile full-body frames with feet visible and stable neck-to-torso connection.
- `back/back180_neutral`: Collect straight back-view full-body frames with readable shoulder, waist, hip, and leg contour.
- `back/back180_subtle_gait_shift`: Collect mild back-view gait or stance shifts while keeping body structure and framing stable.

Operator doc: `docs/31_topology_replay_pack.md`
