# Lighting Replay Pack

This folder is for controlled lighting replay only.

Rules:
- Keep the same lane inside one lane folder.
- Do not mix lighting validation images into `input_split/`.
- Change lighting only. Do not intentionally change identity, body structure, outfit class, or framing.
- When more images are added, rerun `prepare_lighting_replay_pack` to refresh manifests and metadata templates.

Variants:
- `neutral_base`: neutral base - Use normal even lighting as the control baseline under the same lane and outfit class.
- `bright_exposure`: bright exposure - Make the frame brighter without clipping face or leg detail.
- `dim_exposure`: dim exposure - Make the frame dimmer while keeping facial and leg structure visible.
- `warm_cast`: warm cast - Allow a mild warm shift without damaging identity or garment boundaries.
- `cool_cast`: cool cast - Allow a mild cool shift without making the person look different.

Operator doc: `docs/27_lighting_replay_pack.md`
