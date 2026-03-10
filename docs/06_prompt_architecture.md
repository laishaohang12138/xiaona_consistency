# Prompt Architecture

## Global Rule
- All generation prompts in this project use:
  - Universal Base
  - Shot Plugin
  - Universal Negative

## Formula
- Final prompt = Universal Base + Shot Plugin + Universal Negative

## Universal Base
- Owns:
  - Identity
  - Body constitution
  - Scene
  - lighting
  - clothing discipline
  - camera discipline
  - material discipline
  - anatomy safety boundaries
- Cross-batch reusable
- Must not be lightly edited because of a single image

## Shot Plugin
- Owns only one image or one batch's local control
- Allowed scope:
  - Direction
  - Pose
  - Weight shift
  - Arm spacing
  - Foot placement
  - Local framing
- Rule:
  - Minimal change only
  - Cannot overthrow the base

## Universal Negative
- Global failure interception layer
- Owns:
  - Identity drift
  - Influencer face
  - Hip pop
  - Model-pose smell
  - Foot errors
  - Knee shadow
  - Background pollution
  - Lens distortion
  - Erotic / ad-like drift

## Scope Rule
- The current imported master prompt belongs to BODY GOLD only
- It is not the whole-project master prompt
- Other layers must use their own layer-scoped plugin or master assets

## Versioning Rule
- Every prompt asset needs:
  - `scope`
  - `version`
  - `status`
  - `refs_required`
  - `refs_optional`
  - `source`

