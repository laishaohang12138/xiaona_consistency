# QA Reason Dictionary

## Purpose
- This file translates the current QA reason codes into project-level meaning and expected action
- It is intentionally smaller than the runtime code surface
- Add new reasons here when they become part of stable review language

## Core Face Reasons
- `FACE_PASS`
  - Face module passed
- `FACE_WARN`
  - Face module needs human review
- `FACE_FAIL`
  - Face module failed
- `FACE_NOT_FOUND`
  - Face signal missing; identity arbitration unreliable
- `FACE_TOO_SMALL`
  - Face pixels insufficient; identity confidence reduced
- `FACE_UNDEREXPOSED_DARK`
  - Face brightness too low for safe identity judgment
- `FACE_NO_RELIABLE_SIGNAL`
  - Face confidence too low for strict automated decision

## Core Pose / Framing Reasons
- `UPPER_PASS`
  - Upper-body geometry stable enough
- `FULL_PASS`
  - Full-body module stable enough
- `FRAMING_OK`
  - Full-body framing acceptable
- `FEET_IN_FRAME`
  - Feet detected inside the frame
- `FEET_CROPPED_OR_TOO_HIGH`
  - Framing does not safely teach feet
- `ANKLES_NOT_VISIBLE`
  - Lower-body endpoint signal unreliable

## Constitution Reasons
- `BODY_CONSTITUTION_READY`
  - Constitution module produced a usable result
- `BODY_CONSTITUTION_WARN`
  - Constitution is suspicious at a moderate level
- `BODY_CONSTITUTION_STRONG_WARN`
  - Constitution is strongly suspicious
- `BODY_CONSTITUTION_LOW_CONF_SKIP`
  - Constitution result exists but is too unreliable to gate

## Skin Reasons
- `SKIN_UNIFORMITY_READY`
  - Skin-consistency result usable
- `SKIN_UNIFORMITY_WARN`
  - Moderate skin inconsistency
- `SKIN_UNIFORMITY_STRONG_WARN`
  - Strong skin inconsistency
- `SKIN_UNIFORMITY_LOW_CONF_SKIP`
  - Skin result too unreliable to gate

## Depth Reasons
- `DEPTH_3D_LITE_READY`
  - Depth-lite result usable
- `DEPTH_3D_LITE_WARN`
  - Spatial thickness suspicious
- `DEPTH_3D_LITE_STRONG_WARN`
  - Strong fake-turn or thin-body suspicion
- `DEPTH_3D_LITE_LOW_CONF_SKIP`
  - Depth-lite result too unreliable to gate

## Usage Rule
- Stable reason codes can appear in Custom GPT knowledge
- Dynamic numeric outputs stay in batch artifacts, not in knowledge

