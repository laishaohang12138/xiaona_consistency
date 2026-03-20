# Anchor Registry

## Priority Rule
- Ref #1 always owns face identity
- Ref #2 always owns body proportion, body architecture, and full-body framing
- Ref #3 only joins when shoulders, neckline, clavicle, or upper-body boundary drifts
- Single person only
- No redesign
- No beautify
- Do not misuse anchors as loose style references

## Ref #1
- ID: `ref_1_face_master`
- Role: FACE MASTER
- File: `anchors/face/front/A-Core_01_0deg_MASTER.png`
- Absolute path: `D:\xiaona_consistency\anchors\face\front\A-Core_01_0deg_MASTER.png`
- Purpose:
  - Lock face identity
  - Lock five-feature layout
  - Lock stable face geometry across batches
- Conflict rule:
  - If face disagrees with any other source, Ref #1 wins

## Ref #2
- ID: `ref_2_full_body_master`
- Role: FULL BODY MASTER
- File: `anchors/full/Task-63987060-116-1.png`
- Absolute path: `D:\xiaona_consistency\anchors\full\Task-63987060-116-1.png`
- Purpose:
  - Lock body proportions
  - Lock body constitution
  - Lock full-body framing
  - Lock whole-body read and silhouette discipline
- Conflict rule:
  - If body proportion, waistline, skeleton, or framing disagrees with other sources, Ref #2 wins

## Ref #3
- ID: `ref_3_upper_support_94`
- Role: UPPER SUPPORT
- File: `anchors/upper/Task-63987060-94-1.png`
- Absolute path: `D:\xiaona_consistency\anchors\upper\Task-63987060-94-1.png`

- ID: `ref_3_upper_support_97`
- Role: UPPER SUPPORT
- File: `anchors/upper/Task-63987060-97-1.png`
- Absolute path: `D:\xiaona_consistency\anchors\upper\Task-63987060-97-1.png`

- Purpose:
  - Stabilize shoulder line
  - Stabilize clavicle read
  - Stabilize neckline and upper-body edge
- Conflict rule:
  - Ref #3 never overrides Ref #1 on face
  - Ref #3 never overrides Ref #2 on full-body proportion

## Side Supports
- ID: `ref_4_face_profile_side90_left`
- Role: FACE SUPPORT
- File: `anchors/face/profile_like/strict_side_90_left_support_v2.png`
- Purpose:
  - Support left-side `side_90` / `profile_like` face comparison
  - Do not override Ref #1 front identity lock

- ID: `ref_5_face_profile_side90_right`
- Role: FACE SUPPORT
- File: `anchors/face/profile_like/strict_side_90_right_support_v2.png`
- Purpose:
  - Support right-side strict `side_90` face comparison
  - Do not override Ref #1 front identity lock

## Perfect XiaoNa Reference
- Canonical full-body benchmark:
  - `Task-63987060-116-1`
- Use this anchor whenever the review question is:
  - "Is this still XiaoNa?"
  - "Is this the current body constitution of XiaoNa?"
