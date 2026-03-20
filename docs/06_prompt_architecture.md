# Prompt Architecture

## Active BODY GOLD Prompt Stack
- The active BODY GOLD prompt pack is `v1.1F-compat`
- It is a flow runtime stack, not a single fixed prompt
- Scope is BODY GOLD only; BRIDGE now has its own scoped prompt asset pack

## BODY GOLD Runtime Formulas
- Flow first pass:
  `flow_runtime_base_v1_1F_compat.txt` + one main plugin `BG-01~BG-12` + `negative_core_v1_1F.txt`
- Flow shortlist re-roll:
  `flow_runtime_base_v1_1F_compat.txt` + one main plugin `BG-01~BG-12` + `negative_core_v1_1F.txt` + `negative_finish_v1_1F.txt`
- Shadow observation:
  `flow_runtime_base_v1_1F_compat.txt` + one shadow plugin `BG-13~BG-16` + `negative_core_v1_1F.txt` + `negative_shadow_v1_1F.txt`
- ALT experimental lane:
  `flow_runtime_base_v1_1F_compat.txt` + one ALT plugin `BG-05A / BG-06A / BG-09A` + `negative_core_v1_1F.txt`

## Active BRIDGE Prompt Stack
- The active BRIDGE prompt pack is `v0.3.1-nb2`
- It is a scoped runtime stack aligned to the BRIDGE layer role
- Scope is BRIDGE only; NECKLINE / OUTER / FACE LOCK still need their own scoped assets

## BRIDGE Runtime Formulas
- Bridge first pass:
  `bridge_runtime_base_v0_3_1_nb2.txt` + one main plugin `BR-01~BR-08` + `negative_core_bridge_v0_3_1_nb2.txt`
- Bridge shortlist re-roll:
  `bridge_runtime_base_v0_3_1_nb2.txt` + one main plugin `BR-01~BR-08` + `negative_core_bridge_v0_3_1_nb2.txt` + `negative_finish_bridge_v0_3_1_nb2.txt`

## BRIDGE Runtime Base
- Owns:
  - reference priority doctrine
  - anchor role split
  - identity geometry hard-lock
  - body architecture carryover
  - global skin read
  - clothing transition discipline
  - camera / optics / framing discipline
  - low-entropy anatomy and motion rules
- Cross-batch reusable
- Must remain stable across BRIDGE slots and must not be casually edited for a single image

## BRIDGE Plugin Families
- Main plugins `BR-01~BR-08` own sleeve coverage variants plus front / mild three-quarter verification
- `BR-01~BR-04` remain front transition baselines
- `BR-05~BR-08` remain mild three-quarter verification slots
- BRIDGE plugins must not overturn the runtime base and must not expand into `side_90`, `back_180`, NECKLINE, or OUTER ownership

## BRIDGE Negative Layers
- `negative_core_bridge_v0_3_1_nb2.txt` is always on for active BRIDGE lanes
- `negative_finish_bridge_v0_3_1_nb2.txt` is shortlist re-roll tightening only

## BODY GOLD Runtime Base
- Owns:
  - reference priority doctrine
  - anchor role split
  - identity geometry hard-lock
  - body architecture
  - global skin read
  - lighting
  - wardrobe discipline
  - camera / optics / framing discipline
  - low-entropy anatomy and motion rules
- Cross-batch reusable
- Must remain stable across plugins and must not be casually edited for a single image

## Shared Anchor Role Split
- `Ref #1` owns facial identity geometry only
- `Ref #2` owns body architecture, framing, and global skin read
- `Ref #3` is upper-boundary support only
- If `Ref #1` face skin is paler than the body read, preserve `Ref #1` geometry and harmonize skin family toward `Ref #2`

## BODY GOLD Plugin Families
- Main plugins `BG-01~BG-12` own front and mild three-quarter calibration variants
- Shadow plugins `BG-13~BG-16` own side/back observation lanes
- ALT plugins `BG-05A`, `BG-06A`, `BG-09A` are supplementary lanes and must not replace old BG numbering semantics
- Plugins may control pose, weight distribution, limb readability, and local framing, but must not overturn the runtime base

## BODY GOLD Negative Layers
- `negative_core_v1_1F.txt` is always on for active runtime lanes
- `negative_finish_v1_1F.txt` is only for shortlist re-roll tightening
- `negative_shadow_v1_1F.txt` is only for side/back shadow lanes
- `negative_current_reference.txt` is preserved as a reference comparison, not an active runtime layer

## Numbering Protection
- Old BG numbering must not be rebound to a different pose family
- `BG-07` and `BG-08` remain three-quarter subtle weight-bias lanes
- `BG-10` remains a front static staggered lane and must stay non-dynamic

## Versioning Rule
- Every prompt asset manifest should record:
  - `scope`
  - `version`
  - `status`
  - `refs_required`
  - `refs_optional`
  - `source_import`
  - runtime presets
  - plugin registry
