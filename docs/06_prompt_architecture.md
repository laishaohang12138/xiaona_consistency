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
- Scope is BRIDGE only; OUTER / FACE LOCK still need their own scoped assets

## BRIDGE Runtime Formulas
- Bridge first pass:
  `bridge_runtime_base_v0_3_1_nb2.txt` + one main plugin `BR-01~BR-08` + `negative_core_bridge_v0_3_1_nb2.txt`
- Bridge shortlist re-roll:
  `bridge_runtime_base_v0_3_1_nb2.txt` + one main plugin `BR-01~BR-08` + `negative_core_bridge_v0_3_1_nb2.txt` + `negative_finish_bridge_v0_3_1_nb2.txt`

## Active NECKLINE Prompt Stack
- The active NECKLINE prompt pack is `v0.1.1-nb-clean`
- It is a scoped runtime stack aligned to the NECKLINE layer role
- Scope is NECKLINE only; OUTER / FACE LOCK still need their own scoped assets

## NECKLINE Runtime Formulas
- Neckline first pass:
  `neckline_runtime_base_v0_1_1_nb_clean.txt` + one main plugin `NK-A01~NK-E04` + `negative_core_neckline_v0_1_1_nb_clean.txt`
- Neckline shortlist re-roll:
  `neckline_runtime_base_v0_1_1_nb_clean.txt` + one main plugin `NK-A01~NK-E04` + `negative_core_neckline_v0_1_1_nb_clean.txt` + `negative_finish_neckline_v0_1_1_nb_clean.txt`

## Landed FACE_LOCK Prompt Asset
- The landed FACE_LOCK prompt pack is `v0.1.1-nb2-topology-16x1-copyready`
- It lives under `prompts/face_lock/v0_1_1_nb2_topology_16x1_copyready/`
- It is currently `landed_review_only`, not an active runtime formula
- Scope is FACE_LOCK only; it reinforces face identity geometry, 3D face topology, and head-neck-body attachment
- It does not create a new face truth, body truth, winner-bank truth, or training-admission signal

## FACE_LOCK Copy-Ready Formula
- Copy-ready shortlist:
  one assembled prompt `FL-01~FL-16`, with positive prompt, shot plugin, negative core, and negative finish embedded in the same file
- The pack is intentionally not split into base/plugin/negative files yet
- Runtime formula extraction requires a later governance pass

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

## NECKLINE Runtime Base
- Owns:
  - reference priority doctrine
  - anchor role split
  - identity geometry hard-lock
  - upper-boundary continuity
  - neckline exposure discipline
  - fabric-skin boundary stability
  - camera / optics / framing discipline
  - low-entropy anatomy and motion rules
- Cross-batch reusable
- Must remain stable across five neckline families and must not be casually edited for a single image

## NECKLINE Plugin Families
- Family A `NK-A01~NK-A06` owns HIGH / MOCK NECK coverage
- Family B `NK-B01~NK-B06` owns CREW / U NECK coverage
- Family C `NK-C01~NK-C06` owns V NECK coverage
- Family D `NK-D01~NK-D06` owns SHIRT COLLAR / OPEN COLLAR coverage
- Family E `NK-E01~NK-E04` owns OFF-SHOULDER / HALTER coverage
- NECKLINE plugins may control neckline openness, collar / placket readability, clavicle reveal, and upper-boundary framing, but must not overturn the runtime base
- NECKLINE does not own outerwear volume, heavy layering, or FACE LOCK identity recovery

## NECKLINE Negative Layers
- `negative_core_neckline_v0_1_1_nb_clean.txt` is always on for active NECKLINE lanes
- `negative_finish_neckline_v0_1_1_nb_clean.txt` is shortlist re-roll tightening only

## FACE_LOCK Shot Families
- `FL-01~FL-02` own canonical front baselines
- `FL-03~FL-06` own 10-22 degree yaw identity carry
- `FL-07~FL-08` own micro pitch stability
- `FL-09~FL-10` own head-neck-body attachment hold
- `FL-11~FL-12` own light microrelief stability
- `FL-13~FL-14` own clavicle, hairline, and ear-root readability
- `FL-15~FL-16` own 28-30 degree contour threshold

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
