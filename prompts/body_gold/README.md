# BODY GOLD Prompt Asset

## Active Version
- `v1.1F-compat` is the active BODY GOLD flow runtime prompt pack
- It replaces the earlier single-plugin `v1` asset as the active prompt documentation set
- Scope remains BODY GOLD only; this is not the whole-project master prompt

## Runtime Formula
- Flow first pass:
  `flow_runtime_base_v1_1F_compat.txt` + one main plugin `BG-01~BG-12` + `negative_core_v1_1F.txt`
- Flow shortlist re-roll:
  `flow_runtime_base_v1_1F_compat.txt` + one main plugin `BG-01~BG-12` + `negative_core_v1_1F.txt` + `negative_finish_v1_1F.txt`
- Shadow observation:
  `flow_runtime_base_v1_1F_compat.txt` + one shadow plugin `BG-13~BG-16` + `negative_core_v1_1F.txt` + `negative_shadow_v1_1F.txt`
- ALT lane:
  `flow_runtime_base_v1_1F_compat.txt` + one ALT plugin `BG-05A / BG-06A / BG-09A` + `negative_core_v1_1F.txt`

## Anchor Doctrine
- `Ref #1` = face master for identity geometry only
- `Ref #2` = full-body master for body architecture, framing, and global skin read
- `Ref #3` = upper support only when shoulder / neckline / clavicle drift appears
- If `Ref #1` face skin is paler than the body read, preserve facial geometry from `Ref #1` but harmonize the skin family to `Ref #2`

## File Groups
- Base:
  `flow_runtime_base_v1_1F_compat.txt`
- Main plugins:
  `shot_plugin_bg01_*` through `shot_plugin_bg12_*`
- Shadow plugins:
  `shot_plugin_bg13_*` through `shot_plugin_bg16_*`
- ALT plugins:
  `shot_plugin_bg05a_*`, `shot_plugin_bg06a_*`, `shot_plugin_bg09a_*`
- Negatives:
  `negative_core_v1_1F.txt`, `negative_finish_v1_1F.txt`, `negative_shadow_v1_1F.txt`
- Reference-only carryover:
  `negative_current_reference.txt`

## Numbering Rules
- Old BG numbering must not be rebound to a different pose family
- `BG-07` and `BG-08` remain three-quarter subtle weight-bias plugins
- `BG-10` remains a front static staggered family, rewritten to stay static and non-dynamic
- `BG-05A`, `BG-06A`, and `BG-09A` are supplementary ALT plugins and must not replace the main numbering semantics
