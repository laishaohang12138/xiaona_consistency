# BODY GOLD Prompt Asset

## Scope
- This prompt asset is valid for BODY GOLD only
- It is not the whole-project master prompt

## Files
- `universal_base_v1.txt`
- `shot_plugin_bg01_front_neutral_symmetry_v1.txt`
- `universal_negative_v1.txt`
- `manifest.yaml`

## Compose Rule
- Final prompt = universal base + shot plugin + universal negative

## Use Rule
- Always place reference priority rules first
- Always keep Ref #1 for face identity
- Always keep Ref #2 for body architecture and framing
- Only add Ref #3 when shoulder or neckline drift appears

