# Minimal Patch Playbook

## Patch Priority
1. Keep the main line stable
2. Prefer the smallest reversible patch
3. Fix structure before adding volume
4. Fix identity before aesthetics
5. Fix body architecture before face micro-polish

## Allowed Patch Types
- Shot plugin micro-adjustment
- Anchor routing adjustment
- QA threshold or reason-code tuning
- Layer quota rebalance
- Add targeted补图 batch
- Add new provider behind a stable interface

## When to Patch Shot Plugin
- Pose drift
- Arm spacing drift
- Foot placement instability
- Framing micro-shift

## When to Patch Anchor Strategy
- Face drift across tools
- Upper-body drift while full-body still matches
- Shoulder / clavicle / neckline instability

## When to Patch QA
- Repeated false warnings caused by legacy silhouette logic
- Repeated skin-color false positives from clothing contamination
- Repeated foot misses caused by incomplete landmark usage

## When to Patch Training Set
- One layer teaches too many things at once
- One layer starts producing a second XiaoNa
- One layer accumulates the same structural defect

## Hard Stop Conditions
- Two XiaoNa faces appear in one batch family
- 3/4 images become less like XiaoNa than front images
- Dressed images change person identity more than BODY GOLD images
- High-neck and shirt-collar images swallow clavicles or neck
- Outerwear becomes stable while the person becomes unstable

