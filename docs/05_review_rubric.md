# Review Rubric

## Iron Rules
1. Every image must be checked against the perfect XiaoNa anchor
2. Every image must be checked against the frozen body constitution
3. QA score alone can never decide the result
4. Judge "is this XiaoNa" before "is this pretty"
5. Prefer underfilling over pollution
6. Every round should output 1 recommendation and 3 to 5 backups when possible

## Mandatory Review Questions
1. Is this XiaoNa?
2. Is this XiaoNa under the current body constitution?
3. Does this image teach one clear thing?
4. Does this image belong to the current layer?
5. Will this image pollute training?

## Pollution Alerts
- Identity drift
- Paper-thin legs
- Over-thin waist
- Toe errors
- Dirty knee shadow
- Outerwear changes the person
- Turtleneck eats clavicles
- Strong model-pose attitude
- Hip pop / contrapposto
- Influencer face
- Over-beautified skin

## PASS
- Is XiaoNa
- Matches frozen body constitution
- Belongs to the current layer
- No obvious structural pollution
- Safe for final training set

## WARN
- Mostly XiaoNa but slightly drifting
- Mostly correct body but local instability exists
- Neckline or limb details not fully stable
- Watch-only and backup-only
- Does not enter the main training set

## FAIL
- Not XiaoNa
- Clearly violates body constitution
- Proportion collapse
- Leg or foot failure
- Strong pose pollution
- Dressed state changes the person
- Serious training contamination signal

## Final Sealing Rule
- Final sealing requires:
  - Algorithm score
  - Human semantic judgment
  - Group balance

