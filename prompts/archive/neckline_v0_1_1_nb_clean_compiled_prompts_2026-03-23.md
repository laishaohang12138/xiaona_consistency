# NK v0.1.1-nb-clean | Nano Banana Clean Compiled Prompts

All non-draw metadata has been decoupled from the model-facing prompt body. Operator routing notes are kept outside the prompt.

## NK-A01

- Ratio: 3:4
- Family: FAMILY A | HIGH / MOCK NECK
- Recommended shot plugin: nk_shot_F1_front_neutral.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if mock/high neck starts swallowing clavicles, thickening the neck, or collapsing the shoulder line.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted mock-neck or soft high-neck basic top. Keep the neck cylinder clean, elegant, and natural. Clavicles must remain subtly legible; the neckline must not swallow the clavicle basin. Compact knit or fine-rib knit only. Long-sleeve basic top only. No bulky roll-neck, no stacked collar, no scarf-like folds, no oversized sweater volume. The neck cylinder must stay clean and close to the neck without swallowing the clavicle basin.

Garment: clean dark-charcoal fitted mock-neck compact-knit top, smooth surface, long sleeves, no visible seam emphasis.

Aspect ratio 3:4. Static front-facing product pose. Body faces camera directly. Head faces camera directly. Both shoulders calm and nearly level. Neck natural and vertical. Weight distribution quiet and neutral. Framing head to hip. Both shoulder heads fully visible. The entire neckline edge, clavicle zone, and upper chest cut must be readable in one glance.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid bulky roll-neck, stacked collar, scarf folds, winter-sweater volume, high neck swallowing clavicles.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-A02

- Ratio: 3:4
- Family: FAMILY A | HIGH / MOCK NECK
- Recommended shot plugin: nk_shot_F2_front_natural.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if mock/high neck starts swallowing clavicles, thickening the neck, or collapsing the shoulder line.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted mock-neck or soft high-neck basic top. Keep the neck cylinder clean, elegant, and natural. Clavicles must remain subtly legible; the neckline must not swallow the clavicle basin. Compact knit or fine-rib knit only. Long-sleeve basic top only. No bulky roll-neck, no stacked collar, no scarf-like folds, no oversized sweater volume. The neck cylinder must stay clean and close to the neck without swallowing the clavicle basin.

Garment: charcoal fitted mock-neck fine-rib knit top, long sleeves, slightly richer rib texture, neckline edge still clean and tight.

Aspect ratio 3:4. Static front-facing product pose with a very small natural-comfort signal only. Body faces camera directly. Head faces camera directly. Shoulders calm and nearly level. Neck natural and uncompressed. Framing head to hip or upper hip. Both shoulder heads fully visible. The entire neckline edge and clavicle zone remain clearly readable in one glance.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid bulky roll-neck, stacked collar, scarf folds, winter-sweater volume, high neck swallowing clavicles.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-A03

- Ratio: 3:4
- Family: FAMILY A | HIGH / MOCK NECK
- Recommended shot plugin: nk_shot_Q1_3q_left_mild.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if mock/high neck starts swallowing clavicles, thickening the neck, or collapsing the shoulder line.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted mock-neck or soft high-neck basic top. Keep the neck cylinder clean, elegant, and natural. Clavicles must remain subtly legible; the neckline must not swallow the clavicle basin. Compact knit or fine-rib knit only. Long-sleeve basic top only. No bulky roll-neck, no stacked collar, no scarf-like folds, no oversized sweater volume. The neck cylinder must stay clean and close to the neck without swallowing the clavicle basin.

Garment: ink-navy fitted mock-neck compact-knit top, smooth surface, long sleeves.

Aspect ratio 3:4. Quiet standing product pose. Body rotates about 20 to 30 degrees to the left. Head follows naturally with only a small return toward the camera for stable identity. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to hip. The full neckline edge, clavicle zone, and shoulder line must remain readable despite the mild depth turn.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid bulky roll-neck, stacked collar, scarf folds, winter-sweater volume, high neck swallowing clavicles.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-A04

- Ratio: 3:4
- Family: FAMILY A | HIGH / MOCK NECK
- Recommended shot plugin: nk_shot_Q2_3q_right_mild.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if mock/high neck starts swallowing clavicles, thickening the neck, or collapsing the shoulder line.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted mock-neck or soft high-neck basic top. Keep the neck cylinder clean, elegant, and natural. Clavicles must remain subtly legible; the neckline must not swallow the clavicle basin. Compact knit or fine-rib knit only. Long-sleeve basic top only. No bulky roll-neck, no stacked collar, no scarf-like folds, no oversized sweater volume. The neck cylinder must stay clean and close to the neck without swallowing the clavicle basin.

Garment: espresso fitted soft high-neck knit top, smooth compact surface, long sleeves.

Aspect ratio 3:4. Quiet standing product pose. Body rotates about 20 to 30 degrees to the right. Head follows naturally with only a small return toward the camera for stable identity. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to hip. The full neckline edge, clavicle zone, and shoulder line must remain readable despite the mild depth turn.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid bulky roll-neck, stacked collar, scarf folds, winter-sweater volume, high neck swallowing clavicles.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-A05

- Ratio: 9:16
- Family: FAMILY A | HIGH / MOCK NECK
- Recommended shot plugin: nk_shot_V1_vertical_continuity.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if mock/high neck starts swallowing clavicles, thickening the neck, or collapsing the shoulder line.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted mock-neck or soft high-neck basic top. Keep the neck cylinder clean, elegant, and natural. Clavicles must remain subtly legible; the neckline must not swallow the clavicle basin. Compact knit or fine-rib knit only. Long-sleeve basic top only. No bulky roll-neck, no stacked collar, no scarf-like folds, no oversized sweater volume. The neck cylinder must stay clean and close to the neck without swallowing the clavicle basin.

Garment: black fitted mock-neck fine-rib top, long sleeves, clean vertical torso continuity, neckline edge tight and stable.

Aspect ratio 9:16. Quiet upright continuity pose. Body faces camera directly or with at most a very small natural turn. Head stays calm with a stable direct read. Shoulders remain relaxed and nearly level. The neckline area remains the main visual focus, while enough torso and lower-body continuity stays visible so the body read remains clearly consistent with Ref #2. Framing head to knee. The entire neckline edge, clavicle zone, shoulder line, and torso continuity must remain readable.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid bulky roll-neck, stacked collar, scarf folds, winter-sweater volume, high neck swallowing clavicles.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-A06

- Ratio: 1:1
- Family: FAMILY A | HIGH / MOCK NECK
- Recommended shot plugin: nk_shot_S1_square_product_safe.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if mock/high neck starts swallowing clavicles, thickening the neck, or collapsing the shoulder line.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted mock-neck or soft high-neck basic top. Keep the neck cylinder clean, elegant, and natural. Clavicles must remain subtly legible; the neckline must not swallow the clavicle basin. Compact knit or fine-rib knit only. Long-sleeve basic top only. No bulky roll-neck, no stacked collar, no scarf-like folds, no oversized sweater volume. The neck cylinder must stay clean and close to the neck without swallowing the clavicle basin.

Garment: ink-navy fitted soft high-neck knit top with very subtle tone-on-tone seam emphasis, no decorative seam contrast.

Aspect ratio 1:1. Static front or near-front upper-body product pose, at most a 10-degree natural turn. Head faces camera directly. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to upper waist. Both shoulder heads fully visible. The entire neckline edge and upper chest cut must stay clearly readable. The neckline area must remain the main visual focus of the frame; the image must not collapse into a face crop.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid bulky roll-neck, stacked collar, scarf folds, winter-sweater volume, high neck swallowing clavicles.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-B01

- Ratio: 3:4
- Family: FAMILY B | CREW / U NECK
- Recommended shot plugin: nk_shot_F1_front_neutral.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if neckline edge, clavicle basin, or shoulder line drifts in 3/4 or vertical lanes.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted crew-neck or modest U-neck basic top. Keep the neckline edge clean and stable with clear shoulder heads. The opening must stay modest, calm, and identity-safe. Matte jersey, compact knit, or fine-rib knit only. No scoop exaggeration, no droopy stretched collar, no underwear-tank read. The neckline edge must stay clean, stable, and close to the body without sagging.

Garment: clean black fitted crew-neck matte-jersey top, short sleeves, minimal surface noise, stable neckline edge.

Aspect ratio 3:4. Static front-facing product pose. Body faces camera directly. Head faces camera directly. Both shoulders calm and nearly level. Neck natural and vertical. Weight distribution quiet and neutral. Framing head to hip. Both shoulder heads fully visible. The entire neckline edge, clavicle zone, and upper chest cut must be readable in one glance.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid scoop-neck exaggeration, stretched collar, droopy neckline, underwear-tank read.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-B02

- Ratio: 3:4
- Family: FAMILY B | CREW / U NECK
- Recommended shot plugin: nk_shot_F2_front_natural.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if neckline edge, clavicle basin, or shoulder line drifts in 3/4 or vertical lanes.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted crew-neck or modest U-neck basic top. Keep the neckline edge clean and stable with clear shoulder heads. The opening must stay modest, calm, and identity-safe. Matte jersey, compact knit, or fine-rib knit only. No scoop exaggeration, no droopy stretched collar, no underwear-tank read. The neckline edge must stay clean, stable, and close to the body without sagging.

Garment: soft-ivory fitted modest U-neck compact-knit top, short sleeves, clean neckline edge, calm upper-chest read.

Aspect ratio 3:4. Static front-facing product pose with a very small natural-comfort signal only. Body faces camera directly. Head faces camera directly. Shoulders calm and nearly level. Neck natural and uncompressed. Framing head to hip or upper hip. Both shoulder heads fully visible. The entire neckline edge and clavicle zone remain clearly readable in one glance.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid scoop-neck exaggeration, stretched collar, droopy neckline, underwear-tank read.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-B03

- Ratio: 3:4
- Family: FAMILY B | CREW / U NECK
- Recommended shot plugin: nk_shot_Q1_3q_left_mild.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if neckline edge, clavicle basin, or shoulder line drifts in 3/4 or vertical lanes.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted crew-neck or modest U-neck basic top. Keep the neckline edge clean and stable with clear shoulder heads. The opening must stay modest, calm, and identity-safe. Matte jersey, compact knit, or fine-rib knit only. No scoop exaggeration, no droopy stretched collar, no underwear-tank read. The neckline edge must stay clean, stable, and close to the body without sagging.

Garment: charcoal fitted crew-neck fine-rib knit top, short sleeves, collar edge clean and close to the body.

Aspect ratio 3:4. Quiet standing product pose. Body rotates about 20 to 30 degrees to the left. Head follows naturally with only a small return toward the camera for stable identity. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to hip. The full neckline edge, clavicle zone, and shoulder line must remain readable despite the mild depth turn.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid scoop-neck exaggeration, stretched collar, droopy neckline, underwear-tank read.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-B04

- Ratio: 3:4
- Family: FAMILY B | CREW / U NECK
- Recommended shot plugin: nk_shot_Q2_3q_right_mild.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if neckline edge, clavicle basin, or shoulder line drifts in 3/4 or vertical lanes.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted crew-neck or modest U-neck basic top. Keep the neckline edge clean and stable with clear shoulder heads. The opening must stay modest, calm, and identity-safe. Matte jersey, compact knit, or fine-rib knit only. No scoop exaggeration, no droopy stretched collar, no underwear-tank read. The neckline edge must stay clean, stable, and close to the body without sagging.

Garment: taupe fitted modest U-neck matte-jersey top, short sleeves, calm shoulder line, no droop at the neckline.

Aspect ratio 3:4. Quiet standing product pose. Body rotates about 20 to 30 degrees to the right. Head follows naturally with only a small return toward the camera for stable identity. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to hip. The full neckline edge, clavicle zone, and shoulder line must remain readable despite the mild depth turn.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid scoop-neck exaggeration, stretched collar, droopy neckline, underwear-tank read.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-B05

- Ratio: 9:16
- Family: FAMILY B | CREW / U NECK
- Recommended shot plugin: nk_shot_V1_vertical_continuity.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if neckline edge, clavicle basin, or shoulder line drifts in 3/4 or vertical lanes.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted crew-neck or modest U-neck basic top. Keep the neckline edge clean and stable with clear shoulder heads. The opening must stay modest, calm, and identity-safe. Matte jersey, compact knit, or fine-rib knit only. No scoop exaggeration, no droopy stretched collar, no underwear-tank read. The neckline edge must stay clean, stable, and close to the body without sagging.

Garment: slate fitted crew-neck compact-knit top, long sleeves, clean torso continuity, stable collar edge.

Aspect ratio 9:16. Quiet upright continuity pose. Body faces camera directly or with at most a very small natural turn. Head stays calm with a stable direct read. Shoulders remain relaxed and nearly level. The neckline area remains the main visual focus, while enough torso and lower-body continuity stays visible so the body read remains clearly consistent with Ref #2. Framing head to knee. The entire neckline edge, clavicle zone, shoulder line, and torso continuity must remain readable.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid scoop-neck exaggeration, stretched collar, droopy neckline, underwear-tank read.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-B06

- Ratio: 1:1
- Family: FAMILY B | CREW / U NECK
- Recommended shot plugin: nk_shot_S1_square_product_safe.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if neckline edge, clavicle basin, or shoulder line drifts in 3/4 or vertical lanes.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted crew-neck or modest U-neck basic top. Keep the neckline edge clean and stable with clear shoulder heads. The opening must stay modest, calm, and identity-safe. Matte jersey, compact knit, or fine-rib knit only. No scoop exaggeration, no droopy stretched collar, no underwear-tank read. The neckline edge must stay clean, stable, and close to the body without sagging.

Garment: oatmeal fitted modest U-neck fine-rib top with very subtle tone-on-tone seam emphasis, no decorative seam contrast.

Aspect ratio 1:1. Static front or near-front upper-body product pose, at most a 10-degree natural turn. Head faces camera directly. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to upper waist. Both shoulder heads fully visible. The entire neckline edge and upper chest cut must stay clearly readable. The neckline area must remain the main visual focus of the frame; the image must not collapse into a face crop.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid scoop-neck exaggeration, stretched collar, droopy neckline, underwear-tank read.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-C01

- Ratio: 3:4
- Family: FAMILY C | V NECK
- Recommended shot plugin: nk_shot_F1_front_neutral.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if V symmetry, clavicle basin, shoulder line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted shallow-to-moderate V-neck basic top. Keep the V opening stable, modest, and product-safe, with a calm sternum read and no sensual chest emphasis. Compact knit, fine-rib knit, or soft matte jersey only. No plunging V, no wrap-dress read, no cleavage emphasis. The V opening must remain centered, modest, and stable.

Garment: clean black fitted shallow V-neck compact-knit top, long sleeves, modest opening, stable V point.

Aspect ratio 3:4. Static front-facing product pose. Body faces camera directly. Head faces camera directly. Both shoulders calm and nearly level. Neck natural and vertical. Weight distribution quiet and neutral. Framing head to hip. Both shoulder heads fully visible. The entire neckline edge, clavicle zone, and upper chest cut must be readable in one glance.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid plunging V, wrap-dress read, cleavage emphasis, bra-cup shaping.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-C02

- Ratio: 3:4
- Family: FAMILY C | V NECK
- Recommended shot plugin: nk_shot_F2_front_natural.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if V symmetry, clavicle basin, shoulder line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted shallow-to-moderate V-neck basic top. Keep the V opening stable, modest, and product-safe, with a calm sternum read and no sensual chest emphasis. Compact knit, fine-rib knit, or soft matte jersey only. No plunging V, no wrap-dress read, no cleavage emphasis. The V opening must remain centered, modest, and stable.

Garment: soft-ivory fitted moderate V-neck matte-jersey top, short sleeves, calm non-sensual chest read, stable V opening.

Aspect ratio 3:4. Static front-facing product pose with a very small natural-comfort signal only. Body faces camera directly. Head faces camera directly. Shoulders calm and nearly level. Neck natural and uncompressed. Framing head to hip or upper hip. Both shoulder heads fully visible. The entire neckline edge and clavicle zone remain clearly readable in one glance.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid plunging V, wrap-dress read, cleavage emphasis, bra-cup shaping.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-C03

- Ratio: 3:4
- Family: FAMILY C | V NECK
- Recommended shot plugin: nk_shot_Q1_3q_left_mild.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if V symmetry, clavicle basin, shoulder line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted shallow-to-moderate V-neck basic top. Keep the V opening stable, modest, and product-safe, with a calm sternum read and no sensual chest emphasis. Compact knit, fine-rib knit, or soft matte jersey only. No plunging V, no wrap-dress read, no cleavage emphasis. The V opening must remain centered, modest, and stable.

Garment: charcoal fitted shallow V-neck fine-rib knit top, long sleeves, modest V depth, clean sternum line.

Aspect ratio 3:4. Quiet standing product pose. Body rotates about 20 to 30 degrees to the left. Head follows naturally with only a small return toward the camera for stable identity. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to hip. The full neckline edge, clavicle zone, and shoulder line must remain readable despite the mild depth turn.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid plunging V, wrap-dress read, cleavage emphasis, bra-cup shaping.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-C04

- Ratio: 3:4
- Family: FAMILY C | V NECK
- Recommended shot plugin: nk_shot_Q2_3q_right_mild.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if V symmetry, clavicle basin, shoulder line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted shallow-to-moderate V-neck basic top. Keep the V opening stable, modest, and product-safe, with a calm sternum read and no sensual chest emphasis. Compact knit, fine-rib knit, or soft matte jersey only. No plunging V, no wrap-dress read, no cleavage emphasis. The V opening must remain centered, modest, and stable.

Garment: taupe fitted moderate V-neck soft-drape jersey top, long sleeves, controlled drape, stable V edges.

Aspect ratio 3:4. Quiet standing product pose. Body rotates about 20 to 30 degrees to the right. Head follows naturally with only a small return toward the camera for stable identity. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to hip. The full neckline edge, clavicle zone, and shoulder line must remain readable despite the mild depth turn.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid plunging V, wrap-dress read, cleavage emphasis, bra-cup shaping.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-C05

- Ratio: 9:16
- Family: FAMILY C | V NECK
- Recommended shot plugin: nk_shot_V1_vertical_continuity.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if V symmetry, clavicle basin, shoulder line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted shallow-to-moderate V-neck basic top. Keep the V opening stable, modest, and product-safe, with a calm sternum read and no sensual chest emphasis. Compact knit, fine-rib knit, or soft matte jersey only. No plunging V, no wrap-dress read, no cleavage emphasis. The V opening must remain centered, modest, and stable.

Garment: ink-navy fitted shallow V-neck compact-knit top, long sleeves, clean torso continuity, modest opening.

Aspect ratio 9:16. Quiet upright continuity pose. Body faces camera directly or with at most a very small natural turn. Head stays calm with a stable direct read. Shoulders remain relaxed and nearly level. The neckline area remains the main visual focus, while enough torso and lower-body continuity stays visible so the body read remains clearly consistent with Ref #2. Framing head to knee. The entire neckline edge, clavicle zone, shoulder line, and torso continuity must remain readable.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid plunging V, wrap-dress read, cleavage emphasis, bra-cup shaping.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-C06

- Ratio: 1:1
- Family: FAMILY C | V NECK
- Recommended shot plugin: nk_shot_S1_square_product_safe.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if V symmetry, clavicle basin, shoulder line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted shallow-to-moderate V-neck basic top. Keep the V opening stable, modest, and product-safe, with a calm sternum read and no sensual chest emphasis. Compact knit, fine-rib knit, or soft matte jersey only. No plunging V, no wrap-dress read, no cleavage emphasis. The V opening must remain centered, modest, and stable.

Garment: black fitted moderate V-neck knit top with very subtle tone-on-tone seam emphasis, still modest, no cleavage read.

Aspect ratio 1:1. Static front or near-front upper-body product pose, at most a 10-degree natural turn. Head faces camera directly. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to upper waist. Both shoulder heads fully visible. The entire neckline edge and upper chest cut must stay clearly readable. The neckline area must remain the main visual focus of the frame; the image must not collapse into a face crop.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid plunging V, wrap-dress read, cleavage emphasis, bra-cup shaping.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-D01

- Ratio: 3:4
- Family: FAMILY D | SHIRT COLLAR / OPEN COLLAR
- Recommended shot plugin: nk_shot_F1_front_neutral.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if collar lay, collar points, placket line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: clean shirt-collar or open-collar basic top. Keep the collar stand stable, the collar points clean, the collar lay natural, and the placket line straight and readable. Matte shirting or clean knit-shirt hybrid only. No oversized shirt volume, no stripe, no check, no jacket-like silhouette, no floating collar points, no warped placket. Collar symmetry should stay natural and stable; the placket must remain straight enough to read clearly.

Garment: soft-ivory matte shirt-collar top, long sleeves, stable collar stand, closed-but-relaxed collar state, clean placket.

Aspect ratio 3:4. Static front-facing product pose. Body faces camera directly. Head faces camera directly. Both shoulders calm and nearly level. Neck natural and vertical. Weight distribution quiet and neutral. Framing head to hip. Both shoulder heads fully visible. The entire neckline edge, clavicle zone, and upper chest cut must be readable in one glance.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid floating collar points, warped placket, twisted placket, jagged collar edge, shirt collar eating the neck.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-D02

- Ratio: 3:4
- Family: FAMILY D | SHIRT COLLAR / OPEN COLLAR
- Recommended shot plugin: nk_shot_F2_front_natural.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if collar lay, collar points, placket line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: clean shirt-collar or open-collar basic top. Keep the collar stand stable, the collar points clean, the collar lay natural, and the placket line straight and readable. Matte shirting or clean knit-shirt hybrid only. No oversized shirt volume, no stripe, no check, no jacket-like silhouette, no floating collar points, no warped placket. Collar symmetry should stay natural and stable; the placket must remain straight enough to read clearly.

Garment: charcoal matte open-collar shirt top, long sleeves, one-button-open state, clean placket, natural collar lay.

Aspect ratio 3:4. Static front-facing product pose with a very small natural-comfort signal only. Body faces camera directly. Head faces camera directly. Shoulders calm and nearly level. Neck natural and uncompressed. Framing head to hip or upper hip. Both shoulder heads fully visible. The entire neckline edge and clavicle zone remain clearly readable in one glance.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid floating collar points, warped placket, twisted placket, jagged collar edge, shirt collar eating the neck.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-D03

- Ratio: 3:4
- Family: FAMILY D | SHIRT COLLAR / OPEN COLLAR
- Recommended shot plugin: nk_shot_Q1_3q_left_mild.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if collar lay, collar points, placket line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: clean shirt-collar or open-collar basic top. Keep the collar stand stable, the collar points clean, the collar lay natural, and the placket line straight and readable. Matte shirting or clean knit-shirt hybrid only. No oversized shirt volume, no stripe, no check, no jacket-like silhouette, no floating collar points, no warped placket. Collar symmetry should stay natural and stable; the placket must remain straight enough to read clearly.

Garment: ink-navy knit-shirt hybrid top with a clean open collar, slightly firmer fabric lay, stable collar points and placket.

Aspect ratio 3:4. Quiet standing product pose. Body rotates about 20 to 30 degrees to the left. Head follows naturally with only a small return toward the camera for stable identity. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to hip. The full neckline edge, clavicle zone, and shoulder line must remain readable despite the mild depth turn.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid floating collar points, warped placket, twisted placket, jagged collar edge, shirt collar eating the neck.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-D04

- Ratio: 9:16
- Family: FAMILY D | SHIRT COLLAR / OPEN COLLAR
- Recommended shot plugin: nk_shot_V1_vertical_continuity.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if collar lay, collar points, placket line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: clean shirt-collar or open-collar basic top. Keep the collar stand stable, the collar points clean, the collar lay natural, and the placket line straight and readable. Matte shirting or clean knit-shirt hybrid only. No oversized shirt volume, no stripe, no check, no jacket-like silhouette, no floating collar points, no warped placket. Collar symmetry should stay natural and stable; the placket must remain straight enough to read clearly.

Garment: oatmeal matte shirt-collar top, long sleeves, clean placket, stable collar stand, calm vertical torso continuity.

Aspect ratio 9:16. Quiet upright continuity pose. Body faces camera directly or with at most a very small natural turn. Head stays calm with a stable direct read. Shoulders remain relaxed and nearly level. The neckline area remains the main visual focus, while enough torso and lower-body continuity stays visible so the body read remains clearly consistent with Ref #2. Framing head to knee. The entire neckline edge, clavicle zone, shoulder line, and torso continuity must remain readable.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid floating collar points, warped placket, twisted placket, jagged collar edge, shirt collar eating the neck.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-D05

- Ratio: 9:16
- Family: FAMILY D | SHIRT COLLAR / OPEN COLLAR
- Recommended shot plugin: nk_shot_V2_vertical_mild_right.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if collar lay, collar points, placket line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: clean shirt-collar or open-collar basic top. Keep the collar stand stable, the collar points clean, the collar lay natural, and the placket line straight and readable. Matte shirting or clean knit-shirt hybrid only. No oversized shirt volume, no stripe, no check, no jacket-like silhouette, no floating collar points, no warped placket. Collar symmetry should stay natural and stable; the placket must remain straight enough to read clearly.

Garment: espresso clean open-collar shirt top with subtle tone-on-tone placket emphasis, stable collar points, no decorative contrast.

Aspect ratio 9:16. Quiet upright continuity pose with a very small rightward turn. Body rotates about 10 to 15 degrees to the right, never dynamic. Head returns slightly toward the camera for stable identity. Shoulders stay relaxed and nearly level. The neckline, collar lay, and placket remain the main visual focus, while enough torso and lower-body continuity stays visible so the body read remains clearly consistent with Ref #2. Framing head to mid-thigh. The collar opening, placket line, clavicle zone, and torso continuity must remain readable.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid floating collar points, warped placket, twisted placket, jagged collar edge, shirt collar eating the neck.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-D06

- Ratio: 1:1
- Family: FAMILY D | SHIRT COLLAR / OPEN COLLAR
- Recommended shot plugin: nk_shot_S1_square_product_safe.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if collar lay, collar points, placket line, or neck-column continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: clean shirt-collar or open-collar basic top. Keep the collar stand stable, the collar points clean, the collar lay natural, and the placket line straight and readable. Matte shirting or clean knit-shirt hybrid only. No oversized shirt volume, no stripe, no check, no jacket-like silhouette, no floating collar points, no warped placket. Collar symmetry should stay natural and stable; the placket must remain straight enough to read clearly.

Garment: soft-ivory open-collar knit-shirt hybrid top with subtle tone-on-tone placket emphasis, clean collar lay, no decorative contrast.

Aspect ratio 1:1. Static front or near-front upper-body product pose, at most a 10-degree natural turn. Head faces camera directly. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to upper waist. Both shoulder heads fully visible. The entire neckline edge and upper chest cut must stay clearly readable. The neckline area must remain the main visual focus of the frame; the image must not collapse into a face crop.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid floating collar points, warped placket, twisted placket, jagged collar edge, shirt collar eating the neck.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-E01

- Ratio: 3:4
- Family: FAMILY E | OFF-SHOULDER / HALTER
- Recommended shot plugin: nk_shot_F1_front_neutral.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if shoulder-top edge, clavicle basin, neck length, or upper-boundary continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted off-shoulder or fitted halter basic top. Keep the upper chest read calm, the shoulder definition natural, the neck length unchanged, and the result product-safe rather than lingerie-like. Smooth matte jersey or compact knit only. No bra-cup shape, no string-strap styling, no glossy stretch fabric, no swimwear read, no dramatic bare-skin styling. The garment edge must stay clean and supportive, never drooping, never lingerie-like.

Garment: clean black fitted off-shoulder matte top, straight calm upper edge, natural shoulder line, no sweetheart shaping, no lingerie read.

Aspect ratio 3:4. Static front-facing product pose. Body faces camera directly. Head faces camera directly. Both shoulders calm and nearly level. Neck natural and vertical. Weight distribution quiet and neutral. Framing head to hip. Both shoulder heads fully visible. The entire neckline edge, clavicle zone, and upper chest cut must be readable in one glance.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid bra-cup shape, exposed bra strap, lingerie vibe, swimwear vibe, string-strap halter, halter pulling the neck inward, decorative straps.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-E02

- Ratio: 3:4
- Family: FAMILY E | OFF-SHOULDER / HALTER
- Recommended shot plugin: nk_shot_F1_front_neutral.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if shoulder-top edge, clavicle basin, neck length, or upper-boundary continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted off-shoulder or fitted halter basic top. Keep the upper chest read calm, the shoulder definition natural, the neck length unchanged, and the result product-safe rather than lingerie-like. Smooth matte jersey or compact knit only. No bra-cup shape, no string-strap styling, no glossy stretch fabric, no swimwear read, no dramatic bare-skin styling. The garment edge must stay clean and supportive, never drooping, never lingerie-like.

Garment: charcoal fitted halter matte top with a broad clean halter line, natural neck column, stable shoulder width, no string straps.

Aspect ratio 3:4. Static front-facing product pose. Body faces camera directly. Head faces camera directly. Both shoulders calm and nearly level. Neck natural and vertical. Weight distribution quiet and neutral. Framing head to hip. Both shoulder heads fully visible. The entire neckline edge, clavicle zone, and upper chest cut must be readable in one glance.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid bra-cup shape, exposed bra strap, lingerie vibe, swimwear vibe, string-strap halter, halter pulling the neck inward, decorative straps.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-E03

- Ratio: 3:4
- Family: FAMILY E | OFF-SHOULDER / HALTER
- Recommended shot plugin: nk_shot_Q1_3q_left_mild.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if shoulder-top edge, clavicle basin, neck length, or upper-boundary continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted off-shoulder or fitted halter basic top. Keep the upper chest read calm, the shoulder definition natural, the neck length unchanged, and the result product-safe rather than lingerie-like. Smooth matte jersey or compact knit only. No bra-cup shape, no string-strap styling, no glossy stretch fabric, no swimwear read, no dramatic bare-skin styling. The garment edge must stay clean and supportive, never drooping, never lingerie-like.

Garment: espresso fitted off-shoulder compact-knit top, slightly richer surface texture, straight calm upper edge, no lingerie read.

Aspect ratio 3:4. Quiet standing product pose. Body rotates about 20 to 30 degrees to the left. Head follows naturally with only a small return toward the camera for stable identity. Shoulders remain calm and nearly level. Neck remains natural and elongated. Framing head to hip. The full neckline edge, clavicle zone, and shoulder line must remain readable despite the mild depth turn.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid bra-cup shape, exposed bra strap, lingerie vibe, swimwear vibe, string-strap halter, halter pulling the neck inward, decorative straps.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```

## NK-E04

- Ratio: 9:16
- Family: FAMILY E | OFF-SHOULDER / HALTER
- Recommended shot plugin: nk_shot_V1_vertical_continuity.txt
- Ref routing: Baseline Ref #1 + Ref #2. Add Ref #3 only if shoulder-top edge, clavicle basin, neck length, or upper-boundary continuity drifts.

### First-pass prompt

```text
Use Ref #1 for facial identity geometry only. Use Ref #2 for body constitution, shoulder width, torso identity, upper-to-lower continuity, and framing only. If Ref #3 is provided, use it only to stabilize the shoulder line, clavicle basin, neckline edge, collar lay, placket line, and neck-column continuity; never let Ref #3 change the face or the body constitution. One woman only, same XiaoNa, no redesign, no beautify, no face replacement drift.

Same face as Ref #1: soft oval face, rounded chin, natural jaw width, blunt rounded nose tip, natural eye size and spacing, natural brows, relaxed closed mouth, calm neutral expression, no smile, no teeth, no pout. Do not genericize the face. Do not make her younger, cuter, glamorous, or influencer-like.

Black center-part low ponytail, tidy natural hairline, hair tucked behind ears or behind shoulders, both clavicles and the entire neckline edge fully visible, no loose strands crossing the neckline. Small silver hoop earrings only if visible. No necklace, no choker, no scarf, no brooch, no extra accessories.

Soft studio broad light with clean frontal fill and subtle edge separation. Neutral-cool white balance. Real skin texture, visible pores, no beauty filter, no wax skin, no plastic skin, no heavy makeup. Face, neck, clavicles, and upper chest must stay in the same exposure family. No dark neck, no dark clavicles, no red chest patch, no muddy collar shadow.

Slim Chinese woman, 170 cm visual height, 50 kg visual weight, slender professional model build, long-leg proportion, 1:7.5 to 1:8 head-to-body ratio, straight relaxed shoulders, distinct clavicles, 30 cm neck, high natural waist if visible, narrow compact pelvis if visible. Shoulder width must stay equal to Ref #2. The neckline must never shorten the neck, widen the shoulders, thicken the trapezius, enlarge the bust, or change torso identity.

Clean basic wardrobe only. No outerwear, no layering, no blazer, no cardigan, no coat, no jacket. No logo, no print, no slogan, no lace, no sheer fabric, no mesh, no metallic shine, no sequins, no lingerie read.

Clean neutral grey seamless studio background, no props, no furniture, no text, no watermark. 65 mm equivalent lens, neutral perspective, eye-level or slightly upper-chest-level camera, level horizon, straight verticals. Static standing or quiet upright pose only. Hands stay away from the neckline and chest. Hair must never block the garment-skin boundary.

Photorealistic, natural anatomy, stable identity, high detail on face, clavicles, neckline edge, collar seam or placket when present, and upper-chest contour. Product-reference image, not glamour, not campaign, not sensual.

Garment family: fitted off-shoulder or fitted halter basic top. Keep the upper chest read calm, the shoulder definition natural, the neck length unchanged, and the result product-safe rather than lingerie-like. Smooth matte jersey or compact knit only. No bra-cup shape, no string-strap styling, no glossy stretch fabric, no swimwear read, no dramatic bare-skin styling. The garment edge must stay clean and supportive, never drooping, never lingerie-like.

Garment: black fitted halter top with a broad clean halter line, calm vertical continuity, product-safe non-lingerie read.

Aspect ratio 9:16. Quiet upright continuity pose. Body faces camera directly or with at most a very small natural turn. Head stays calm with a stable direct read. Shoulders remain relaxed and nearly level. The neckline area remains the main visual focus, while enough torso and lower-body continuity stays visible so the body read remains clearly consistent with Ref #2. Framing head to knee. The entire neckline edge, clavicle zone, shoulder line, and torso continuity must remain readable.

Avoid: wrong identity, face drift, generic face, influencer face, doll face, over-beautified face, altered facial proportions, glamour portrait crop, face-only crop, macro face crop, heavy makeup, thick neck, short neck, swollen neck, compressed neck column, swallowed clavicles, lost clavicle basin, collapsed shoulder line, broadened shoulders, trapezius inflation, chest-forward pose, runway pose, contrapposto, hip pop, dramatic twist, hands touching the collar, hands covering the chest, hair blocking the neckline, wide-angle distortion, fisheye look, tilted horizon, cropped shoulder heads, cropped collar edge, busy background, props, furniture, textured wall, bright pure white background, black void background, text, watermark.

Avoid bra-cup shape, exposed bra strap, lingerie vibe, swimwear vibe, string-strap halter, halter pulling the neck inward, decorative straps.
```

### Shortlist reroll add-on

```text
Reroll add-on: chalky neck, dark neck, dark clavicles, red chest patch, muddy collar shadow, underexposed face, underexposed upper chest, asymmetric neckline depth, unstable collar lay, unreadable placket, clipped fabric edge, fabric sinking into skin, jagged collar seam, glossy skin, wax skin, plastic skin, retouched skin, synthetic chest highlight, unreadable shoulder heads, unreadable upper chest cut, over-styled campaign gloss.
```
