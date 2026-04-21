from __future__ import annotations

from typing import Any, Dict, List


WINNER_BANK_BOOTSTRAP_DEFERRED_BLOCKER = "WINNER_BANK_BOOTSTRAP_DEFERRED_UNTIL_REVIEW_ONLY_INVARIANCE_MATURE"


def winner_bank_bootstrap_policy() -> Dict[str, Any]:
    requirements: List[str] = [
        "review_only angle invariance is stable across front / three_quarter / side-like noise",
        "review_only clothing invariance is stable under basic outfit and OUTER-style occlusion",
        "review_only lighting invariance does not confuse exposure drift with identity drift",
        "face/body 3D topology consistency is stable enough for repeated batch replay",
    ]
    return {
        "state": "deferred",
        "blocker": WINNER_BANK_BOOTSTRAP_DEFERRED_BLOCKER,
        "reason": (
            "Do not start winner_bank bootstrap until review-only angle, clothing, lighting, "
            "and 3D topology invariance are mature enough for industrial LoRA screening."
        ),
        "requirements": requirements,
        "allowed_now": [
            "use front top candidates for diagnostic review",
            "keep building input manifests",
            "run clean-lane review-only replay",
            "improve invariance metrics and evidence completeness",
        ],
        "disallowed_now": [
            "promote winner into curated winner_bank",
            "use winner_bank drift as a training admission signal",
        ],
    }


def winner_bank_bootstrap_is_deferred() -> bool:
    return str(winner_bank_bootstrap_policy().get("state") or "").strip() == "deferred"
