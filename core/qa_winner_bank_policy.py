from __future__ import annotations

from typing import Any, Dict, List


WINNER_BANK_BOOTSTRAP_DEFERRED_BLOCKER = "WINNER_BANK_BOOTSTRAP_DEFERRED_UNTIL_REVIEW_ONLY_INVARIANCE_MATURE"


def winner_bank_bootstrap_policy() -> Dict[str, Any]:
    requirements: List[str] = [
        "review_only angle invariance is stable across front / three_quarter / side-like noise",
        "review_only clothing invariance is stable under basic outfit and OUTER-style occlusion",
        "review_only lighting invariance does not confuse exposure drift with identity drift",
        "face/body 3D topology consistency is stable enough for repeated batch replay",
        "body truth reads separate pose/gait expression from unexplained body-structure drift",
    ]
    return {
        "state": "mutable_candidate_memory",
        "freeze_state": "not_frozen",
        "blocker": None,
        "reason": (
            "Winner bank may remain mutable for GPT-plus-human review memory. Do not freeze it, "
            "use it as identity truth, use it as a final-admission signal, or use it to decide the "
            "final image set. This project only screens, routes review priority, and packages "
            "evidence for external decision flows."
        ),
        "requirements": requirements,
        "allowed_now": [
            "use front top candidates for diagnostic review",
            "manually record or update human-confirmed candidates as mutable winner_bank entries",
            "keep building input manifests",
            "run clean-lane review-only replay",
            "improve invariance metrics and evidence completeness",
        ],
        "disallowed_now": [
            "freeze winner_bank as a release reference",
            "treat winner_bank as a new identity or body truth source",
            "use winner_bank drift as a final-admission signal",
            "use winner_bank entries as final image-set membership",
            "feed winner_bank entries into parameter fitting",
        ],
    }


def winner_bank_bootstrap_is_deferred() -> bool:
    return str(winner_bank_bootstrap_policy().get("state") or "").strip() == "deferred"
