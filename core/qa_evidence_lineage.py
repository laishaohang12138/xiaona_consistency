from __future__ import annotations

from typing import Any, Dict, List, Optional


EVIDENCE_LINEAGE_SCHEMA = "identity_evidence_lineage_v0_1"
DERIVATION_RELATIONS = {"OBSERVED_FROM", "TRANSFORMED_FROM", "DERIVED_FROM"}


class EvidenceLineageGraph:
    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: List[Dict[str, str]] = []

    def add_node(
        self,
        node_id: str,
        node_type: str,
        *,
        evidence_family: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        normalized_id = str(node_id).strip()
        if not normalized_id:
            raise ValueError("lineage node_id must not be empty")
        node = {
            "node_id": normalized_id,
            "node_type": str(node_type).strip().upper(),
            "evidence_family": str(evidence_family or "").strip() or None,
            "attributes": dict(attributes or {}),
        }
        existing = self._nodes.get(normalized_id)
        if existing is not None and existing != node:
            raise ValueError(f"lineage node_id collision: {normalized_id}")
        self._nodes[normalized_id] = node
        return normalized_id

    def add_edge(self, source_id: str, target_id: str, relation: str) -> None:
        normalized_relation = str(relation).strip().upper()
        if normalized_relation not in DERIVATION_RELATIONS:
            raise ValueError(f"unsupported lineage relation: {relation!r}")
        edge = {
            "source_id": str(source_id).strip(),
            "target_id": str(target_id).strip(),
            "relation": normalized_relation,
        }
        if edge not in self._edges:
            self._edges.append(edge)

    def validate(self) -> List[str]:
        issues: List[str] = []
        adjacency: Dict[str, List[str]] = {node_id: [] for node_id in self._nodes}
        for edge in self._edges:
            source_id = edge["source_id"]
            target_id = edge["target_id"]
            if source_id not in self._nodes:
                issues.append(f"LINEAGE_SOURCE_MISSING:{source_id}")
                continue
            if target_id not in self._nodes:
                issues.append(f"LINEAGE_TARGET_MISSING:{target_id}")
                continue
            adjacency[source_id].append(target_id)

        state: Dict[str, int] = {}

        def visit(node_id: str) -> bool:
            node_state = state.get(node_id, 0)
            if node_state == 1:
                return True
            if node_state == 2:
                return False
            state[node_id] = 1
            if any(visit(child_id) for child_id in adjacency.get(node_id, [])):
                return True
            state[node_id] = 2
            return False

        if any(visit(node_id) for node_id in adjacency if state.get(node_id, 0) == 0):
            issues.append("LINEAGE_CYCLE_DETECTED")
        return list(dict.fromkeys(issues))

    def to_dict(self) -> Dict[str, Any]:
        issues = self.validate()
        return {
            "schema_version": EVIDENCE_LINEAGE_SCHEMA,
            "status": "VALID" if not issues else "INVALID",
            "nodes": [self._nodes[key] for key in sorted(self._nodes)],
            "edges": sorted(
                self._edges,
                key=lambda row: (row["source_id"], row["target_id"], row["relation"]),
            ),
            "issues": issues,
        }
