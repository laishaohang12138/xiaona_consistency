from __future__ import annotations

import unittest

from core.qa_evidence_lineage import EvidenceLineageGraph


class EvidenceLineageTests(unittest.TestCase):
    def test_valid_derivation_dag(self) -> None:
        graph = EvidenceLineageGraph()
        graph.add_node("image", "OBSERVATION_ASSET")
        graph.add_node("artifact", "PROVIDER_ARTIFACT")
        graph.add_node("measurement", "NATIVE_MEASUREMENT")
        graph.add_edge("image", "artifact", "OBSERVED_FROM")
        graph.add_edge("artifact", "measurement", "DERIVED_FROM")

        payload = graph.to_dict()
        self.assertEqual(payload["status"], "VALID")
        self.assertEqual(payload["issues"], [])

    def test_cycle_is_rejected(self) -> None:
        graph = EvidenceLineageGraph()
        graph.add_node("a", "ARTIFACT")
        graph.add_node("b", "MEASUREMENT")
        graph.add_edge("a", "b", "DERIVED_FROM")
        graph.add_edge("b", "a", "DERIVED_FROM")

        self.assertIn("LINEAGE_CYCLE_DETECTED", graph.validate())

    def test_non_derivation_relation_is_not_persisted(self) -> None:
        graph = EvidenceLineageGraph()
        graph.add_node("a", "ARTIFACT")
        graph.add_node("b", "MEASUREMENT")
        with self.assertRaises(ValueError):
            graph.add_edge("a", "b", "SHARES_UPSTREAM_WITH")


if __name__ == "__main__":
    unittest.main()
