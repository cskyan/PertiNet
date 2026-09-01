import json
import math
import pickle
import unittest
from pathlib import Path

from scripts.validate_bernett import validate
from model.evaluation import apply_operating_boundary, select_mcc_threshold
from examples.dset_smoke import validate_dset_smoke


ROOT = Path(__file__).resolve().parents[1]


class ReleaseIntegrityTests(unittest.TestCase):
    def test_deepppisp_referenced_full_dset_split(self):
        source = ROOT / "data" / "Dset_186_72_PDB164" / "source"
        with (source / "fused_training_list.pkl").open("rb") as handle:
            train = set(pickle.load(handle))
        with (source / "fused_validing_list.pkl").open("rb") as handle:
            validation = set(pickle.load(handle))
        with (source / "fused_test_list.pkl").open("rb") as handle:
            test = set(pickle.load(handle))
        self.assertEqual([len(train), len(validation), len(test)], [302, 50, 70])
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(train | validation | test, set(range(422)))

    def test_three_source_dset_executable_example(self):
        report = validate_dset_smoke()
        self.assertEqual(report["records"], 3)
        self.assertEqual(
            report["source_subsets"],
            ["Dset_186", "Dset_72", "PDBset_164"],
        )
        self.assertEqual(report["sequence_channels"], 40)
        self.assertEqual(report["dssp_scalar_channels"], 9)
        self.assertEqual(report["ca_cutoff_angstrom"], 10.0)
        self.assertGreater(report["ca_graph_edges"], 0)

    def test_bernett_fixed_splits(self):
        report = validate(ROOT / "data" / "bernett")
        self.assertEqual(report["splits"]["Intra1"]["total_pairs"], 163_192)
        self.assertEqual(report["splits"]["Intra0"]["total_pairs"], 59_260)
        self.assertEqual(report["splits"]["Intra2"]["total_pairs"], 52_048)
        self.assertEqual(set(report["protein_overlap"].values()), {0})

    def test_rbp400_locked_case_summary(self):
        summary = json.loads(
            (ROOT / "case_study" / "rbp400" / "results" / "prediction_summary.json").read_text()
        )
        self.assertEqual(summary["case_study_name"], "RBP400")
        self.assertEqual(summary["models"], 5)
        self.assertEqual(summary["hcc_scored"], 16_471)
        self.assertEqual(summary["lung_scored"], 27_261)
        self.assertEqual(summary["shared_scored"], 5_151)
        self.assertEqual(summary["case_pairs_used_for_training"], 0)
        self.assertFalse(summary["resource_roles"]["public_benchmark_claim"])
        self.assertTrue(math.isclose(summary["development_metrics"]["auprc"]["mean"], 0.6042717958562601))

    def test_validation_selected_boundary_is_frozen_for_test(self):
        threshold = select_mcc_threshold([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
        predictions, confidence = apply_operating_boundary([0.2, 0.8], threshold)
        self.assertEqual(predictions.tolist(), [0, 1])
        self.assertTrue((confidence >= 0).all())


if __name__ == "__main__":
    unittest.main()
