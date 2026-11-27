"""Dataset invariants the analysis relies on.   Run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_llm import eda  # noqa: E402
from privacy_llm.config import RAW_DIR, SPLITS  # noqa: E402
from privacy_llm.data import FIELDS, load_records, profiles_frame, to_frame, value_kind  # noqa: E402

HAVE_DATA = all((RAW_DIR / f).exists() for f in SPLITS.values())


class UnitTests(unittest.TestCase):
    def test_value_kind_bool_before_int(self):
        self.assertEqual(value_kind(True), "bool")
        self.assertEqual(value_kind(3), "int")
        self.assertEqual(value_kind(None), "null")

    def test_matchable(self):
        self.assertFalse(eda.is_matchable([]))
        self.assertFalse(eda.is_matchable(False))
        self.assertFalse(eda.is_matchable("None"))
        self.assertFalse(eda.is_matchable(34))
        self.assertTrue(eda.is_matchable(["Hypertension"]))

    def test_fields_unique(self):
        self.assertEqual(len(FIELDS), 26)
        self.assertEqual(len(set(FIELDS)), 26)


@unittest.skipUnless(HAVE_DATA, "run scripts/download_data.py first")
class DatasetInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = load_records("full")
        cls.df = to_frame(cls.full)

    def test_full_grid(self):
        ov = eda.overview(self.df)
        self.assertEqual((ov["records"], ov["profiles"], ov["scenarios"], ov["fields"]), (4160, 20, 8, 26))
        self.assertTrue(ov["full_grid"])
        self.assertEqual(ov["duplicate_keys"], 0)

    def test_label_depends_only_on_scenario_and_field(self):
        self.assertTrue(eda.mixed_label_cells(self.df).empty)
        self.assertEqual(eda.norm_matrix(self.df).shape, (8, 26))

    def test_ref_answer_is_profile_value_or_none_string(self):
        c = eda.answer_checks(self.df)
        self.assertEqual(c["shared_ref_equals_profile_value"], c["shared_total"])
        self.assertEqual(c["withheld_ref_is_none_string"], c["withheld_total"])

    def test_small_is_subset(self):
        rep = eda.subset_report(load_records("small"), self.full)
        self.assertEqual(rep["identical_in_full"], rep["records"])

    def test_swap_pairs_consistent(self):
        rep = eda.swap_report(load_records("swap_original"), load_records("swap_flipped"), self.full)
        for k in ("same_task_and_label", "only_asked_field_changed", "answers_swapped_when_shared",
                  "original_labels_match_full"):
            self.assertTrue(rep[k], k)

    def test_profiles(self):
        self.assertEqual(len(profiles_frame(self.full)), 20)


if __name__ == "__main__":
    unittest.main()
