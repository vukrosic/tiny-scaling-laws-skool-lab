from __future__ import annotations

import unittest

from skool_studies import Measurement, parse_positive_list, summarize


class SkoolStudiesTests(unittest.TestCase):
    def test_positive_list_is_sorted(self) -> None:
        self.assertEqual(parse_positive_list("30,5,15"), (5, 15, 30))

    def test_summary_groups_replicates(self) -> None:
        measurements = [
            Measurement(seed=1, scale=10, value=0.2, exact_accuracy=None, seconds=0.1),
            Measurement(seed=2, scale=10, value=0.4, exact_accuracy=None, seconds=0.1),
            Measurement(seed=1, scale=20, value=0.1, exact_accuracy=None, seconds=0.1),
            Measurement(seed=2, scale=20, value=0.1, exact_accuracy=None, seconds=0.1),
        ]
        summaries = summarize(measurements)
        self.assertAlmostEqual(summaries[0].mean, 0.3)
        self.assertAlmostEqual(summaries[0].standard_deviation, 0.1)
        self.assertAlmostEqual(summaries[1].mean, 0.1)


if __name__ == "__main__":
    unittest.main()
