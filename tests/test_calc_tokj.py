import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calc_tokj import calc_tokj


class CalcTokjTest(unittest.TestCase):
    def test_supports_request_start_ns_timestamp_field(self):
        tegra_rows = [(1_700_000_000.0, 1000, 1000)]
        profile_records = [{
            "run": 1,
            "target_prompt_tokens": 256,
            "target_gen_tokens": 128,
            "output_tokens": 64,
            "latency_s": 1.0,
            "request_start_ns": 1_700_000_000_000_000_000,
            "request_end_ns": 1_700_000_001_000_000_000,
        }]

        result = calc_tokj(tegra_rows, profile_records)

        self.assertIn((256, 128), result)
        self.assertEqual(result[(256, 128)]["n"], 1)
        self.assertGreater(result[(256, 128)]["tokj"], 0)


if __name__ == "__main__":
    unittest.main()
