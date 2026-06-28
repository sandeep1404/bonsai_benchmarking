import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calc_tokj import calc_tokj, save_chart


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

    def test_save_chart_writes_png_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            all_data = {
                "25W": {
                    (256, 128): {"tokj": 7.5, "avg_W": 3.2},
                    (512, 256): {"tokj": 8.1, "avg_W": 3.3},
                },
                "MAXN": {
                    (256, 128): {"tokj": 9.1, "avg_W": 2.4},
                    (512, 256): {"tokj": 9.6, "avg_W": 2.5},
                },
            }
            save_chart(all_data, [(256, 128), (512, 256)], tmpdir)
            self.assertTrue(Path(tmpdir, "tokj_comparison.png").exists())
            self.assertTrue(Path(tmpdir, "tokj_avg_power.png").exists())


if __name__ == "__main__":
    unittest.main()
