"""无需下载 TimesFM：验证数据时间边界与根数预测入口。"""
import hashlib
from types import SimpleNamespace
import unittest

import numpy as np
import main as demo
from elements import prepare_history
from tle import load_tles, positions


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.loadtxt(demo.ROOT / "data/orbit.csv", delimiter=",", skiprows=1)

    def test_manifest_and_time_grid(self):
        self.assertEqual(demo.INFO["rows"], len(self.data))
        self.assertEqual(demo.INFO["output_sha256"],
                         hashlib.sha256((demo.ROOT / "data/orbit.csv").read_bytes()).hexdigest())
        self.assertEqual(demo.INFO["tle_sha256"],
                         hashlib.sha256((demo.ROOT / "data/debris.tle").read_bytes()).hexdigest())
        for number in range(1, demo.WINDOWS + 1):
            sample = demo.select_window(self.data, number)
            self.assertEqual(len(sample), 420)
            self.assertEqual(sample[359, 0], demo.origin_seconds(number))
            self.assertEqual(sample[360, 0], demo.origin_seconds(number) + 60)
            self.assertEqual(sample[-1, 0], demo.origin_seconds(number) + 3600)
        sample = demo.select_window(self.data, 1)
        with self.assertRaises(ValueError):
            demo.select_window(np.delete(sample, 200, axis=0), 1)
        with self.assertRaises(ValueError):
            demo.select_window(sample[::-1], 1)

    def test_past_tles_and_held_out_reference_are_separated(self):
        records = load_tles()
        self.assertEqual(len(records), 6)
        for number in range(1, demo.WINDOWS + 1):
            sample = demo.select_window(self.data, number)
            origin = sample[359, 0]
            known, *_ = prepare_history(origin, records)
            self.assertEqual(known, records[:number + 1])
            self.assertGreater(records[number + 1]["epoch"], origin + 3600)
            np.testing.assert_allclose(positions(known[-1], sample[:360, 0]), sample[:360, 1:], atol=1e-6)
            np.testing.assert_allclose(positions(records[number + 1], sample[360:, 0]), sample[360:, 1:], atol=1e-6)
        with self.assertRaises(ValueError):
            prepare_history(records[1]["available"] - 1, records)

    def test_default_forecast_only_calls_six_element_model(self):
        calls = []
        class Evaluator:
            def predict_batch(self, inputs, horizon, **kwargs):
                calls.append((len(inputs), horizon))
                return [SimpleNamespace(forecast=np.full(horizon, x[-1])) for x in inputs]
        result, audit = demo.predict(demo.origin_seconds(1), Evaluator())
        self.assertEqual(calls, [(6, 25)])
        self.assertEqual(set(result), {"sgp4_latest", "elements_linear", "elements_timesfm"})
        self.assertTrue(all(v.shape == (60, 3) and np.isfinite(v).all() for v in result.values()))
        baselines, _ = demo.predict(demo.origin_seconds(1))
        self.assertEqual(set(baselines), {"sgp4_latest", "elements_linear"})
        for model in baselines:
            np.testing.assert_array_equal(baselines[model], result[model])


if __name__ == "__main__":
    unittest.main()
