"""无需下载 TimesFM：验证动力学、时间边界和已归档的逐点结果。"""
import csv
import hashlib
import json
from types import SimpleNamespace
import unittest

import numpy as np
import main as demo
from orbit import MU, R_EARTH, J2, OMEGA, gravity, propagate


class PhysicsTests(unittest.TestCase):
    def test_circular_orbit_against_analytic_solution(self):
        radius = 7e6
        speed = np.sqrt(MU / radius)
        states = propagate([radius, 0, 0, 0, speed, 0], use_j2=False)
        phase = np.sqrt(MU / radius**3) * np.arange(1, 61) * 60
        exact = radius * np.column_stack([np.cos(phase), np.sin(phase), np.zeros(60)])
        self.assertLess(np.max(np.linalg.norm(states[:, :3] - exact, axis=1)), .02)
        energy = np.sum(states[:, 3:]**2, axis=1) / 2 - MU / np.linalg.norm(states[:, :3], axis=1)
        self.assertLess(np.max(np.abs(energy / (-MU / (2 * radius)) - 1)), 1e-8)

    def test_j2_matches_potential_gradient(self):
        position = np.array([5e6, 3e6, 4e6])
        def potential(r):
            radius = np.linalg.norm(r)
            return -MU / radius * (1 - J2 * (R_EARTH / radius)**2 * (.5 * (3 * (r[2] / radius)**2 - 1)))
        gradient = np.array([(potential(position + d) - potential(position - d)) / 2
                             for d in np.eye(3)])
        np.testing.assert_allclose(gravity(position), -gradient, atol=3e-8, rtol=0)

    def test_rotating_frame_matches_inertial_frame(self):
        initial = np.array([7e6, 0, 0, 0, 6500, 3800.])
        inertial = propagate(initial)
        rotating_initial = initial.copy()
        rotating_initial[3:] -= np.cross(OMEGA, initial[:3])
        rotating = propagate(rotating_initial, rotating=True)
        angles = OMEGA[2] * np.arange(1, 61) * 60
        x, y, z = inertial[:, :3].T
        expected = np.column_stack([np.cos(angles)*x + np.sin(angles)*y,
                                    -np.sin(angles)*x + np.cos(angles)*y, z])
        self.assertLess(np.max(np.linalg.norm(rotating[:, :3] - expected, axis=1)), .02)
        finer = propagate(rotating_initial, rotating=True, max_step=2.5)
        self.assertLess(np.max(np.linalg.norm(rotating[:, :3] - finer[:, :3], axis=1)), .02)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.loadtxt(demo.ROOT / "data/orbit.csv", delimiter=",", skiprows=1)

    def test_manifest_and_time_grid(self):
        manifest = json.loads((demo.ROOT / "data/orbit.manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["rows"], len(self.data))
        self.assertEqual(manifest["output_sha256"], hashlib.sha256((demo.ROOT / "data/orbit.csv").read_bytes()).hexdigest())
        self.assertTrue(np.all(np.diff(self.data[:, 0]) > 0))
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

    def test_tle_epochs_and_time_only_selection(self):
        from tle import load_tles, available_tles, positions
        source = demo.ROOT / "data/debris.tle"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), demo.INFO["tle_sha256"])
        records = load_tles()  # 同时验证六份 TLE 的固定列格式和校验和。
        self.assertEqual(len(records), 6)
        for number in range(1, demo.WINDOWS + 1):
            sample = demo.select_window(self.data, number)
            origin = sample[359, 0]
            old, latest = available_tles(origin, records)
            self.assertIs(old, records[number - 1])
            self.assertIs(latest, records[number])
            future_record = records[number + 1]
            self.assertLessEqual(latest["available"], origin)
            self.assertGreater(future_record["epoch"], origin + 3600)
            np.testing.assert_allclose(positions(latest, sample[:360, 0]), sample[:360, 1:], atol=1e-6)
            np.testing.assert_allclose(positions(future_record, sample[360:, 0]), sample[360:, 1:], atol=1e-6)
            # 把未来 TLE 的轨道内容替换成不可传播对象，不影响起报前选择。
            future_changed = records[:number + 1] + [{**r, "satellite": None} for r in records[number + 1:]]
            self.assertEqual(available_tles(origin, future_changed), [old, latest])
        with self.assertRaises(ValueError):
            available_tles(records[1]["available"] - 1, records)

    def test_timesfm_inputs_and_additive_correction(self):
        sample = demo.select_window(self.data, 1)
        inputs_seen = []
        class Evaluator:
            def predict_batch(self, inputs, **kwargs):
                inputs_seen.append(np.array(inputs))
                assert kwargs["horizon"] == 60
                return [SimpleNamespace(forecast=np.full(60, 2.)) for _ in range(3)]
        result, residual = demo.predict(sample[:360], Evaluator())
        self.assertEqual(len(inputs_seen), 2)
        np.testing.assert_allclose(inputs_seen[0], ((sample[:360, 1:4] - sample[359, 1:4])/1000).T,
                                   rtol=1e-6, atol=1e-4)
        np.testing.assert_allclose(inputs_seen[1], (residual - residual[-1]).T, rtol=1e-6, atol=1e-4)
        np.testing.assert_allclose(result["timesfm"], np.tile(sample[359, 1:4] + 2000, (60, 1)))
        np.testing.assert_allclose(result[demo.HYBRID] - result[demo.BASE],
                                   np.tile(residual[-1] + 2, (60, 1)), atol=1e-8)
        with self.assertRaises(ValueError):
            demo.predict(sample, Evaluator())  # 禁止把含未来参考的 420 点传进预测函数。

    def test_archived_metrics_from_xyz_not_plot(self):
        for device in ("cpu", "gpu"):
            folder = demo.ROOT / "reference" / device
            rms_values = {}
            for number in range(1, demo.WINDOWS + 1):
                output = folder / f"window-{number:02d}"
                pred = np.genfromtxt(output / "prediction.csv", delimiter=",", names=True)
                sample = demo.select_window(self.data, number)
                np.testing.assert_array_equal(pred["time_seconds"], sample[360:, 0])
                reference = np.column_stack([pred[f"reference_{axis}_m"] for axis in "xyz"])
                np.testing.assert_array_equal(reference, sample[360:, 1:4])
                with (output / "metrics.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                    self.assertEqual({r["model"] for r in rows},
                                     {"two_body", "j2", "sgp4_old", "sgp4_latest", "timesfm", demo.HYBRID})
                    for row in rows:
                        model = row["model"]
                        xyz = np.column_stack([pred[f"{model}_{axis}_m"] for axis in "xyz"])
                        error = np.sqrt(np.sum((xyz - reference)**2, axis=1))
                        np.testing.assert_allclose(error, pred[f"{model}_deviation_m"], rtol=1e-12)
                        rms = np.sqrt(np.mean(error**2))
                        self.assertAlmostEqual(rms, float(row["rms_3d_m"]), places=6)
                        self.assertAlmostEqual(error[-1], float(row["endpoint_3d_m"]), places=6)
                        rms_values.setdefault(model, []).append(rms)
                past = np.genfromtxt(output / "history.csv", delimiter=",", names=True)
                np.testing.assert_array_equal(past["time_seconds"], sample[:360, 0])
                for axis, col in zip("xyz", (1, 2, 3)):
                    np.testing.assert_array_equal(past[f"reference_{axis}_m"], sample[:360, col])
                    np.testing.assert_allclose(past[f"reference_{axis}_m"] - past[f"baseline_{axis}_m"],
                                               past[f"residual_{axis}_m"], atol=1e-8)
                if number == 1:
                    physics, residual = demo.predict(sample[:360])
                    for model, xyz in physics.items():
                        archived = np.column_stack([pred[f"{model}_{axis}_m"] for axis in "xyz"])
                        np.testing.assert_allclose(xyz, archived, atol=.001, rtol=0)
            with (folder / "summary.csv").open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    rms = np.asarray(rms_values[row["model"]])
                    self.assertEqual(int(row["windows"]), len(rms))
                    self.assertAlmostEqual(float(row["pooled_rms_3d_m"]), np.sqrt(np.mean(rms**2)), places=6)


if __name__ == "__main__":
    unittest.main()
