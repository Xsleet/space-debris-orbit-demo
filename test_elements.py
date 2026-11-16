"""根数实验：检查相位、SGP4 单位/历元和未来信息隔离，不下载大模型。"""
import csv
import json
from types import SimpleNamespace
import unittest

import numpy as np
from elements import (NAMES, TAU, make_record, parameters, prepare_history,
                      predict_elements, restore_parameters)
from main import ROOT, origin_seconds, select_window
from tle import DELAY, load_tles, positions


class ElementTests(unittest.TestCase):
    def test_sgp4_reinitialization_preserves_orbit(self):
        # 原始根数在原始历元重建，应恢复同一条轨道；检验 rev/day、rad 和 epoch。
        for record in load_tles():
            rebuilt = make_record(parameters(record), record["epoch"], record)
            times = record["epoch"] + np.array([-3600., 0., 3600.])
            np.testing.assert_allclose(positions(rebuilt, times), positions(record, times),
                                       atol=.01, rtol=0)

    @staticmethod
    def linear_history():
        base = load_tles()[-1]
        latest = np.array([14.2, .001, 1.2, .05, .1, .2])
        rate = np.array([0., 0., 0., .003, .002, 14.2 * TAU / 24 + .004])
        records = []
        for hour in [-72., -36., 0.]:
            values = latest + hour * rate
            values[3:] %= TAU
            epoch = base["epoch"] + hour * 3600
            record = make_record(values, epoch, base)
            record["available"] = epoch + DELAY
            records.append(record)
        return records, latest, rate

    def test_fast_phase_and_angle_wrap_on_sparse_tles(self):
        records, latest, rate = self.linear_history()
        origin = records[-1]["available"]
        known, hours, raw, values, grid, inputs = prepare_history(origin, records)
        # 三天跨越四十多圈，仍须恢复慢相位，不能把 M 当作普通小角度插值。
        np.testing.assert_allclose(values[:, 5], hours * .004, atol=1e-12)
        np.testing.assert_allclose(values[:, 3] - values[-1, 3], hours * .003, atol=1e-12)
        np.testing.assert_allclose(values[:, 4] - values[-1, 4], hours * .002, atol=1e-12)
        restored = restore_parameters(inputs[-1], 0, known[-1])
        np.testing.assert_allclose(restored, latest, atol=1e-12)

    def test_linear_extrapolation_and_fractional_target_epoch(self):
        records, latest, rate = self.linear_history()
        origin = records[-1]["available"] + 37  # 非整小时，检验最后一步插值。
        future = origin + np.arange(1, 61) * 60
        class LinearEvaluator:
            def predict_batch(self, inputs, horizon, **kwargs):
                return [SimpleNamespace(forecast=x[-1] + np.arange(1, horizon + 1)
                        * (float(x[-1]) - float(x[0])) / (len(x) - 1)) for x in inputs]
        result, audit = predict_elements(origin, future, LinearEvaluator(), records)
        target = latest + (24 + 37 / 3600) * rate
        target[3:] %= TAU
        expected = positions(make_record(target, origin, records[-1]), future)[:, :3]
        np.testing.assert_allclose(audit["parameters"]["elements_linear"], target, atol=1e-12)
        np.testing.assert_allclose(result["elements_linear"], expected, atol=.001, rtol=0)
        np.testing.assert_allclose(result["elements_timesfm"], expected, atol=.1, rtol=0)

    def test_future_tle_contents_never_reach_forecaster(self):
        records = load_tles()
        class ConstantEvaluator:
            def predict_batch(self, inputs, horizon, **kwargs):
                return [SimpleNamespace(forecast=np.full(horizon, x[-1])) for x in inputs]
        for number in range(1, 5):
            origin = origin_seconds(number)
            future = origin + np.arange(1, 61) * 60
            poisoned = [r if r["available"] <= origin else {**r, "satellite": None}
                        for r in records]
            original, audit = predict_elements(origin, future, ConstantEvaluator(), records)
            changed, _ = predict_elements(origin, future, ConstantEvaluator(), poisoned)
            self.assertEqual(len(audit["known"]), number + 1)
            self.assertTrue(all(r["available"] <= origin for r in audit["known"]))
            self.assertLessEqual(audit["grid"][-1], 0)
            for model in original:
                np.testing.assert_array_equal(original[model], changed[model])


class ArchivedElementTests(unittest.TestCase):
    def test_archived_inputs_elements_xyz_and_metrics(self):
        data = np.loadtxt(ROOT / "data/orbit.csv", delimiter=",", skiprows=1)
        for device in ("cpu", "gpu"):
            folder = ROOT / "reference/elements" / device
            all_rms = {}
            environment = json.loads((folder / "environment.json").read_text())
            self.assertEqual(environment["device"], device)
            for number in range(1, 5):
                output = folder / f"window-{number:02d}"
                sample = select_window(data, number)
                origin, future = sample[359, 0], sample[360:, 0]
                known, hours, raw, values, grid, inputs = prepare_history(origin)
                metadata = json.loads((output / "elements.json").read_text())
                self.assertEqual(metadata["known_tle_count"], number + 1)
                self.assertEqual(metadata["input_grid_points"], len(inputs))
                target_hour = (origin - known[-1]["epoch"]) / 3600
                self.assertAlmostEqual(metadata["element_extrapolation_hours"], target_hour)
                for filename, expected in [
                    ("element_tle_knots.csv", np.column_stack([hours, raw])),
                    ("element_input.csv", np.column_stack([grid, inputs]))]:
                    saved = np.loadtxt(output / filename, delimiter=",", skiprows=1)
                    np.testing.assert_array_equal(saved, expected)
                forecast = np.loadtxt(output / "element_forecast.csv", delimiter=",", skiprows=1)
                self.assertEqual(len(forecast), int(np.ceil(target_hour)) + 1)
                np.testing.assert_array_equal(forecast[0, 1:], inputs[-1])
                slow_parameters = np.array([np.interp(target_hour, forecast[:, 0], y)
                                            for y in forecast[:, 1:].T])
                expected_parameters = restore_parameters(slow_parameters, target_hour, known[-1])
                stored_parameters = metadata["parameters_at_origin"]["elements_timesfm"]
                np.testing.assert_allclose([stored_parameters[n] for n in NAMES], expected_parameters,
                                           atol=1e-12, rtol=0)

                predictions = np.genfromtxt(output / "prediction.csv", delimiter=",", names=True)
                np.testing.assert_array_equal(predictions["time_seconds"], future)
                reference = np.column_stack([predictions[f"reference_{c}_m"] for c in "xyz"])
                np.testing.assert_array_equal(reference, sample[360:, 1:4])
                with (output / "metrics.csv").open(newline="") as stream:
                    metrics = list(csv.DictReader(stream))
                models = {"sgp4_latest", "elements_linear", "elements_timesfm"}
                self.assertEqual({r["model"] for r in metrics}, models)
                expected_columns = {"time_seconds", *[f"reference_{c}_m" for c in "xyz"]}
                for model in models:
                    expected_columns.update([f"{model}_{c}_m" for c in "xyz"] + [f"{model}_deviation_m"])
                self.assertEqual(set(predictions.dtype.names), expected_columns)
                run = json.loads((output / "run.json").read_text())
                self.assertEqual(set(run["models"]), models)
                past = np.genfromtxt(output / "history.csv", delimiter=",", names=True)
                self.assertEqual(past.dtype.names, ("time_seconds", "sgp4_x_m", "sgp4_y_m", "sgp4_z_m"))
                np.testing.assert_array_equal(past["time_seconds"], sample[:360, 0])
                for axis, column in zip("xyz", (1, 2, 3)):
                    np.testing.assert_array_equal(past[f"sgp4_{axis}_m"], sample[:360, column])
                for row in metrics:
                    model = row["model"]
                    xyz = np.column_stack([predictions[f"{model}_{c}_m"] for c in "xyz"])
                    errors = np.linalg.norm(xyz - reference, axis=1)
                    rms = np.sqrt(np.mean(errors**2))
                    self.assertAlmostEqual(float(row["rms_3d_m"]), rms, places=6)
                    self.assertAlmostEqual(float(row["endpoint_3d_m"]), errors[-1], places=6)
                    np.testing.assert_allclose(predictions[f"{model}_deviation_m"], errors, atol=1e-8)
                    all_rms.setdefault(model, []).append(rms)
                    if model.startswith("elements_"):
                        p = [metadata["parameters_at_origin"][model][n] for n in NAMES]
                        rebuilt = positions(make_record(p, origin, known[-1]), future)[:, :3]
                        np.testing.assert_allclose(xyz, rebuilt, atol=.001, rtol=0)
                with (folder / "by_window.csv").open(newline="") as stream:
                    reported = [r for r in csv.DictReader(stream) if int(r["window"]) == number]
                self.assertEqual([r["rms_3d_m"] for r in reported], [r["rms_3d_m"] for r in metrics])
            with (folder / "summary.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    pooled = np.sqrt(np.mean(np.array(all_rms[row["model"]])**2))
                    self.assertAlmostEqual(float(row["pooled_rms_3d_m"]), pooled, places=6)


if __name__ == "__main__":
    unittest.main()
