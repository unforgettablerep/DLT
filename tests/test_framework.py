import json
import os
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run import parse_args  # noqa: E402
from src.configs import dlt_profiles  # noqa: E402
from src.configs.baseline_profiles import to_cli_string  # noqa: E402
from src.configs.model_parser_dict import canonical_model  # noqa: E402
from src.utils import result_io  # noqa: E402
from src.utils.checkpoint import ckpt_prefix  # noqa: E402


class ProfileTests(unittest.TestCase):
    def test_default_profile_is_the_paper_recipe(self):
        cfg = dlt_profiles.resolve("ETTh1", 96)
        self.assertEqual({k: cfg[k] for k in ("dlt_variant", "dlt_k", "dlt_L", "dlt_R", "dlt_impl")}, {"dlt_variant": "dltdyn", "dlt_k": 8, "dlt_L": 48, "dlt_R": 4, "dlt_impl": "basis"})
        self.assertEqual((cfg["learning_rate"], cfg["batch_size"], cfg["train_epochs"], cfg["patience"]), (0.003, 32, 10, 3))
        self.assertNotIn("drop_last", cfg)
        pems = dlt_profiles.resolve("PEMS08", 12)
        self.assertTrue(pems["no_time_marks"] and pems["drop_last"] and pems["test_batch_size"] == 1 and pems["pems_split"] == "itransformer")

    def test_efficiency_profile_holds_the_selected_recipes(self):
        rows = result_io.latest(result_io.read_results(REPO / "output" / "efficiency" / "eff_tune.tsv"))
        for ds, H in dlt_profiles.EFFICIENCY_SETTING.items():
            cand = [r for r in rows if r["dataset"] == ds and r["pred_len"] == H]
            best = min(cand, key=lambda r: r["best_vali"])
            cfg = dlt_profiles.resolve(ds, H, "efficiency")
            self.assertEqual((cfg["dlt_k"], cfg["dlt_L"], cfg["dlt_R"]), (best["dlt_k"], best["dlt_L"], best["dlt_R"]), ds)
            self.assertEqual(cfg["batch_size"], 16)

    def test_cli_flags_parse(self):
        for ds in dlt_profiles.list_datasets():
            for H in dlt_profiles.get_pred_lens(ds):
                args = parse_args(["--model", "DLT", "--des", "t"] + dlt_profiles.to_cli_args(ds, H, with_meta=True))
                self.assertEqual(result_io.dataset_of(args), ds)
                self.assertEqual(args.pred_len, H)
                base = parse_args(["--model", "iTransformer", "--des", "t"] + shlex.split(to_cli_string("iTransformer", ds, H, with_meta=True)))
                self.assertEqual(base.label_len, 48)
                self.assertTrue(base.drop_last)

    def test_model_aliases(self):
        self.assertEqual(canonical_model("SMamba"), "S_Mamba")
        args = parse_args(["--model", "DLTPlug", "--plug_base", "SMamba", "--d_state", "4"])
        self.assertEqual((args.model, args.plug_base, args.d_state), ("DLTPlug", "S_Mamba", 4))
        self.assertEqual(parse_args(["--model", "DLT", "--fix_seed", "2022"]).seed, 2022)

    def test_no_epoch_test_by_default(self):
        # every recorded run skipped the per-epoch test pass, which draws from the RNG and so changes the run
        self.assertTrue(parse_args(["--model", "DLT"]).no_epoch_test)
        self.assertTrue(parse_args(["--model", "DLT", "--no_epoch_test"]).no_epoch_test)
        self.assertFalse(parse_args(["--model", "DLT", "--epoch_test"]).no_epoch_test)


class ResultStoreTests(unittest.TestCase):
    def args(self, **kw):
        a = dict(model="DLT", data="ETTh1", data_path="ETTh1.csv", seq_len=96, label_len=48, pred_len=96, des="t", seed=2021, batch_size=32, test_batch_size=0, no_time_marks=False, dlt_variant="dltdyn", dlt_gate="none", dlt_hp_trial=-1)
        a.update(kw)
        return types.SimpleNamespace(**a)

    def test_round_trip_and_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.tsv")
            os.environ["RESULT_FILE"] = path
            try:
                result_io.write_test_result(self.args(), {"mse": 0.25, "mae": 0.5, "mape": None}, {"epoch_secs": [1.5, 2.0], "best_vali": 0.3})
                result_io.write_test_result(self.args(des="u"), {"mse": float("nan"), "mae": 0.5}, {"new_column": "x"})
            finally:
                del os.environ["RESULT_FILE"]
            rows = result_io.read_results(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual((rows[0]["mse"], rows[0]["epoch_secs"], rows[0]["status"], rows[0]["time_marks"]), (0.25, [1.5, 2.0], "ok", True))
            self.assertIsNone(rows[0]["mape"])
            self.assertEqual((rows[1]["status"], rows[1]["new_column"]), ("diverged", "x"))
            self.assertIsNone(rows[0]["new_column"])
            self.assertIsNone(rows[0]["d_model"])

    def test_shell_resume_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.tsv")
            result_io.write_rows(path, [{"des": "a", "seed": 2021, "mse": 0.1, "status": "ok"}, {"des": "b", "seed": 2021, "mse": float("nan"), "status": "diverged"}])
            check = "source {}; result_done {} des=$1 seed=2021".format(REPO / "scripts" / "utils" / "result_common.sh", path)
            self.assertEqual(subprocess.run(["bash", "-c", check, "_", "a"]).returncode, 0)
            self.assertNotEqual(subprocess.run(["bash", "-c", check, "_", "b"]).returncode, 0)

    def test_legacy_record_migration(self):
        rec = {"setting": "long_term_forecast_x_PatchTST_custom_ftM_sl96_ll48_pl96_dm128_nh16_el3_dl1_df256_expand2_dc4_fc3_ebtimeF_dtTrue_t_0", "model": "DLTPlug", "plug_base": "SMamba", "data": "custom", "data_path": "weather.csv", "lr": 0.001, "variant": "dltdyn", "k": 8, "fusion": "output", "mse": 0.2, "mae": 0.3, "seed": 2021, "des": "x"}
        row = result_io.migrate_record(rec)
        self.assertEqual((row["dataset"], row["plug_base"], row["learning_rate"], row["dlt_variant"], row["dlt_k"], row["dlt_fusion"]), ("Weather", "S_Mamba", 0.001, "dltdyn", 8, "output"))
        self.assertEqual((row["label_len"], row["d_layers"], row["factor"], row["status"]), (48, 1, 3, "ok"))
        self.assertNotIn("setting", row)
        self.assertEqual(result_io.migrate_record({"dataset": "Traffic", "model": "DLT", "pred_len": 96, "mse": 0.5, "mae": 0.3})["dataset"], "Traffic")

    def test_checkpoint_prefix_keeps_runs_apart(self):
        a = self.args()
        self.assertNotEqual(ckpt_prefix(a), ckpt_prefix(self.args(seed=2022)))
        self.assertNotEqual(ckpt_prefix(a), ckpt_prefix(self.args(des="v")))


class GeneratedScriptTests(unittest.TestCase):
    def test_per_dataset_scripts_are_current(self):
        res = subprocess.run([sys.executable, str(REPO / "scripts" / "multivariate_forecasting" / "make_scripts.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

    def test_registry_matches_the_p0_grid(self):
        from src.ablation.registry import P0_CONFIGS, P0_LABELS, p0_args

        self.assertEqual(len(P0_CONFIGS), 13)
        self.assertEqual(set(P0_LABELS), set(P0_CONFIGS))
        self.assertEqual(p0_args("dltdyn", "static")["dlt_impl"], "basis")
        self.assertEqual(p0_args("dlt", "none")["dlt_impl"], "kernel")


class MigrationTests(unittest.TestCase):
    def test_legacy_record_is_recognised_in_its_store(self):
        sys.path.insert(0, str(REPO / "scripts" / "utils"))
        import migrate_work

        rec = {"model": "DLT", "model_id": "long_small_Weather_96", "data": "custom", "data_path": "weather.csv", "seq_len": 96, "pred_len": 96, "seed": 2021, "mse": 0.1641, "mae": 0.2117, "variant": "dltdyn", "des": "Exp"}
        row = result_io.migrate_record(rec)
        stored = dict(row, des="Weather_96_dltdyn", dlt_variant="dltdyn")
        self.assertEqual(migrate_work.key(row), migrate_work.key(stored))
        self.assertNotEqual(migrate_work.key(row), migrate_work.key(dict(stored, seed=2022)))
        self.assertEqual((migrate_work.category("long_small"), migrate_work.category("hp_pems"), migrate_work.category("plugin_all")), ("long_term_forecast", "hyperparam", "plugin"))
        self.assertIsNone(migrate_work.category("quickstart"))


class ResumeTests(unittest.TestCase):
    def test_missing_jobs(self):
        sys.path.insert(0, str(REPO / "scripts" / "utils"))
        import run_missing

        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "s.tsv")
            result_io.write_rows(store, [{"des": "done", "seed": 2021, "mse": 0.1, "status": "ok"}])
            jobs = os.path.join(tmp, "jobs.txt")
            lines = ["python -u scripts/run.py --model DLT --des {} --seed 2021 --result_file {} > x.log 2>&1".format(d, store) for d in ("done", "todo")]
            Path(jobs).write_text("\n".join(lines) + "\n")
            self.assertEqual(run_missing.missing(jobs), [lines[1]])


if __name__ == "__main__":
    unittest.main()
