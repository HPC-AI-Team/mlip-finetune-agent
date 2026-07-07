import os
import shutil
import subprocess
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "plugins" / "mlip-finetune-agent"
SETUP_SCRIPT = PLUGIN_ROOT / "scripts" / "setup_elora_env.sh"
TRAIN_SCRIPT = PLUGIN_ROOT / "scripts" / "run_elora_smoke_train.sh"
VENV_DIR = Path(os.environ.get("ELORA_VENV_DIR", ROOT / ".venv-elora"))
DEFAULT_CONDA_BIN = Path("/home/lmz/miniconda3/bin/conda")
DEFAULT_CONDA_ENV = "pytorch"
SOURCE_XYZ = Path(
    os.environ.get(
        "ELORA_SOURCE_XYZ",
        "/hdd/mlip-finetune/datasets/BOTNet-datasets/dataset_3BPA/train_300K.xyz",
    )
)
FOUNDATION_MODEL = Path(
    os.environ.get(
        "ELORA_FOUNDATION_MODEL",
        "/hdd/mlip-finetune/models/2024-01-07-mace-128-L2_epoch-199.model",
    )
)


class ELoRAGPUEndToEndTest(unittest.TestCase):
    """Installs/checks the ELoRA env and runs a tiny real-data GPU fine-tune."""

    @classmethod
    def setUpClass(cls):
        cls.run_id = os.environ.get(
            "ELORA_RUN_ID",
            time.strftime("elora-real-integration-%Y%m%d-%H%M%S"),
        )
        cls.model_name = os.environ.get("ELORA_MODEL_NAME", "elora_real_integration")
        cls.run_dir = ROOT / "mlip_runs" / cls.run_id
        cls.release_dir = ROOT / "mlip_releases" / cls.run_id
        cls.trained_model = cls.run_dir / "models" / f"{cls.model_name}.model"
        cls.predictions = cls.run_dir / "results" / "predictions.xyz"

        cls.e2e_env = cls.build_e2e_env()
        cls.assert_inputs_exist(cls.e2e_env)
        cls.setup_output = cls.run_command(["bash", str(SETUP_SCRIPT)], timeout=3600, env=cls.e2e_env)
        train_env = cls.e2e_env.copy()
        train_env["ELORA_RUN_ID"] = cls.run_id
        train_env["ELORA_MODEL_NAME"] = cls.model_name
        train_env["ELORA_SOURCE_XYZ"] = str(SOURCE_XYZ)
        train_env["ELORA_FOUNDATION_MODEL"] = str(FOUNDATION_MODEL)
        cls.train_output = cls.run_command(["bash", str(TRAIN_SCRIPT)], timeout=2400, env=train_env)

    @classmethod
    def build_e2e_env(cls):
        env = os.environ.copy()
        env.setdefault("ELORA_SETUP_MODE", "existing")
        env.setdefault("ELORA_SKIP_TORCH", "1")
        if "ELORA_PYTHON_BIN" not in env and "ELORA_CONDA_ENV" not in env:
            env["ELORA_CONDA_ENV"] = DEFAULT_CONDA_ENV
        if "CONDA_BIN" not in env and DEFAULT_CONDA_BIN.is_file():
            env["CONDA_BIN"] = str(DEFAULT_CONDA_BIN)
        return env

    @classmethod
    def assert_inputs_exist(cls, env):
        missing = [str(path) for path in (SETUP_SCRIPT, TRAIN_SCRIPT, SOURCE_XYZ, FOUNDATION_MODEL) if not path.exists()]
        if missing:
            raise AssertionError("Missing ELoRA E2E input(s): " + ", ".join(missing))
        if FOUNDATION_MODEL.stat().st_size < 10 * 1024 * 1024:
            raise AssertionError(
                f"{FOUNDATION_MODEL} is too small for a real MACE model; "
                "check for an HTML download or Git LFS pointer."
            )
        if shutil.which("uv") is None:
            raise AssertionError("uv is required for the ELoRA setup test")
        if "ELORA_CONDA_ENV" in env:
            conda_bin = env.get("CONDA_BIN", "conda")
            if not Path(conda_bin).is_file() and shutil.which(conda_bin) is None:
                raise AssertionError(f"conda is required for ELORA_CONDA_ENV={env['ELORA_CONDA_ENV']}")

    @classmethod
    def run_command(cls, command, timeout, env=None):
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        merged_env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            tail = "\n".join(output.splitlines()[-160:])
            raise AssertionError(
                f"Command failed ({result.returncode}): {' '.join(command)}\n{tail}"
            )
        return output

    def test_01_setup_validated_target_elora_environment(self):
        self.assertIn("ELoRA environment: ok", self.setup_output)
        self.assertIn("python:", self.setup_output)
        self.assertIn("prefix:", self.setup_output)
        self.assertIn("mace_run_train:", self.setup_output)
        self.assertIn("mace_eval_configs:", self.setup_output)
        self.assertIn("LoRA_weight", self.setup_output)
        self.assertIn("ELoRA_weights", self.setup_output)
        if self.e2e_env.get("ELORA_SETUP_MODE") == "fresh":
            self.assertTrue((VENV_DIR / "bin" / "python").is_file())
            self.assertTrue((VENV_DIR / "bin" / "mace_run_train").is_file())
            self.assertTrue((VENV_DIR / "bin" / "mace_eval_configs").is_file())

    def test_02_data_prep_used_real_xyz(self):
        train_xyz = self.run_dir / "dataset" / "train.xyz"
        valid_xyz = self.run_dir / "dataset" / "valid.xyz"
        test_xyz = self.run_dir / "dataset" / "test.xyz"
        self.assertTrue(train_xyz.is_file())
        self.assertTrue(valid_xyz.is_file())
        self.assertTrue(test_xyz.is_file())
        text = train_xyz.read_text(encoding="utf-8")
        self.assertIn("REF_energy", text)
        self.assertIn("REF_forces", text)

    def test_03_elora_fine_tune_wrote_model_and_logs(self):
        self.assertTrue(self.trained_model.is_file())
        self.assertGreater(self.trained_model.stat().st_size, 10 * 1024 * 1024)
        logs = list((self.run_dir / "logs").glob(f"{self.model_name}_run-*.log"))
        self.assertTrue(logs)
        log_text = logs[0].read_text(encoding="utf-8")
        self.assertIn("CUDA", log_text)
        self.assertIn("Training complete", log_text)
        self.assertIn("Error-table on TEST", log_text)
        train_command = (self.run_dir / "train-command.sh").read_text(encoding="utf-8")
        self.assertIn("mace_run_train", train_command)
        self.assertIn("ELoRA", (self.run_dir / "config.yaml").read_text(encoding="utf-8"))

    def test_04_elora_evaluate_wrote_predictions(self):
        self.assertTrue(self.predictions.is_file())
        predictions = self.predictions.read_text(encoding="utf-8")
        self.assertIn("MACE_energy", predictions)
        self.assertIn("MACE_forces", predictions)

    def test_05_publish_artifacts_exist(self):
        required = {
            f"{self.model_name}.model",
            "config.yaml",
            "train-command.sh",
            "dataset-manifest.md",
            "evaluation.md",
            "release-manifest.md",
        }
        self.assertTrue(self.release_dir.is_dir())
        self.assertEqual(required, {path.name for path in self.release_dir.iterdir()})
        manifest = (self.release_dir / "release-manifest.md").read_text(encoding="utf-8")
        self.assertIn("ELoRA@main", manifest)
        self.assertIn("ELoRA@MACE_ELoRA", manifest)
        self.assertIn("LoRA rank/alpha", manifest)


if __name__ == "__main__":
    unittest.main()
