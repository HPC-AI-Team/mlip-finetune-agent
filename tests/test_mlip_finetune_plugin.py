import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "plugins" / "mlip-finetune-agent"
MARKETPLACE = ROOT / ".agents" / "plugins" / "marketplace.json"

EXPECTED_SKILLS = {
    "setup",
    "data-prep",
    "fine-tune",
    "evaluate",
    "publish",
    "help",
}


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_frontmatter(path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise AssertionError(f"{path} does not start with YAML frontmatter")
    end = text.find("\n---", 4)
    if end == -1:
        raise AssertionError(f"{path} frontmatter is not closed")

    frontmatter = {}
    for raw_line in text[4:end].splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise AssertionError(f"{path} has malformed frontmatter line: {raw_line}")
        frontmatter[key.strip()] = value.strip().strip('"')
    return frontmatter, text[end + len("\n---") :]


def plugin_text_files():
    for path in PLUGIN_ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".py", ".sh"}:
            yield path


class MLIPFineTunePluginTest(unittest.TestCase):
    def test_manifest_declares_plugin_interface(self):
        manifest = read_json(PLUGIN_ROOT / ".codex-plugin" / "plugin.json")

        self.assertEqual(manifest["name"], "mlip-finetune-agent")
        self.assertRegex(manifest["version"], r"^\d+\.\d+\.\d+")
        self.assertEqual(manifest["skills"], "./skills/")
        self.assertEqual(manifest["interface"]["displayName"], "ELoRA MLIP Fine-Tune Agent")
        self.assertEqual(manifest["interface"]["category"], "Developer Tools")
        self.assertIn("Interactive", manifest["interface"]["capabilities"])
        self.assertIn("Write", manifest["interface"]["capabilities"])
        self.assertLessEqual(len(manifest["interface"]["defaultPrompt"]), 3)
        self.assertIn("MACE", manifest["description"])
        self.assertIn("ELoRA", manifest["description"])

    def test_marketplace_points_to_repo_local_plugin(self):
        marketplace = read_json(MARKETPLACE)
        entries = marketplace["plugins"]

        self.assertEqual(marketplace["name"], "mlip-finetune-agent")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "mlip-finetune-agent")
        self.assertEqual(entries[0]["source"], {
            "source": "local",
            "path": "./plugins/mlip-finetune-agent",
        })
        self.assertEqual(entries[0]["policy"]["installation"], "AVAILABLE")
        self.assertEqual(entries[0]["policy"]["authentication"], "ON_INSTALL")
        self.assertEqual(entries[0]["category"], "Developer Tools")

    def test_all_expected_skills_have_valid_frontmatter(self):
        skill_dirs = {
            path.name for path in (PLUGIN_ROOT / "skills").iterdir() if path.is_dir()
        }
        self.assertEqual(skill_dirs, EXPECTED_SKILLS)

        for skill in EXPECTED_SKILLS:
            with self.subTest(skill=skill):
                frontmatter, body = parse_frontmatter(
                    PLUGIN_ROOT / "skills" / skill / "SKILL.md"
                )
                self.assertRegex(frontmatter["name"], r"^MLIP-Finetune-")
                self.assertTrue(frontmatter["description"])
                self.assertRegex(frontmatter["version"], r"^\d+\.\d+\.\d+")
                self.assertTrue(body.strip())

    def test_skill_contracts_cover_each_workflow_stage(self):
        required_terms = {
            "setup": [
                "uv venv",
                ".venv-elora",
                "ELORA_SETUP_MODE",
                "ELORA_CONDA_ENV",
                "ELORA_PYTHON_BIN",
                "ELORA_SKIP_TORCH",
                "setup_elora_env.sh",
                "check_elora_env.py",
                "git+https://github.com/hyjwpk/ELoRA.git@main",
                "git+https://github.com/hyjwpk/ELoRA.git@MACE_ELoRA",
                "LoRA_weight",
                "ELoRA_weights",
                "/hdd/mlip-finetune/models",
                "/hdd/mlip-finetune/datasets",
            ],
            "data-prep": [
                "dataset-manifest.md",
                "config.yaml",
                "train.xyz",
                "valid.xyz",
                "test.xyz",
                "seed `123`",
            ],
            "fine-tune": [
                ".venv-elora/bin/mace_run_train",
                "train-command.sh",
                "organic",
                "inorganic",
                "MACE-OFF23_medium.model",
                "2024-01-07-mace-128-L2_epoch-199.model",
                "plain PyPI `mace-torch` is not an acceptable substitute",
            ],
            "evaluate": [
                "<target-prefix>/bin/mace_eval_configs",
                "Energy RMSE",
                "Force RMSE",
                "evaluation.md",
                "predictions.xyz",
            ],
            "publish": [
                "release-manifest.md",
                "mlip_releases/<release_name>/",
                "dataset-manifest.md",
                "evaluation.md",
                "train-command.sh",
            ],
            "help": [
                "/mlip-finetune:setup",
                "/mlip-finetune:data-prep",
                "/mlip-finetune:fine-tune",
                "/mlip-finetune:evaluate",
                "/mlip-finetune:publish",
            ],
        }

        for skill, terms in required_terms.items():
            body = (PLUGIN_ROOT / "skills" / skill / "SKILL.md").read_text(
                encoding="utf-8"
            )
            for term in terms:
                with self.subTest(skill=skill, term=term):
                    self.assertIn(term, body)

    def test_reference_templates_support_training_evaluation_and_release(self):
        references = PLUGIN_ROOT / "references"
        expected_files = {
            "datasets.md",
            "training-templates.md",
            "evaluation-report-template.md",
            "release-checklist.md",
        }
        self.assertEqual({path.name for path in references.iterdir()}, expected_files)

        datasets = (references / "datasets.md").read_text(encoding="utf-8")
        self.assertIn("3BPA", datasets)
        self.assertIn("AcAc", datasets)
        self.assertIn("/hdd/mlip-finetune/models", datasets)
        self.assertIn("/hdd/mlip-finetune/datasets", datasets)

        templates = (references / "training-templates.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(re.findall(r"\bmace_run_train\b", templates)), 2)
        self.assertIn(".venv-elora/bin/mace_run_train", templates)
        self.assertIn("ELoRA.git@main", templates)
        self.assertIn("ELoRA.git@MACE_ELoRA", templates)
        self.assertIn("lora_rank: 16", templates)
        self.assertIn("ScaleShiftMACE", templates)
        self.assertIn('mode: "organic"', templates)
        self.assertIn("MACE-OFF23_medium.model", templates)

        evaluation = (
            references / "evaluation-report-template.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Energy RMSE", evaluation)
        self.assertIn("Force RMSE", evaluation)

        release = (references / "release-checklist.md").read_text(encoding="utf-8")
        self.assertIn("Final `.model` file", release)
        self.assertIn("Foundation model filename", release)
        self.assertIn("ELoRA marker evidence", release)

    def test_elora_scripts_are_present(self):
        scripts = PLUGIN_ROOT / "scripts"
        expected = {
            "setup_elora_env.sh",
            "check_elora_env.py",
            "patch_elora_mace.py",
            "prepare_real_xyz_subset.py",
            "run_elora_smoke_train.sh",
        }
        self.assertEqual(expected, {path.name for path in scripts.iterdir() if path.is_file()})

        setup = (scripts / "setup_elora_env.sh").read_text(encoding="utf-8")
        self.assertIn("UV_BIN", setup)
        self.assertIn("ELORA_SETUP_MODE", setup)
        self.assertIn("ELORA_CONDA_ENV", setup)
        self.assertIn("ELORA_PYTHON_BIN", setup)
        self.assertIn("CONDA_PREFIX", setup)
        self.assertIn("ELORA_SKIP_TORCH", setup)
        self.assertIn("venv --python", setup)
        self.assertIn("ELoRA.git@main", setup)
        self.assertIn("ELoRA.git@MACE_ELoRA", setup)
        self.assertIn("patch_elora_mace.py", setup)

        check = (scripts / "check_elora_env.py").read_text(encoding="utf-8")
        self.assertIn("LoRA_weight", check)
        self.assertIn("ELoRA_weights", check)
        self.assertIn("expected-prefix", check)
        self.assertIn("sys.prefix", check)

        e2e = (scripts / "run_elora_smoke_train.sh").read_text(encoding="utf-8")
        self.assertIn("ELORA_CONDA_ENV", e2e)
        self.assertIn("CONDA_PREFIX", e2e)
        self.assertIn("mace_run_train", e2e)
        self.assertIn("--lora=True", e2e)

        patch = (scripts / "patch_elora_mace.py").read_text(encoding="utf-8")
        self.assertIn("name-matched foundation copy", patch)
        self.assertIn("LoRA", patch)

    def test_plain_mace_is_not_the_main_path(self):
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in plugin_text_files()
        )
        self.assertIn("Plain PyPI `mace-torch` is not the main path", combined)
        self.assertNotIn("conda --version", combined)

    def test_no_placeholder_text_remains(self):
        forbidden = ("[TODO", "PLACEHOLDER", "Local developer", "Mlip Finetune")
        for path in [*plugin_text_files(), MARKETPLACE]:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for term in forbidden:
                with self.subTest(path=path, term=term):
                    self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
