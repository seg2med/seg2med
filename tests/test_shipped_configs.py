"""The YAML files shipped under configs/ stay loadable as the code changes."""

import unittest
from pathlib import Path

import yaml

from seg2med_train.dataloader_registry import available_dataset_names
from seg2med_train.model_selection import available_model_names
from seg2med_train.utils.my_configs_yacs import init_cfg


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").rglob("*.yaml"))


class ShippedConfigTests(unittest.TestCase):
    def test_configs_exist(self):
        self.assertTrue(CONFIGS)

    def test_every_config_is_valid_yaml_with_forward_slashes(self):
        for path in CONFIGS:
            with self.subTest(config=path.relative_to(ROOT).as_posix()):
                text = path.read_text()
                self.assertIsInstance(yaml.safe_load(text), dict)
                for line in text.splitlines():
                    if "regex" not in line:
                        self.assertNotIn("\\", line)

    def test_training_configs_load_and_name_registered_components(self):
        models, datasets = set(available_model_names()), set(available_dataset_names())
        training = [p for p in CONFIGS if "model_name" in yaml.safe_load(p.read_text())]
        self.assertTrue(training)
        for path in training:
            with self.subTest(config=path.relative_to(ROOT).as_posix()):
                cfg = init_cfg(str(path))  # rejects keys the defaults no longer know
                self.assertIn(cfg.model_name, models)
                self.assertIn(cfg.dataset_name, datasets)


if __name__ == "__main__":
    unittest.main()
