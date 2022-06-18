from typing import Any, Dict

import yaml

from configs.config_schema import Config


class ConfigLoader:
    @classmethod
    def load(cls, file_path: str):
        config_dict = cls._load_dict_from_yaml(file_path)
        return Config(config_dict)

    @classmethod
    def _load_dict_from_yaml(cls, file_path: str) -> Dict[str, Any]:
        with open(file_path) as config_file:
            return yaml.full_load(config_file)
