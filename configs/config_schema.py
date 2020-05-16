from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CustomDataForTextNormalizationConfig:
    stopwords: List[str]


@dataclass
class OperationsForTextNormalizationConfig:
    remove_stopwords: bool
    apply_stemming: bool
    apply_lemmatization: bool
    expand_contractions: bool
    remove_special_characters: bool
    remove_emoticons: bool


@dataclass
class TextNormalizationConfig:
    operations: OperationsForTextNormalizationConfig
    custom_data: CustomDataForTextNormalizationConfig

    def __init__(
        self, operations: Dict[str, Any], custom_data: Dict[str, Any],
    ):
        self.operations = OperationsForTextNormalizationConfig(**operations)
        self.custom_data = CustomDataForTextNormalizationConfig(**custom_data)


@dataclass
class CorpusConfig:
    text_normalization: TextNormalizationConfig

    def __init__(self, text_normalization: Dict[str, Any]):
        self.text_normalization = TextNormalizationConfig(**text_normalization)


@dataclass
class HyperparametersConfig:
    corpus: CorpusConfig

    def __init__(self, corpus: Dict[str, Any]):
        self.corpus = CorpusConfig(**corpus)


class Config:
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self._settings = self._create_settings_from_dict(config_dict)

    @classmethod
    def _create_settings_from_dict(cls, config_dict: Dict[str, Any]) -> List[HyperparametersConfig]:
        return [HyperparametersConfig(**setting) for setting_name, setting in config_dict.items()]

    @property
    def settings(self) -> List[HyperparametersConfig]:
        return self._settings
