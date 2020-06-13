from dataclasses import dataclass
from typing import Any, Dict, List, Callable
from corpus.tokenizing.tokenizers_provider import TokenizersProvider


@dataclass
class CustomDataForTextNormalizationConfig:
    stopwords: List[str]


@dataclass
class OperationsForTextNormalizationConfig:
    lower: bool
    remove_commas: bool
    remove_stopwords: bool
    apply_stemming: bool
    apply_lemmatization: bool
    expand_contractions: bool
    remove_special_characters: bool
    remove_emoticons: bool
    remove_hashtags: bool
    remove_links: bool
    remove_user_mentions: bool


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
class TextTokenizationConfig:
    sentence_tokenizer: Callable
    word_tokenizer: Callable

    def __init__(
        self, sentence_tokenizer: str, word_tokenizer: str,
    ):
        self.sentence_tokenizer = TokenizersProvider.get_sentence_tokenizer(
            sentence_tokenizer
        )
        self.word_tokenizer = TokenizersProvider.get_word_tokenizer(word_tokenizer)


@dataclass
class CorpusConfig:
    normalization: TextNormalizationConfig
    tokenization: TextTokenizationConfig

    def __init__(self, normalization: Dict[str, Any], tokenization: Dict[str, Any]):
        self.normalization = TextNormalizationConfig(**normalization)
        self.tokenization = TextTokenizationConfig(**tokenization)


@dataclass
class HyperParametersConfig:
    corpus: CorpusConfig

    def __init__(self, corpus: Dict[str, Any]):
        self.corpus = CorpusConfig(**corpus)


class Config:
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self._settings = self._create_settings_from_dict(config_dict)

    @classmethod
    def _create_settings_from_dict(
        cls, config_dict: Dict[str, Any]
    ) -> Dict[str, HyperParametersConfig]:
        return {
            setting_name: HyperParametersConfig(**setting)
            for setting_name, setting in config_dict.items()
        }

    @property
    def settings(self) -> Dict[str, HyperParametersConfig]:
        return self._settings
