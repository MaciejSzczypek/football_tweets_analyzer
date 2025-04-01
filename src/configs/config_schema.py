from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Union
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


@dataclass
class PathsConfig:
    results_dir: str
    wordcloud_mask: str
    standford_model_class: str
    standford_model_jar: str


@dataclass
class BasicFactsConfig:
    enabled: bool


@dataclass
class TopicModelingConfig:
    enabled: bool


@dataclass
class TopNGramsConfig:
    enabled: bool


@dataclass
class SentimentAnalysisConfig:
    enabled: bool


@dataclass
class WordCloudConfig:
    enabled: bool


@dataclass
class SummarizationConfig:
    enabled: bool

@dataclass
class TimeFramesConfig:
    enabled: bool

@dataclass
class SectionsConfig:
    basic_facts: BasicFactsConfig
    topic_modeling: TopicModelingConfig
    top_n_grams: TopNGramsConfig
    sentiment_analysis: SentimentAnalysisConfig
    wordcloud: WordCloudConfig
    summarization: SummarizationConfig
    time_frames: TimeFramesConfig

    def __init__(
            self,
            basic_facts: Dict[str, Any],
            topic_modeling: Dict[str, Any],
            top_n_grams: Dict[str, Any],
            sentiment_analysis: Dict[str, Any],
            wordcloud: Dict[str, Any],
            summarization: Dict[str, Any],
            time_frames: Dict[str, Any]
    ):
        self.basic_facts = BasicFactsConfig(**basic_facts)
        self.topic_modeling = TopicModelingConfig(**topic_modeling)
        self.top_n_grams = TopNGramsConfig(**top_n_grams)
        self.sentiment_analysis = SentimentAnalysisConfig(**sentiment_analysis)
        self.wordcloud = WordCloudConfig(**wordcloud)
        self.summarization = SummarizationConfig(**summarization)
        self.time_frames = TimeFramesConfig(**time_frames)

    @property
    def is_basic_facts_enabled(self):
        return self.basic_facts.enabled

    @property
    def is_topic_modeling_enabled(self):
        return self.topic_modeling.enabled

    @property
    def is_top_n_grams_enabled(self):
        return self.top_n_grams.enabled

    @property
    def is_sentiment_analysis_enabled(self):
        return self.sentiment_analysis.enabled

    @property
    def is_wordcloud_enabled(self):
        return self.wordcloud.enabled

    @property
    def is_summarization_enabled(self):
        return self.summarization.enabled

    @property
    def is_time_frames_enabled(self):
        return self.time_frames.enabled

class ConfigError(AttributeError):
    def __init__(self, attribute):
        AttributeError(f"'{attribute}' config has not been set in yaml file!")


class Config:
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self._settings = self._create_settings_from_dict(config_dict)

    @classmethod
    def _create_settings_from_dict(
            cls, config_dict: Dict[str, Any]
    ) -> Dict[str, Union[HyperParametersConfig, PathsConfig, SectionsConfig]]:
        config = {}
        for setting_name, setting in config_dict.items():
            if setting_name == "data_transformations":
                for sub_setting_name, sub_setting in setting.items():
                    config[sub_setting_name] = HyperParametersConfig(**sub_setting)
            elif setting_name == "paths":
                config["paths"] = PathsConfig(**setting)
            elif setting_name == "sections":
                config["sections"] = SectionsConfig(**setting)
        return config

    @property
    def settings(self) -> Dict[str, Union[HyperParametersConfig, PathsConfig]]:
        return self._settings

    @property
    def paths(self) -> PathsConfig:
        try:
            return self._settings["paths"]
        except:
            raise ConfigError("paths")

    @property
    def sections(self) -> SectionsConfig:
        try:
            return self._settings["sections"]
        except:
            raise ConfigError("sections")
