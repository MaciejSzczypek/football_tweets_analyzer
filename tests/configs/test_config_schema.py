import unittest
from unittest.mock import patch

from configs.config_schema import Config, HyperParametersConfig, ConfigError, TextTokenizationConfig, \
    TextNormalizationConfig


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.mock_config_dict = {
            "data_transformations": {
                "test_config": {
                    "corpus": {
                        "normalization": {
                            "operations": {
                                "lower": True,
                                "remove_commas": True,
                                "remove_stopwords": True,
                                "apply_stemming": True,
                                "apply_lemmatization": True,
                                "expand_contractions": True,
                                "remove_special_characters": True,
                                "remove_emoticons": True,
                                "remove_hashtags": True,
                                "remove_links": True,
                                "remove_user_mentions": True
                            },
                            "custom_data": {
                                "stopwords": ["a", "the", "in"]
                            }
                        },
                        "tokenization": {
                            "sentence_tokenizer": "simple_sentence_tokenizer",
                            "word_tokenizer": "simple_word_tokenizer"
                        }
                    }
                }
            },
            "paths": {
                "results_dir": "/path/to/results",
                "wordcloud_mask": "/path/to/mask.png",
                "standford_model_class": "edu.stanford.nlp.Class",
                "standford_model_jar": "/path/to/model.jar"
            },
            "sections": {
                "basic_facts": {"enabled": True},
                "topic_modeling": {"enabled": True},
                "top_n_grams": {"enabled": False},
                "sentiment_analysis": {"enabled": True},
                "wordcloud": {"enabled": True},
                "summarization": {"enabled": False},
                "time_frames": {"enabled": False}
            }
        }

    @patch('corpus.tokenizing.tokenizers_provider.TokenizersProvider.get_sentence_tokenizer')
    @patch('corpus.tokenizing.tokenizers_provider.TokenizersProvider.get_word_tokenizer')
    def test_config_initialization(self, mock_get_word_tokenizer, mock_get_sentence_tokenizer):
        mock_get_sentence_tokenizer.return_value = "mock_sentence_tokenizer"
        mock_get_word_tokenizer.return_value = "mock_word_tokenizer"

        config = Config(self.mock_config_dict)

        self.assertTrue(isinstance(config.settings["test_config"], HyperParametersConfig))
        corpus_config = config.settings["test_config"].corpus
        self.assertTrue(isinstance(corpus_config.normalization, TextNormalizationConfig))
        self.assertTrue(isinstance(corpus_config.tokenization, TextTokenizationConfig))
        self.assertEqual(corpus_config.tokenization.sentence_tokenizer, "mock_sentence_tokenizer")
        self.assertEqual(corpus_config.tokenization.word_tokenizer, "mock_word_tokenizer")

    def test_paths_config(self):
        config = Config(self.mock_config_dict)
        paths = config.paths

        self.assertEqual(paths.results_dir, "/path/to/results")
        self.assertEqual(paths.wordcloud_mask, "/path/to/mask.png")
        self.assertEqual(paths.standford_model_class, "edu.stanford.nlp.Class")
        self.assertEqual(paths.standford_model_jar, "/path/to/model.jar")

    def test_sections_config(self):
        config = Config(self.mock_config_dict)
        sections = config.sections

        self.assertTrue(sections.is_basic_facts_enabled)
        self.assertTrue(sections.is_topic_modeling_enabled)
        self.assertFalse(sections.is_top_n_grams_enabled)
        self.assertTrue(sections.is_sentiment_analysis_enabled)
        self.assertTrue(sections.is_wordcloud_enabled)
        self.assertFalse(sections.is_summarization_enabled)
        self.assertFalse(sections.is_time_frames_enabled)


    def test_missing_sections_config(self):
        incomplete_config_dict = {
            "data_transformations": {
                "test_config": {
                    "corpus": {
                        "normalization": {
                            "operations": {
                                "lower": True,
                                "remove_commas": True,
                                "remove_stopwords": True,
                                "apply_stemming": True,
                                "apply_lemmatization": True,
                                "expand_contractions": True,
                                "remove_special_characters": True,
                                "remove_emoticons": True,
                                "remove_hashtags": True,
                                "remove_links": True,
                                "remove_user_mentions": True
                            },
                            "custom_data": {
                                "stopwords": ["a", "the", "in"]
                            }
                        },
                        "tokenization": {
                            "sentence_tokenizer": "simple_sentence_tokenizer",
                            "word_tokenizer": "simple_word_tokenizer"
                        }
                    }
                }
            },
            "paths": {
                "results_dir": "/path/to/results",
                "wordcloud_mask": "/path/to/mask.png",
                "standford_model_class": "edu.stanford.nlp.Class",
                "standford_model_jar": "/path/to/model.jar"
            }
        }
        config = Config(incomplete_config_dict)
        with self.assertRaises(ConfigError):
            config.sections



if __name__ == '__main__':
    unittest.main()
