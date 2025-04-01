import unittest
import pandas as pd
from unittest.mock import MagicMock
from sklearn.feature_extraction.text import TfidfVectorizer
from data.dataset_manager import DataSet, HyperParametersConfig


class TestDataSet(unittest.TestCase):

    def setUp(self):
        self.data = {
            'text': [
                "Hello world! 😊",
                "This is a test tweet. 😃",
                "Another tweet for testing. 😎"
            ],
            'label': [1, 0, 1]
        }
        self.df = pd.DataFrame(self.data)
        self.hyper_params_config = MagicMock(spec=HyperParametersConfig)
        self.tfidf_vectorizer = TfidfVectorizer(token_pattern=r"\S+", stop_words="english", min_df=1)

    def test_create(self):
        dataset = DataSet.create(self.df, self.hyper_params_config, tfidf_vectorizer=self.tfidf_vectorizer)
        self.assertIsInstance(dataset, DataSet)
        self.assertEqual(dataset.initial_df.shape[0], len(self.df))
        self.assertEqual(len(dataset.normalized_tweets_as_token_lists), len(self.df))
        self.assertIsInstance(dataset.tfidf, type(self.tfidf_vectorizer.fit_transform(self.df['text'])))

    def test_normalized_tweets_as_demojized_strings(self):
        dataset = DataSet.create(self.df, self.hyper_params_config, tfidf_vectorizer=self.tfidf_vectorizer)
        demojized_strings = dataset.normalized_tweets_as_demojized_strings
        self.assertTrue(all("😊" not in s for s in demojized_strings))

    def test_initial_df_with_emojis_converted_to_text(self):
        dataset = DataSet.create(self.df, self.hyper_params_config, tfidf_vectorizer=self.tfidf_vectorizer)
        df_with_converted_emojis = dataset.initial_df_with_emojis_converted_to_text
        self.assertTrue(df_with_converted_emojis[TEXT_COLUMN_NAME].str.contains("😊").any())

    def test_prepare_for_nltk_classifier(self):
        dataset = DataSet.create(self.df, self.hyper_params_config, tfidf_vectorizer=self.tfidf_vectorizer)
        prepared_data = dataset.prepare_for_nltk_classifier(convert_emojis_to_text=True)
        self.assertIsInstance(prepared_data, list)
        self.assertTrue(all(isinstance(item, tuple) for item in prepared_data))
        self.assertTrue(all(isinstance(item[0], dict) for item in prepared_data))
        self.assertTrue(all(isinstance(item[1], float) for item in prepared_data))

    def test_get_aggregated_tweets(self):
        dataset = DataSet.create(self.df, self.hyper_params_config, tfidf_vectorizer=self.tfidf_vectorizer)
        aggregated_tweets = dataset._get_aggregated_tweets(aggregation_factor=2)
        self.assertTrue(len(aggregated_tweets) > 1)
        self.assertTrue(isinstance(aggregated_tweets[0], str))

    def test_tfidf_for_aggregated_tweets(self):
        dataset = DataSet.create(self.df, self.hyper_params_config, tfidf_vectorizer=self.tfidf_vectorizer)
        tfidf_matrix = dataset.tfidf_for_aggregated_tweets
        self.assertTrue(tfidf_matrix.shape[0] > 0)

    def test_original_tweets_array(self):
        original_df = self.df.copy()
        dataset = DataSet.create(self.df, self.hyper_params_config, tfidf_vectorizer=self.tfidf_vectorizer,
                                 original_df=original_df)
        original_tweets = dataset.original_tweets_array
        self.assertTrue(isinstance(original_tweets, list))
        self.assertEqual(len(original_tweets), len(self.df))

    def test_normalized_tweets_as_strings(self):
        dataset = DataSet.create(self.df, self.hyper_params_config, tfidf_vectorizer=self.tfidf_vectorizer)
        normalized_strings = dataset.normalized_tweets_as_strings
        self.assertEqual(len(normalized_strings), len(self.df))
        self.assertTrue(all(isinstance(item, str) for item in normalized_strings))

    def test_invalid_column_name(self):
        invalid_data = {
            'wrong_column': ["Hello", "Test"]
        }
        invalid_df = pd.DataFrame(invalid_data)
        with self.assertRaises(KeyError):
            DataSet.create(invalid_df, self.hyper_params_config, tfidf_vectorizer=self.tfidf_vectorizer)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=['text', 'label'])
        dataset = DataSet.create(empty_df, self.hyper_params_config, tfidf_vectorizer=self.tfidf_vectorizer)
        self.assertEqual(dataset.initial_df.shape[0], 0)
        self.assertEqual(len(dataset.normalized_tweets_as_token_lists), 0)


if __name__ == "__main__":
    unittest.main()
