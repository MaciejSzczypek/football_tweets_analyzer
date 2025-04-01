
import unittest
import pandas as pd
from data.filters import TweetsFilterer
from data.column_names import TEXT_COLUMN_NAME

class TestTweetsFilterer(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            TEXT_COLUMN_NAME: [
                "This is a test tweet in English.",
                "RT @user: This is a retweet.",
                "Ceci est un tweet en français.",
                "Este es un tweet en español.",
                "😊😊😊😊😊",
            ],
            "language": ["en", "en", "fr", "es", None]
        })

    def test_filter_out_tweets_with_invalid_language_text(self):
        filtered_df = TweetsFilterer.filter_out_tweets_with_invalid_language_text(self.df)
        self.assertEqual(len(filtered_df), 3)
        self.assertTrue(all(filtered_df[TEXT_COLUMN_NAME].str.contains("test tweet in English|RT @user: This is a retweet.|😊😊😊😊😊", regex=True)))

        filtered_df_lang = TweetsFilterer.filter_out_tweets_with_invalid_language_text(self.df, "language")
        self.assertEqual(len(filtered_df_lang), 2)
        self.assertTrue(all(filtered_df_lang["language"] == "en"))

    def test_filter_out_retweets(self):
        filtered_df = TweetsFilterer.filter_out_retweets(self.df)
        self.assertEqual(len(filtered_df), 4)
        self.assertFalse(any(filtered_df[TEXT_COLUMN_NAME].str.contains("RT @")))

    def test_tweet_contains_valid_text(self):
        valid_tweet = pd.Series({TEXT_COLUMN_NAME: "This is a test tweet in English."})
        invalid_tweet = pd.Series({TEXT_COLUMN_NAME: "Ceci est un tweet en français."})
        emoji_tweet = pd.Series({TEXT_COLUMN_NAME: "😊😊😊😊😊"})

        self.assertTrue(TweetsFilterer._tweet_contains_valid_text(valid_tweet))
        self.assertFalse(TweetsFilterer._tweet_contains_valid_text(invalid_tweet))
        self.assertTrue(TweetsFilterer._tweet_contains_valid_text(emoji_tweet))

    def test_text_is_written_in_english(self):
        self.assertTrue(TweetsFilterer._text_is_written_in_english("This is a test tweet in English."))
        self.assertFalse(TweetsFilterer._text_is_written_in_english("Ceci est un tweet en français."))

    def test_text_contains_emojis(self):
        self.assertTrue(TweetsFilterer._text_contains_emojis("😊😊😊😊😊"))
        self.assertFalse(TweetsFilterer._text_contains_emojis("This is a test tweet in English."))

    def test_tweet_is_a_retweet(self):
        retweet = pd.Series({TEXT_COLUMN_NAME: "RT @user: This is a retweet."})
        normal_tweet = pd.Series({TEXT_COLUMN_NAME: "This is a test tweet in English."})

        self.assertTrue(TweetsFilterer._tweet_is_a_retweet(retweet))
        self.assertFalse(TweetsFilterer._tweet_is_a_retweet(normal_tweet))

    def test_get_text_from_tweet(self):
        tweet = pd.Series({TEXT_COLUMN_NAME: "This is a test tweet in English."})
        self.assertEqual(TweetsFilterer._get_text_from_tweet(tweet), "This is a test tweet in English.")

