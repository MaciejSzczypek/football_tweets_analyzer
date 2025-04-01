import unittest
import pandas as pd
from data.column_names import TEXT_COLUMN_NAME, USER_NAME_COLUMN_NAME
from corpus.cleaning.duplicated_tweets_remover import DuplicatedTweetsRemover  # Adjust import to your actual module


class TestDuplicatedTweetsRemover(unittest.TestCase):

    def setUp(self):
        data = {
            TEXT_COLUMN_NAME: [
                "Hello world",
                "Hello world",
                "Check this out",
                "Check this out",
                "RT @user: Check this out",
                "Just another tweet",
                "Hello world",
                "RT @user2: Hello world"
            ],
            USER_NAME_COLUMN_NAME: [
                "user1", "user1", "user2", "user2", "user3", "user1", "user1", "user2"
            ]
        }
        self.df = pd.DataFrame(data)

    def test_remove_duplicated_tweets(self):
        df = DuplicatedTweetsRemover.remove_duplicated_tweets(self.df)

        self.assertEqual(df.shape[0], 5)
        self.assertEqual(1,  list(df[TEXT_COLUMN_NAME].values).count("Check this out"))
        self.assertEqual(1,  list(df[TEXT_COLUMN_NAME].values).count("Hello world"))
        self.assertIn("RT @user: Check this out", df[TEXT_COLUMN_NAME].values)

    def test_no_duplicates_in_dataframe(self):
        unique_data = {
            TEXT_COLUMN_NAME: ["Tweet 1", "Tweet 2", "Tweet 3"],
            USER_NAME_COLUMN_NAME: ["user1", "user2", "user3"]
        }
        df_unique = pd.DataFrame(unique_data)

        df = DuplicatedTweetsRemover.remove_duplicated_tweets(df_unique)

        self.assertEqual(df.shape, df_unique.shape)
        self.assertTrue(df.equals(df_unique))
