import os.path
import re
from typing import Dict, Tuple, List

from configs.config_schema import PathsConfig
from information_extraction.enums import Season, League
import pandas as pd
from emoji import demojize
from emoji.unicode_codes import UNICODE_EMOJI
from nltk.tag.stanford import StanfordNERTagger

from corpus.cleaning.constants import NORMALIZED_FOOTBALL_RESULT_WITH_TEAMS_FORMAT_REGEX, NORMALIZED_FOOTBALL_RESULT_FORMAT_REGEX
from corpus.tokenizing.custom_tokenizers import CustomTokenizer
from data.column_names import (
    CLUBS_KEY_COLUMN_NAME,
    CLUBS_NAME_COLUMN_NAME,
)
from data.loaders import DataLoader
from data.paths import ENGLISH_CLUBS_FILE_PATH, SPANISH_CLUBS_FILE_PATH
from scraping.scrappers import TeamsSquadsScrapper
from utils.printing import section_printing_decorator, new_line_appendix_decorator



class FactsExtractor:
    PERSON_TAG = "PERSON"
    QUESTION_WHO_PLAYED = "Which teams have played?"
    QUESTION_RESULT = "What was the result?"
    QUESTION_MOST_OFTEN_MENTIONED_PERSONS = (
        "Which persons were the most often mentioned?"
    )
    QUESTION_MOST_POPULAR_HASHTAGS = "What are the most popular hashtags?"
    QUESTION_MOST_POPULAR_EMOTICONS = "What are the most popular emoticons?"

    def __init__(
        self,
        corpus_without_emoticons: List[str],
        corpus_with_emoticons: List[str],
        n_latest_tweets_used_for_result_collection: int,
        season: Season,
        league: League,
        paths_config: PathsConfig
    ) -> None:
        self._corpus_without_emoticons = self._tokenize_tweets_corpus(corpus_without_emoticons)
        self._corpus_with_emoticons = corpus_with_emoticons
        self._premier_league_teams = DataLoader.from_csv(ENGLISH_CLUBS_FILE_PATH)
        self._primiera_division_teams = DataLoader.from_csv(SPANISH_CLUBS_FILE_PATH)
        self._premier_league_team_keys = set(self._premier_league_teams[CLUBS_KEY_COLUMN_NAME])
        self._primiera_division_team_keys = set(self._primiera_division_teams[CLUBS_KEY_COLUMN_NAME])
        self._tagger = StanfordNERTagger(paths_config.standford_model_class, paths_config.standford_model_jar)
        self._season = season
        self._league = league
        self._teams_occurrences = {}
        self._score_occurrences = {}
        self._persons_occurrences = {}
        self._hashtag_occurrences = {}
        self._emoticons_occurrences = {}
        self._add_all_simple_facts_related_occurrences(n_latest_tweets_used_for_result_collection)
        self.team_1, self.team_2 = self._get_teams_which_played()
        self._add_all_person_occurrences()

    def print_basic_facts(
            self,
            n_most_often_mentioned_persons: int,
            n_most_popular_hashtags: int,
            n_most_popular_emoticons: int
    ):
        self._print_teams_that_have_played()
        self._print_result()
        self._print_n_most_often_mentioned_persons(n_most_often_mentioned_persons)
        self._print_n_most_popular_hashtags(n_most_popular_hashtags)
        self._print_n_most_popular_emoticons(n_most_popular_emoticons)

    def _add_all_simple_facts_related_occurrences(self, latest_tweets_count):
        tweets_to_process_range = range(
            len(self._corpus_without_emoticons) - latest_tweets_count if latest_tweets_count else 0,
            len(self._corpus_without_emoticons)
        )
        for tweet_idx, tweet in enumerate(self._corpus_without_emoticons):
            if tweet_idx in tweets_to_process_range:
                self._add_potential_match_result_occurrence(tweet)
            self._process_tweet_tokens(tweet)

        self._process_emoticons()

    def _process_tweet_tokens(self, tweet):
        previous_token = None
        for token in tweet:
            token = token.lower()
            self._add_potential_team_occurrence(previous_token, token)
            self._add_potential_hashtag_occurrence(token)
            previous_token = token

    def _process_emoticons(self):
        for tweet in self._corpus_with_emoticons:
            for character in tweet:
                self._add_potential_emoticon_occurrence(character)

    def _add_all_person_occurrences(self):
        tweets_with_tags = self._tagger.tag_sents(self._corpus_without_emoticons)
        person_tagged_tweets = self._remove_tweets_without_persons_tags(
            tweets_with_tags=tweets_with_tags
        )
        self._persons_occurrences = {}
        teams_squads = self._get_teams_squads()

        for tweet in person_tagged_tweets:
            self._process_tweet_for_persons(tweet, teams_squads)

    def _process_tweet_for_persons(self, tweet, teams_squads):
        previous_token = {"value": None, "tag": None}

        for token in tweet:
            token_value, token_tag = token
            if token_tag == self.PERSON_TAG:
                person_found = self._find_person_in_squads(
                    teams_squads, token_value, previous_token
                )
                if person_found:
                    self._persons_occurrences[person_found] = (
                            self._persons_occurrences.get(person_found, 0) + 1
                    )
            previous_token["value"] = token_value
            previous_token["tag"] = token_tag

    def _find_person_in_squads(self, teams_squads, current_value, previous_token):
        person = teams_squads.find_person_in_squads(last_name=current_value)
        if not person and previous_token["tag"] == self.PERSON_TAG:
            full_name = f"{previous_token['value']} {current_value}"
            person = teams_squads.find_person_in_squads(last_name=full_name)
        return person

    @new_line_appendix_decorator
    def _print_teams_that_have_played(self) -> None:
        print(self.QUESTION_WHO_PLAYED, f"- {self.team_1} and {self.team_2}", sep="\n")

    @new_line_appendix_decorator
    def _print_result(self) -> None:
        team_1, score, team_2 = self._get_match_result()
        print(self.QUESTION_RESULT, f"- {team_1} {score} {team_2}", sep="\n")

    @new_line_appendix_decorator
    def _print_n_most_often_mentioned_persons(self, n: int) -> None:
        top_most_often_mentioned_persons = self._get_top_n_counted_items_from_dict(
            dict=self._persons_occurrences, n=n,
        )
        top_persons = []
        for person, person_count in top_most_often_mentioned_persons:
            person_dict = {
                **person.to_dict(),
                "occurrence_count": person_count,
            }
            top_persons.append(person_dict)
        top_persons_df = pd.DataFrame(top_persons)
        print(self.QUESTION_MOST_OFTEN_MENTIONED_PERSONS, top_persons_df, sep="\n")

    @new_line_appendix_decorator
    def _print_n_most_popular_hashtags(self, n: int) -> None:
        top_n_most_popular_hashtags = self._get_top_n_counted_items_from_dict(
            dict=self._hashtag_occurrences, n=n,
        )
        top_n_hashtags = []
        for hashtag, hashtag_count in top_n_most_popular_hashtags:
            hashtag_dict = {
                "hashtag": hashtag,
                "hashtag_count": hashtag_count,
            }
            top_n_hashtags.append(hashtag_dict)
        top_n_hashtags_df = pd.DataFrame(top_n_hashtags)
        print(self.QUESTION_MOST_POPULAR_HASHTAGS, top_n_hashtags_df, sep="\n")

    @new_line_appendix_decorator
    def _print_n_most_popular_emoticons(self, n: int) -> None:
        top_n_most_popular_emoticons = self._get_top_n_counted_items_from_dict(
            dict=self._emoticons_occurrences, n=n,
        )
        top_n_emoticons = []
        for emoticon, emoticon_count in top_n_most_popular_emoticons:
            hashtag_dict = {
                "emoticon": emoticon,
                "emoticon_description": demojize(emoticon.decode('unicode-escape')),
                "emoticon_count": emoticon_count,
            }
            top_n_emoticons.append(hashtag_dict)
        top_n_emoticons_df = pd.DataFrame(top_n_emoticons)
        print(self.QUESTION_MOST_POPULAR_EMOTICONS, top_n_emoticons_df , sep="\n")

    def _get_teams_which_played(self) -> Tuple[str, str]:
        teams = list(sorted(self._teams_occurrences, key=self._teams_occurrences.get,))
        team_1 = self._get_club_name_from_club_key(key=teams[-1])
        team_2 = self._get_club_name_from_club_key(key=teams[-2])
        return team_1, team_2

    def _get_match_result(self) -> Tuple[str, str, str]:
        scores_sorted_by_occurrence = sorted(
            self._score_occurrences.items(), key=lambda score_: score_[1], reverse=True
        )
        most_frequent_score = scores_sorted_by_occurrence[0][0]
        team_1, score, team_2 = most_frequent_score.split()
        team_1 = self._get_club_name_from_club_key(key=team_1)
        team_2 = self._get_club_name_from_club_key(key=team_2)
        return team_1, score, team_2

    @classmethod
    def _remove_tweets_without_persons_tags(cls, tweets_with_tags):
        return [
            tweet
            for tweet in tweets_with_tags
            if cls.PERSON_TAG in {token_with_tag[1] for token_with_tag in tweet}
        ]

    def _add_potential_match_result_occurrence(self, tweet: List[str]):
        tweet_text = " ".join(tweet)
        found_result = re.search(NORMALIZED_FOOTBALL_RESULT_WITH_TEAMS_FORMAT_REGEX, tweet_text)
        if not found_result:
            return
        tweet = [token.lower() for token in tweet]
        tweet_text = " ".join(tweet)
        result = re.findall(NORMALIZED_FOOTBALL_RESULT_FORMAT_REGEX, tweet_text)[0]
        result_index = tweet.index(result)
        n_minus_2_token = tweet[result_index - 2] if result_index >= 2 else None
        n_minus_1_token = tweet[result_index - 1]
        n_plus_1_token = tweet[result_index + 1]
        n_plus_2_token = tweet[result_index + 2] if (len(tweet) - result_index - 1 >= 2) else None
        team_1 = self._get_proper_club_key(first_token=n_minus_2_token, second_token=n_minus_1_token, left_side_of_result=True)
        team_2 = self._get_proper_club_key(first_token=n_plus_1_token, second_token=n_plus_2_token, left_side_of_result=False)
        standardized_result = f"{team_1} {result} {team_2}"
        if team_1 and team_2:
            occurrence_score = self._score_occurrences.get(standardized_result, 0)
            self._score_occurrences[standardized_result] = occurrence_score + 1

    def _get_proper_club_key(self, first_token: str, second_token: str, left_side_of_result: bool):
        joined_tokens = f"{first_token}{second_token}"
        token_closer_to_result = second_token if left_side_of_result else first_token
        token_further_to_result = first_token if left_side_of_result else second_token
        if token_closer_to_result in self._premier_league_team_keys:
            return token_closer_to_result
        elif token_further_to_result and joined_tokens in self._premier_league_team_keys:
            return joined_tokens
        elif token_closer_to_result in self._primiera_division_team_keys:
            return token_closer_to_result
        elif token_further_to_result and joined_tokens in self._primiera_division_team_keys:
            return joined_tokens

    def _add_potential_team_occurrence(self, first_token: str, second_token: str):
        if second_token in self._premier_league_team_keys:
            self._add_team_occurrence(team_key=second_token)
        elif first_token and f"{first_token}{second_token}" in self._premier_league_team_keys:
            self._add_team_occurrence(team_key=f"{first_token}{second_token}")
        elif second_token in self._primiera_division_team_keys:
            self._add_team_occurrence(team_key=second_token)
        elif first_token and f"{first_token}{second_token}" in self._primiera_division_team_keys:
            self._add_team_occurrence(team_key=f"{first_token}{second_token}")

    def _add_team_occurrence(self, team_key: str):
        self._teams_occurrences[team_key] = self._teams_occurrences.get(team_key, 0) + 1

    def _add_potential_hashtag_occurrence(self, token: str):
        if token[0] == "#":
            self._hashtag_occurrences[token] = self._hashtag_occurrences.get(token, 0) + 1

    def _add_potential_emoticon_occurrence(self, character: str):
        if character in UNICODE_EMOJI:
            self._emoticons_occurrences[character.encode('unicode-escape')] = (
                self._emoticons_occurrences.get(character.encode('unicode-escape'), 0) + 1
            )

    def _get_club_name_from_club_key(self, key):
        if key in self._premier_league_team_keys:
            return self._premier_league_teams[
                self._premier_league_teams[CLUBS_KEY_COLUMN_NAME] == key
                ][CLUBS_NAME_COLUMN_NAME].iloc[0]
        elif key in self._primiera_division_team_keys:
            return self._primiera_division_teams[
                self._primiera_division_teams[CLUBS_KEY_COLUMN_NAME] == key
                ][CLUBS_NAME_COLUMN_NAME].iloc[0]

    def _get_teams_squads(self):
        teams_squads_scrapper = TeamsSquadsScrapper(
            team_1_name=self.team_1,
            team_2_name=self.team_2,
            season=self._season,
            league=self._league,
        )
        return teams_squads_scrapper.get_teams_squads()

    @classmethod
    def _get_top_n_counted_items_from_dict(cls, dict: Dict, n: int):
        return sorted(dict.items(), key=lambda t: t[1], reverse=True,)[:n]

    @classmethod
    def _tokenize_tweets_corpus(cls, corpus: List[str]) -> List[List[str]]:
        return [CustomTokenizer.tokenize(tweet) for tweet in corpus]


@section_printing_decorator("BASIC_FACTS")
def print_basic_facts(
    corpus_without_emoticons,
    corpus_with_emoticons,
    season: Season,
    league: League,
    paths_config: PathsConfig,
    n_latest_tweets_used_for_result_collection: int = None,
    n_most_often_mentioned_persons: int = 30,
    n_most_popular_hashtags: int = 10,
    n_most_popular_emoticons: int = 10,
):
    facts_extractor = FactsExtractor(
        corpus_without_emoticons=corpus_without_emoticons,
        corpus_with_emoticons=corpus_with_emoticons,
        n_latest_tweets_used_for_result_collection=n_latest_tweets_used_for_result_collection,
        season=season,
        league=league,
        paths_config=paths_config
    )
    facts_extractor.print_basic_facts(
        n_most_often_mentioned_persons=n_most_often_mentioned_persons,
        n_most_popular_hashtags=n_most_popular_hashtags,
        n_most_popular_emoticons=n_most_popular_emoticons
    )
