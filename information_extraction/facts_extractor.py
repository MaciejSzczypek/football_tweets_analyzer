from string import Template
from data.paths import ENGLISH_CLUBS_FILE_PATH
from data.loaders import DataLoader
from data.column_names import (
    ENGLISH_CLUBS_KEY_COLUMN_NAME,
    ENGLISH_CLUBS_NAME_COLUMN_NAME,
)
from typing import Dict, Optional, Tuple, Set, FrozenSet, List
from corpus.cleaning.constants import NORMALIZED_FOOTBALL_RESULT_WITH_TEAMS_FORMAT_REGEX
from nltk.tag.stanford import StanfordNERTagger
import re
import time
import nltk
import pprint
from collections import OrderedDict
from scraping.scrappers import TeamsSquadsScrapper
from corpus.tokenizing.custom_tokenizers import CustomTokenizer
import re
from utils.printing import section_printing_decorator, new_line_appendix_decorator
import pandas as pd

# todo update pre-trained model version


class FactsExtractor:
    PERSON_TAG = "PERSON"
    QUESTION_WHO_PLAYED = "Which teams have played?"
    QUESTION_RESULT = "What was the result?"
    QUESTION_MOST_OFTEN_MENTIONED_PERSONS = (
        "Which persons were the most often mentioned?"
    )
    QUESTION_MOST_POPULAR_HASHTAGS = "What are the most popular hashtags?"
    QUESTION_MOST_POPULAR_EMOTICONS = "What are the most popular emoticons?"

    def __init__(self, corpus: List[str]) -> None:
        self._corpus = self._tokenize_tweets_corpus(corpus)
        self._premier_league_teams = DataLoader.from_csv(ENGLISH_CLUBS_FILE_PATH)
        self._team_keys = set(self._premier_league_teams[ENGLISH_CLUBS_KEY_COLUMN_NAME])
        self._tagger = StanfordNERTagger(
            "pretrained_models/stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.crf.ser.gz",
            "pretrained_models/stanford-ner-2014-08-27/stanford-ner-3.4.1.jar",
        )
        self._teams_occurrences = {}
        self._score_occurrences = {}
        self._persons_occurrences = {}
        self._hashtag_occurrences = {}
        self._emoticons_occurrences = {}
        self._add_all_simple_facts_related_occurrences()
        self._add_all_person_occurrences()

    def show_basic_facts(self):
        self._print_teams_that_have_played()
        self._print_result()
        self._print_5_most_often_mentioned_persons()
        self._print_10_most_popular_hashtags()
        self._print_5_most_popular_emoticons()

    def _add_all_simple_facts_related_occurrences(self):
        for tweet in self._corpus:
            previous_token_value = None
            for token_index, token in enumerate(tweet):
                token = token.lower()
                next_token_index = token_index + 1
                if next_token_index < len(tweet):
                    self._add_potential_match_result_occurrence(
                        first_token=previous_token_value,
                        second_token=token,
                        third_token=tweet[next_token_index].lower(),
                    )
                self._add_potential_team_occurrence(previous_token_value, token)
                self._add_potential_hashtag_occurrence(token)
                previous_token_value = token

    def _add_all_person_occurrences(self):
        tweets_with_tags = self._tagger.tag_sents(self._corpus)
        tweets_including_persons_tags = self._remove_tweets_without_persons_tags(
            tweets_with_tags=tweets_with_tags
        )
        self._persons_occurrences = {}
        teams_squads = self._get_teams_squads()
        for tweet in tweets_including_persons_tags:
            previous_token_value = None
            previous_token_tag = None
            for token_index, token in enumerate(tweet):
                token_value, token_tag = token
                if token_tag == self.PERSON_TAG:
                    person_found = teams_squads.find_person_in_squads(
                        last_name=token_value
                    )
                    if not person_found and previous_token_tag == self.PERSON_TAG:
                        consecutive_person_tagged_tokens = (
                            f"{previous_token_value} {token_value}"
                        )
                        person_found = teams_squads.find_person_in_squads(
                            last_name=consecutive_person_tagged_tokens
                        )
                    if not person_found:
                        continue
                    if person_found in self._persons_occurrences:
                        self._persons_occurrences[person_found] += 1
                    else:
                        self._persons_occurrences[person_found] = 1
                previous_token_value = token_value
                previous_token_tag = token_tag

    @new_line_appendix_decorator
    def _print_teams_that_have_played(self) -> None:
        team_1, team_2 = self._get_teams_which_played()
        print(self.QUESTION_WHO_PLAYED, f"- {team_1} and {team_2}", sep="\n")

    @new_line_appendix_decorator
    def _print_result(self) -> None:
        team_1, score, team_2 = self._get_match_result()
        print(self.QUESTION_RESULT, f"- {team_1} {score} {team_2}", sep="\n")

    @new_line_appendix_decorator
    def _print_5_most_often_mentioned_persons(self) -> None:
        top_5_most_often_mentioned_persons = self._get_top_n_counted_items_from_dict(
            dict=self._persons_occurrences, n=5,
        )
        top_5_persons = []
        for person, person_count in top_5_most_often_mentioned_persons:
            person_dict = {
                **person.to_dict(),
                "occurrence_count": person_count,
            }
            top_5_persons.append(person_dict)
        top_5_persons_df = pd.DataFrame(top_5_persons)
        print(self.QUESTION_MOST_OFTEN_MENTIONED_PERSONS, top_5_persons_df, sep="\n")

    @new_line_appendix_decorator
    def _print_10_most_popular_hashtags(self) -> None:
        top_5_most_popular_hashtags = self._get_top_n_counted_items_from_dict(
            dict=self._hashtag_occurrences, n=10,
        )
        top_5_hashtags = []
        for hashtag, hashtag_count in top_5_most_popular_hashtags:
            hashtag_dict = {
                "hashtag": hashtag,
                "hashtag_count": hashtag_count,
            }
            top_5_hashtags.append(hashtag_dict)
        top_5_persons_df = pd.DataFrame(top_5_hashtags)
        print(self.QUESTION_MOST_POPULAR_HASHTAGS, top_5_persons_df, sep="\n")

    @new_line_appendix_decorator
    def _print_5_most_popular_emoticons(self) -> None:
        print(self.QUESTION_MOST_POPULAR_EMOTICONS, sep="\n")

    def _get_teams_which_played(self) -> Tuple[str, str]:
        teams = list(sorted(self._teams_occurrences, key=self._teams_occurrences.get,))
        team_1 = self._get_club_name_from_club_key(key=teams[-1])
        team_2 = self._get_club_name_from_club_key(key=teams[-2])
        return team_1, team_2

    def _get_match_result(self) -> Tuple[str, str, str]:
        scores_sorted_by_occurrence = sorted(
            self._score_occurrences.items(), key=lambda score: score[1], reverse=True
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

    def _add_potential_match_result_occurrence(
        self, first_token: str, second_token: str, third_token: str
    ):
        if not first_token:
            return
        concatenated_tokens = f"{first_token} {second_token} {third_token}"
        if re.match(
            NORMALIZED_FOOTBALL_RESULT_WITH_TEAMS_FORMAT_REGEX, concatenated_tokens
        ):
            if self._score_occurrences.get(concatenated_tokens):
                self._score_occurrences[concatenated_tokens] += 1
            else:
                self._score_occurrences[concatenated_tokens] = 1

    def _add_potential_team_occurrence(self, first_token: str, second_token: str):
        if second_token in self._team_keys:
            self._add_team_occurrence(team_key=second_token)
        elif first_token and f"{first_token}{second_token}" in self._team_keys:
            self._add_team_occurrence(team_key=f"{first_token}{second_token}")

    def _add_team_occurrence(self, team_key: str):
        if self._teams_occurrences.get(team_key):
            self._teams_occurrences[team_key] += 1
        else:
            self._teams_occurrences[team_key] = 1

    def _add_potential_hashtag_occurrence(self, token: str):
        if token[0] == "#":
            if self._hashtag_occurrences.get(token):
                self._hashtag_occurrences[token] += 1
            else:
                self._hashtag_occurrences[token] = 1

    def _get_club_name_from_club_key(self, key):
        return self._premier_league_teams[
            self._premier_league_teams[ENGLISH_CLUBS_KEY_COLUMN_NAME] == key
        ][ENGLISH_CLUBS_NAME_COLUMN_NAME].iloc[0]

    @classmethod
    def _get_teams_squads(cls):
        teams_squads_scrapper = TeamsSquadsScrapper(
            team_1_name="Watford FC", team_2_name="Liverpool FC"
        )
        return teams_squads_scrapper.get_teams_squads()

    @classmethod
    def _get_top_n_counted_items_from_dict(cls, dict: Dict, n: int):
        return sorted(dict.items(), key=lambda t: t[1], reverse=True,)[:n]

    @classmethod
    def _tokenize_tweets_corpus(cls, corpus: List[str]) -> List[List[str]]:
        return [CustomTokenizer.tokenize(tweet) for tweet in corpus]
