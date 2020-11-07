from string import Template
from data.paths import ENGLISH_CLUBS_FILE_PATH
from data.loaders import DataLoader
from data.column_names import ENGLISH_CLUBS_KEY_COLUMN_NAME, ENGLISH_CLUBS_NAME_COLUMN_NAME
from typing import Dict, Optional, Tuple, Set, FrozenSet
from corpus.cleaning.constants import NORMALIZED_FOOTBALL_RESULT_WITH_TEAMS_FORMAT_REGEX
from nltk.tag.stanford import StanfordNERTagger
import re
import time
import nltk
import pprint
from collections import OrderedDict
from scraping.scrappers import TeamsSquadsScrapper
# todo update pretrained model version



class QuestionsAnswerer:
    PERSON_TAG = "PERSON"
    QUESTION_WHO_PLAYED = "Who have played?"
    QUESTION_SCORE = "What was the score?"
    QUESTION_MOST_OFTEN_MENTIONED_PLAYERS = "Which persons were the most often mentioned?"
    QUESTION_MOST_POPULAR_HASHTAGS = "What are the most popular hashtags?"
    QUESTION_MOST_POPULAR_EMOTICONS = "What are the most popular emoticons?"
    ANSWERS_TEMPLATE = Template(
        f"{QUESTION_WHO_PLAYED}\n"
        f"- $who_played\n"
        f"{QUESTION_SCORE}\n"
        f"- $score\n"
        f"{QUESTION_MOST_OFTEN_MENTIONED_PLAYERS}\n"
        f"- $most_often_mentioned_players\n"
        f"{QUESTION_MOST_POPULAR_HASHTAGS}\n"
        f"- TODO\n"
        f"{QUESTION_MOST_POPULAR_EMOTICONS}\n"
        f"- TODO\n"
    )

    def __init__(self, corpus) -> None:
        self._corpus = corpus
        self._english_teams = DataLoader.from_csv(ENGLISH_CLUBS_FILE_PATH)
        self._team_keys = set(self._english_teams[ENGLISH_CLUBS_KEY_COLUMN_NAME])
        self._tagger = StanfordNERTagger(
            "pretrained_models/stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.crf.ser.gz",
            'pretrained_models/stanford-ner-2014-08-27/stanford-ner-3.4.1.jar'
        )
        self._teams_occurrences = {}
        self._score_occurrences = {}
        self._persons_occurrences = {}

    def answer_basic_questions(self):
        self._add_all_answer_related_occurrences()
        return self.ANSWERS_TEMPLATE.substitute(
            who_played=self._answer_who_played(),
            score=self._answer_what_was_the_score(),
            most_often_mentioned_players=self._answer_who_were_the_most_often_mentioned_players()
        )

    def _add_all_answer_related_occurrences(self):
        for tweet in self._corpus:
            previous_token_value = None
            for token_index, token in enumerate(tweet):
                token = token.lower()
                next_token_index = token_index + 1
                if next_token_index < len(tweet):
                    self._add_potential_match_result_occurrence(
                        first_token=previous_token_value,
                        second_token=token,
                        third_token=tweet[next_token_index].lower()
                    )
                self._add_potential_team_occurrence(previous_token_value, token)
                previous_token_value = token

        # tags
        tweets_with_tags = self._tagger.tag_sents(self._corpus)
        tweets_including_persons_tags = self._remove_tweets_without_persons_tags(
            tweets_with_tags=tweets_with_tags
        )
        # pprint.pprint(tweets_including_persons_tags)
        self._persons_occurrences = {}
        teams_squads = self._get_teams_squads()
        for tweet in tweets_including_persons_tags:
            previous_token_value = None
            previous_token_tag = None
            for token_index, token in enumerate(tweet):
                token_value, token_tag = token
                if token_tag == self.PERSON_TAG:
                    person_found = teams_squads.find_person_in_squads(last_name=token_value)
                    if not person_found and previous_token_tag == self.PERSON_TAG:
                        consecutive_person_tagged_tokens = f"{previous_token_value} {token_value}"
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
        pprint.pprint(
            sorted(
                self._persons_occurrences.items(),
                key=lambda t: t[1],
            )
        )

    def _answer_who_played(self) -> str:
        teams = list(
            sorted(
                self._teams_occurrences,
                key=self._teams_occurrences.get,
            )
        )
        team_1 = self._english_teams[
            self._english_teams[ENGLISH_CLUBS_KEY_COLUMN_NAME] == teams[-1]
        ][ENGLISH_CLUBS_NAME_COLUMN_NAME].iloc[0]
        team_2 = self._english_teams[
            self._english_teams[ENGLISH_CLUBS_KEY_COLUMN_NAME] == teams[-2]
            ][ENGLISH_CLUBS_NAME_COLUMN_NAME].iloc[0]
        return f"{team_1} and {team_2}"

    def _answer_what_was_the_score(self) -> str:
        scores_sorted_by_occurrence = sorted(
            self._score_occurrences.items(),
            key=lambda score: score[1],
            reverse=True
        )
        most_frequent_score = scores_sorted_by_occurrence[0][0]
        team_1, score, team_2 = most_frequent_score.split()
        team_1 = self._english_teams[
            self._english_teams[ENGLISH_CLUBS_KEY_COLUMN_NAME] == team_1
            ][ENGLISH_CLUBS_NAME_COLUMN_NAME].iloc[0]
        team_2 = self._english_teams[
            self._english_teams[ENGLISH_CLUBS_KEY_COLUMN_NAME] == team_2
            ][ENGLISH_CLUBS_NAME_COLUMN_NAME].iloc[0]
        return f"{team_1} {score} {team_2}"

    def _answer_who_were_the_most_often_mentioned_players(self) -> str:
        return f""

    @classmethod
    def _remove_tweets_without_persons_tags(cls, tweets_with_tags):
        return [
            tweet for tweet in tweets_with_tags
            if cls.PERSON_TAG in {token_with_tag[1] for token_with_tag in tweet}
        ]

    def _add_potential_match_result_occurrence(self, first_token: str, second_token: str, third_token: str):
        if not first_token:
            return
        concatenated_tokens = f"{first_token} {second_token} {third_token}"
        if re.match(NORMALIZED_FOOTBALL_RESULT_WITH_TEAMS_FORMAT_REGEX, concatenated_tokens):
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

    def _add_potential_player_occurrence(self, player_key: str):
        if self._persons_occurrences.get(player_key):
            self._persons_occurrences[player_key] += 1
        else:
            self._persons_occurrences[player_key] = 1

    @classmethod
    def _get_teams_squads(cls):
        teams_squads_scrapper = TeamsSquadsScrapper(team_1_name="Watford FC", team_2_name="Liverpool FC")
        return teams_squads_scrapper.get_teams_squads()
