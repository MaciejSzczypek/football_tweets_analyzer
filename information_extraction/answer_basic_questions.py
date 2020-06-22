from string import Template
from data.paths import ENGLISH_CLUBS_FILE_PATH
from data.loaders import DataLoader
from data.column_names import ENGLISH_CLUBS_KEY_COLUMN_NAME, ENGLISH_CLUBS_NAME_COLUMN_NAME
from typing import Dict, Optional, Tuple
from corpus.cleaning.constants import NORMALIZED_FOOTBALL_RESULT_WITH_TEAMS_FORMAT_REGEX

import re


class QuestionsAnswerer:
    QUESTION_WHO_PLAYED = "Who have played?"
    QUESTION_SCORE = "What was the score?"
    QUESTION_WHERE = "Where did it happen?"
    QUESTION_MOST_POPULAR_HASHTAGS = "What are the most popular hashtags?"
    QUESTION_MOST_POPULAR_EMOTICONS = "What are the most popular emoticons?"
    ANSWERS_TEMPLATE = Template(
        f"{QUESTION_WHO_PLAYED}\n"
        f"- $who_played\n"
        f"{QUESTION_SCORE}\n"
        f"- $score\n"
        f"{QUESTION_WHERE}\n"
        f"- $who_played\n"
    )

    def __init__(self, corpus) -> None:
        self._corpus = corpus
        self._english_teams = DataLoader.from_csv(ENGLISH_CLUBS_FILE_PATH)
        self._team_keys = set(self._english_teams[ENGLISH_CLUBS_KEY_COLUMN_NAME])
        self._teams_occurrences = {}
        self._score_occurrences = {}

    def answer_basic_questions(self):
        for tweet in self._corpus:
            previous_token = None
            for index, token in enumerate(tweet):
                token = token.lower()
                if index + 1 < len(tweet):
                    self._add_match_result_if_occurs(
                        first_token=previous_token,
                        second_token=token,
                        third_token=tweet[index + 1].lower()
                    )
                self._add_team_key_if_occurs(previous_token, token)
                previous_token = token
        score = self._get_score()
        team_1, team_2 = self._get_teams()
        who_played = f"{team_1} and {team_2}"
        return self.ANSWERS_TEMPLATE.substitute(
            who_played=who_played,
            score=score
        )

    def _get_teams(self) -> Tuple[str, str]:
        teams = list(
            sorted(
                self._teams_occurrences,
                key=self._teams_occurrences.get,
            )
        )
        team_1 = self._english_teams[
            self._english_teams[ENGLISH_CLUBS_KEY_COLUMN_NAME] == teams[-2]
        ][ENGLISH_CLUBS_NAME_COLUMN_NAME].iloc[0]
        team_2 = self._english_teams[
            self._english_teams[ENGLISH_CLUBS_KEY_COLUMN_NAME] == teams[-1]
            ][ENGLISH_CLUBS_NAME_COLUMN_NAME].iloc[0]
        return team_1, team_2

    def _get_score(self):
        scores_sorted_by_occurrence = sorted(
            self._score_occurrences.items(),
            key=lambda score: score[1],
            reverse=True
        )
        most_frequent_score = scores_sorted_by_occurrence[0][0]
        return most_frequent_score

    def _add_match_result_if_occurs(self, first_token: str, second_token: str, third_token: str):
        if not first_token:
            return
        concatenated_tokens = f"{first_token} {second_token} {third_token}"
        if re.match(NORMALIZED_FOOTBALL_RESULT_WITH_TEAMS_FORMAT_REGEX, concatenated_tokens):
            if self._score_occurrences.get(concatenated_tokens):
                self._score_occurrences[concatenated_tokens] += 1
            else:
                self._score_occurrences[concatenated_tokens] = 1

    def _add_team_key_if_occurs(self, first_token: str, second_token: str):
        if second_token in self._team_keys:
            self._add_team_occurrence(team_key=second_token)
        elif first_token and f"{first_token}{second_token}" in self._team_keys:
            self._add_team_occurrence(team_key=f"{first_token}{second_token}")

    def _add_team_occurrence(self, team_key):
        if self._teams_occurrences.get(team_key):
            self._teams_occurrences[team_key] += 1
        else:
            self._teams_occurrences[team_key] = 1
