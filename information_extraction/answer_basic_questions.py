from string import Template
from data.paths import ENGLISH_CLUBS_FILE_PATH
from data.loaders import DataLoader
from typing import Dict, Optional
from corpus.cleaning.constants import NORMALIZED_FOOTBALL_RESULT_FORMAT_REGEX

import re


class QuestionsAnswerer:
    SCORE_VERBS = {"scored", "scores"}
    QUESTION_WHO_PLAYED = "Who played?"
    QUESTION_SCORE = "What was the score?"
    QUESTION_WHO_SCORED = "Who scored?"
    QUESTION_WHERE = "Where did it happen?"
    QUESTION_MOST_ACCURATE_TWEET = "What was the most accurate tweet about event?"
    ANSWERS_TEMPLATE = Template(
        f"{QUESTION_WHO_PLAYED}\n"
        f"$who_played\n"
        f"{QUESTION_SCORE}\n"
        f"$score\n"
        f"{QUESTION_WHO_SCORED}\n"
        f"$who_played\n"
        f"{QUESTION_WHERE}\n"
        f"$who_played\n"
        f"{QUESTION_MOST_ACCURATE_TWEET}\n"
        f"$who_played\n"
    )

    def __init__(self, corpus) -> None:
        self._corpus = corpus
        self._english_teams = DataLoader.from_csv(ENGLISH_CLUBS_FILE_PATH)
        self._teams_occurrences = {}
        self._score_occurrences = {}
        self._scorers_occurrences = {}  # todo - make it better (check word occurence with words like goal, goals, score, scores, scored)

    def answer_basic_questions(self):
        who_played = self._get_rivals()
        for tweet in self._corpus:
            previous_word = None
            for token in tweet:
                if re.match(NORMALIZED_FOOTBALL_RESULT_FORMAT_REGEX, token):
                    if self._score_occurrences.get(token):
                        self._score_occurrences[token] += 1
                    else:
                        self._score_occurrences[token] = 1
                if self._scorer_found(first_token=previous_word, second_token=token):
                    if self._score_occurrences.get(token):
                        self._scorers_occurrences[token] += 1
                    else:
                        self._scorers_occurrences[token] = 1
                previous_word = token
        score = self._get_score()
        return self.ANSWERS_TEMPLATE.substitute(
            who_played=who_played,
            score=score
        )

    @classmethod
    def _get_rivals(cls):
        return "Liverpool and Watford"

    def _get_score(self):
        scores_sorted_by_occurence = sorted(
            self._score_occurrences.items(),
            key=lambda score: score[1],
            reverse=True
        )
        most_frequent_score = scores_sorted_by_occurence[0][0]
        return most_frequent_score

    @classmethod
    def _scorer_found(cls, first_token: str, second_token: str) -> Optional[str]:
        if first_token and second_token in cls.SCORE_VERBS:
            return first_token

    @classmethod
    def _add_team_occurrence(cls, prefix, teams ):
        print("Searching prefix:{}".format(prefix))
        while True:
            name = (yield)
            if prefix in name:
                print(name)