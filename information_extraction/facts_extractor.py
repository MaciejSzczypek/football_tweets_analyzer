import re
from typing import Dict, Tuple, List

import pandas as pd
from emoji import demojize, UNICODE_EMOJI
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

    def __init__(
        self,
        corpus_without_emoticons: List[str],
        corpus_with_emoticons: List[str],
        n_latest_tweets_used_for_result_collection: int,
    ) -> None:
        self._corpus_without_emoticons = self._tokenize_tweets_corpus(corpus_without_emoticons)
        self._corpus_with_emoticons = corpus_with_emoticons
        self._premier_league_teams = DataLoader.from_csv(ENGLISH_CLUBS_FILE_PATH)
        self._primiera_division_teams = DataLoader.from_csv(SPANISH_CLUBS_FILE_PATH)
        self._premier_league_team_keys = set(self._premier_league_teams[CLUBS_KEY_COLUMN_NAME])
        self._primiera_division_team_keys = set(self._primiera_division_teams[CLUBS_KEY_COLUMN_NAME])
        self._tagger = StanfordNERTagger(
            "pretrained_models/stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.crf.ser.gz",
            "pretrained_models/stanford-ner-2014-08-27/stanford-ner-3.4.1.jar",
        )
        self._teams_occurrences = {}
        self._score_occurrences = {}
        self._persons_occurrences = {}
        self._hashtag_occurrences = {}
        self._emoticons_occurrences = {}
        self._add_all_simple_facts_related_occurrences(n_latest_tweets_used_for_result_collection)
        self._add_all_person_occurrences()

    def show_basic_facts(self):
        self._print_teams_that_have_played()
        self._print_result()
        self._print_10_most_often_mentioned_persons()
        self._print_10_most_popular_hashtags()
        self._print_10_most_popular_emoticons()

    def _add_all_simple_facts_related_occurrences(self, n_latest_tweets_used_for_result_collection):
        range_of_tweets_for_result_collection = range(
            len(self._corpus_without_emoticons) - n_latest_tweets_used_for_result_collection
            if n_latest_tweets_used_for_result_collection else 0,
            len(self._corpus_without_emoticons)
        )
        for tweet_index, tweet in enumerate(self._corpus_without_emoticons):
            previous_token_value = None
            if tweet_index in range_of_tweets_for_result_collection:
                self._add_potential_match_result_occurrence(tweet)
            for token_index, token in enumerate(tweet):
                token = token.lower()
                self._add_potential_team_occurrence(previous_token_value, token)
                self._add_potential_hashtag_occurrence(token)
                previous_token_value = token
        for tweet in self._corpus_with_emoticons:
            for character in tweet:
                self._add_potential_emoticon_occurrence(character)

    def _add_all_person_occurrences(self):
        tweets_with_tags = self._tagger.tag_sents(self._corpus_without_emoticons)
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
    def _print_10_most_often_mentioned_persons(self) -> None:
        top_10_most_often_mentioned_persons = self._get_top_n_counted_items_from_dict(
            dict=self._persons_occurrences, n=10,
        )
        top_10_persons = []
        for person, person_count in top_10_most_often_mentioned_persons:
            person_dict = {
                **person.to_dict(),
                "occurrence_count": person_count,
            }
            top_10_persons.append(person_dict)
        top_10_persons_df = pd.DataFrame(top_10_persons)
        top_10_persons_df.to_csv("results/top_mentioned_people.csv")
        print(self.QUESTION_MOST_OFTEN_MENTIONED_PERSONS, top_10_persons_df, sep="\n")

    @new_line_appendix_decorator
    def _print_10_most_popular_hashtags(self) -> None:
        top_10_most_popular_hashtags = self._get_top_n_counted_items_from_dict(
            dict=self._hashtag_occurrences, n=10,
        )
        top_10_hashtags = []
        for hashtag, hashtag_count in top_10_most_popular_hashtags:
            hashtag_dict = {
                "hashtag": hashtag,
                "hashtag_count": hashtag_count,
            }
            top_10_hashtags.append(hashtag_dict)
        top_10_hashtags_df = pd.DataFrame(top_10_hashtags)
        top_10_hashtags_df.to_csv("results/top_hashtags.csv")
        print(self.QUESTION_MOST_POPULAR_HASHTAGS, top_10_hashtags_df, sep="\n")

    @new_line_appendix_decorator
    def _print_10_most_popular_emoticons(self) -> None:
        top_10_most_popular_emoticons = self._get_top_n_counted_items_from_dict(
            dict=self._emoticons_occurrences, n=11,
        )
        top_10_emoticons = []
        for emoticon, emoticon_count in top_10_most_popular_emoticons:
            hashtag_dict = {
                "emoticon": emoticon,
                "emoticon_description": demojize(emoticon.decode('unicode-escape')),
                "emoticon_count": emoticon_count,
            }
            top_10_emoticons.append(hashtag_dict)
        top_10_emoticons_df = pd.DataFrame(top_10_emoticons)
        top_10_emoticons_df.to_csv("results/top_emoticons.csv")
        print(self.QUESTION_MOST_POPULAR_EMOTICONS, top_10_emoticons_df, sep="\n")

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

    def _add_potential_emoticon_occurrence(self, character: str):
        if character in UNICODE_EMOJI:
            # print(f"|{character}|", len(character), character.encode('unicode-escape'))
            if self._emoticons_occurrences.get(character.encode('unicode-escape')):
                self._emoticons_occurrences[character.encode('unicode-escape')] += 1
            else:
                self._emoticons_occurrences[character.encode('unicode-escape')] = 1

    def _get_club_name_from_club_key(self, key):
        if key in self._premier_league_team_keys:
            return self._premier_league_teams[
                self._premier_league_teams[CLUBS_KEY_COLUMN_NAME] == key
                ][CLUBS_NAME_COLUMN_NAME].iloc[0]
        elif key in self._primiera_division_team_keys:
            return self._primiera_division_teams[
                self._primiera_division_teams[CLUBS_KEY_COLUMN_NAME] == key
                ][CLUBS_NAME_COLUMN_NAME].iloc[0]

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


@section_printing_decorator
def show_basic_facts(
    corpus_without_emoticons,
    corpus_with_emoticons,
    n_latest_tweets_used_for_result_collection: int = None,
):
    print("2. BASIC FACTS")
    print()
    facts_extractor = FactsExtractor(
        corpus_without_emoticons=corpus_without_emoticons,
        corpus_with_emoticons=corpus_with_emoticons,
        n_latest_tweets_used_for_result_collection=n_latest_tweets_used_for_result_collection,
    )
    facts_extractor.show_basic_facts()
