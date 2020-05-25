from corpus.cleaning.constants import FOOTBALL_RESULT_FORMAT_REGEX
import re


class CustomTokenizer:

    @classmethod
    def tokenize(cls, text: str):
        text = cls._normalize_football_match_result_if_occurs(text)
        return text.split()

    @classmethod
    def _normalize_football_match_result_if_occurs(
            cls,
            text: str
    ) -> str:
        match_result_search_outcome = re.search(
            f"{FOOTBALL_RESULT_FORMAT_REGEX}",
            text
        )
        if match_result_search_outcome:
            match_result = match_result_search_outcome.group(0)
            match_result = match_result.replace(":", "-")
            match_result = cls._remove_non_leading_and_non_trailing_whitespaces(match_result)
            text = re.sub(
                f"{FOOTBALL_RESULT_FORMAT_REGEX}",
                match_result,
                text,
            )
        return text

    @classmethod
    def _remove_non_leading_and_non_trailing_whitespaces(cls, text: str) -> str:
        new_text = ""
        leading_and_trailing_indexes = {0, len(text) - 1}
        for index, character in enumerate(text):
            if character == " " and index not in leading_and_trailing_indexes:
                continue
            new_text += character
        return new_text
