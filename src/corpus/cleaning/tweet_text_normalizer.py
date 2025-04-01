from typing import List, Optional, Set
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords
from corpus.cleaning.constants import (
    TWITTER_SPECIFIC_SPECIAL_CHARACTERS,
    CONTRACTION_EXTENSIONS,
    NORMALIZED_FOOTBALL_RESULT_FORMAT_REGEX,
)
import re


class TweetTextNormalizer:
    def __init__(self, stopwords: Optional[Set[str]] = None):
        self._stopwords = stopwords if stopwords else nltk_stopwords.words("english")

    @property
    def stopwords(self) -> Set[str]:
        return self._stopwords

    def normalize(
        self,
        tokenized_text: List[str],
        lower: bool,
        remove_commas: bool,
        remove_stopwords: bool,
        apply_stemming: bool,
        apply_lemmatization: bool,
        expand_contractions: bool,
        remove_special_characters: bool,
        remove_emoticons: bool,
        remove_hashtags: bool,
        remove_user_mentions: bool,
        remove_links: bool,
    ) -> List[str]:
        normalized_and_tokenized_text = []
        for word in tokenized_text:
            if remove_commas:
                word = word.replace(",", "")
            if lower:
                word = self._lower(word)
            if remove_emoticons:
                word = self._remove_emoticons(word)
            if remove_links:
                word = self._remove_links(word)
            if remove_hashtags:
                word = self._remove_hashtags(word)
            if remove_user_mentions:
                word = self._remove_user_mentions(word)
            if remove_special_characters:
                word = self._remove_special_characters(word)
            if word in self.stopwords and remove_stopwords:
                continue
            if apply_stemming:
                word = self._stem(word)
            if apply_lemmatization:
                word = self._lemmatize(word)
            if expand_contractions:
                words = self._expand_contractions(word)
                words = list(
                    filter(
                        lambda word_: not (
                            word_ in self.stopwords and remove_stopwords
                        ),
                        words,
                    )
                )
                normalized_and_tokenized_text.extend(words)
            else:
                normalized_and_tokenized_text.append(word)
        normalized_and_tokenized_text = [
            token for token in normalized_and_tokenized_text if token
        ]
        return normalized_and_tokenized_text

    @classmethod
    def _lower(cls, token: str) -> str:
        return token.lower()

    @classmethod
    def _remove_emoticons(cls, token: str) -> str:
        return token.encode("ascii", "ignore").decode("ascii")

    @classmethod
    def _remove_special_characters(cls, token: str) -> str:
        if re.match(NORMALIZED_FOOTBALL_RESULT_FORMAT_REGEX, token):
            return token
        return re.sub(f"[^a-zA-Z0-9{TWITTER_SPECIFIC_SPECIAL_CHARACTERS}]+", "", token)

    @classmethod
    def _remove_links(cls, token: str) -> str:
        if re.findall(r"https:\/\/.*", token):
            return ""
        else:
            return token

    @classmethod
    def _remove_hashtags(cls, token: str) -> str:
        return re.sub(r"^#.*$", "", token)

    @classmethod
    def _remove_user_mentions(cls, token: str) -> str:
        return re.sub(r"^@.*$", "", token)

    @classmethod
    def _stem(cls, token: str) -> str:
        return PorterStemmer().stem(token)

    @classmethod
    def _lemmatize(cls, token: str) -> str:
        return WordNetLemmatizer().lemmatize(token)

    @classmethod
    def _expand_contractions(cls, token: str) -> List[str]:
        if token in CONTRACTION_EXTENSIONS:
            extended_contraction = CONTRACTION_EXTENSIONS.get(token)
            words_after_expanding_contraction = extended_contraction.split()
        else:
            words_after_expanding_contraction = [token]
        return words_after_expanding_contraction
