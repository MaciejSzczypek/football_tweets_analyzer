from typing import List, Optional, Set

from corpus.cleaning.constants import (
    BASIC_STOPWORDS,
    SPECIAL_CHARACTERS,
    CONTRACTION_EXTENSIONS,
)


class TextNormalizer:
    def __init__(self, stopwords: Optional[Set[str]]):
        self._stopwords = stopwords if stopwords else BASIC_STOPWORDS

    @property
    def stopwords(self) -> Set[str]:
        return self._stopwords

    def normalize(
        self,
        tokenized_text: List[str],
        remove_stopwords: bool,
        apply_stemming: bool,
        apply_lemmatization: bool,
        expand_contractions: bool,
        remove_special_characters: bool,
        remove_emoticons: bool,
    ) -> List[str]:
        normalized_and_tokenized_text = []
        for word in tokenized_text:
            word = self._lower(word)
            if self._word_should_be_removed(
                word=word,
                remove_special_characters=remove_special_characters,
                remove_stopwords=remove_stopwords,
            ):
                continue
            if remove_emoticons:
                word = self._remove_emoticons(word)
            if apply_stemming:
                word = self._stem(word)
            if apply_lemmatization:
                word = self._lemmatize(word)
            if expand_contractions:
                words = self._expand_contractions(word)
                normalized_and_tokenized_text.extend(words)
            else:
                normalized_and_tokenized_text.append(word)
        normalized_and_tokenized_text = [token for token in normalized_and_tokenized_text if token]
        return normalized_and_tokenized_text

    def _word_should_be_removed(
        self,
        word: str,
        remove_stopwords: bool,
        remove_special_characters: bool,
    ) -> bool:
        if word in self.stopwords and remove_stopwords:
            return True
        if word in SPECIAL_CHARACTERS and remove_special_characters:
            return True
        return False

    @classmethod
    def _lower(cls, token: str) -> str:
        return token.lower()

    @classmethod
    def _remove_emoticons(cls, token: str) -> str:
        return token.encode('ascii', 'ignore').decode('ascii')

    @classmethod
    def _stem(cls, token: str) -> str:
        return token

    @classmethod
    def _lemmatize(cls, token: str) -> str:
        return token

    @classmethod
    def _expand_contractions(cls, token: str) -> List[str]:
        if token in CONTRACTION_EXTENSIONS:
            extended_contraction = CONTRACTION_EXTENSIONS.get(token)
            words_after_expanding_contraction = extended_contraction.split()
        else:
            words_after_expanding_contraction = [token]
        return words_after_expanding_contraction
