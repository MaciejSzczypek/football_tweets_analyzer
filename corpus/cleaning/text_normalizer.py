from typing import List, Optional, Set

from corpus.cleaning.constants import BASIC_STOPWORDS, EMOTICONS, SPECIAL_CHARACTERS


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
                remove_emoticons=remove_emoticons,
                remove_stopwords=remove_stopwords,
            ):
                continue
            if apply_stemming:
                word = self._stem(word)
            if apply_lemmatization:
                word = self._lemmatize(word)
            if expand_contractions:
                word = self._expand_contractions(word)
            normalized_and_tokenized_text.append(word)
        return normalized_and_tokenized_text

    def _word_should_be_removed(
        self,
        word: str,
        remove_stopwords: bool,
        remove_special_characters: bool,
        remove_emoticons: bool,
    ) -> bool:
        if word in self.stopwords and remove_stopwords:
            return True
        if word in EMOTICONS and remove_emoticons:
            return True
        if word in SPECIAL_CHARACTERS and remove_special_characters:
            return True
        return False

    @classmethod
    def _lower(cls, text: str) -> str:
        return text.lower()

    @classmethod
    def _stem(cls, text: str) -> str:
        return text

    @classmethod
    def _lemmatize(cls, text: str) -> str:
        return text

    @classmethod
    def _expand_contractions(cls, text: str) -> str:
        return text
