from typing import List, Optional, Set

from corpus.cleaning.text_normalizer import TextNormalizer


class TokenizedCorpusNormalizer:
    def __init__(self, stopwords: Optional[Set[str]] = None):
        self._text_normalizer = TextNormalizer(stopwords=stopwords)

    def normalize(
        self,
        tokenized_corpus: List[List[str]],
        remove_stopwords: bool = False,
        apply_stemming: bool = False,
        apply_lemmatization: bool = False,
        expand_contractions: bool = False,
        remove_special_characters: bool = False,
        remove_emoticons: bool = False,
    ) -> List[List[str]]:
        return [
            self._text_normalizer.normalize(
                tokenized_text=document,
                remove_stopwords=remove_stopwords,
                apply_stemming=apply_stemming,
                apply_lemmatization=apply_lemmatization,
                expand_contractions=expand_contractions,
                remove_special_characters=remove_special_characters,
                remove_emoticons=remove_emoticons,
            )
            for document in tokenized_corpus
        ]
