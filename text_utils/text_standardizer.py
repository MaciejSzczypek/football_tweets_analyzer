import emoji


class TextNormalizer:
    CUSTOM_STOPWORDS = ["a", "an", "the"]

    @classmethod
    def normalize(
        cls,
        text: str,
        remove_stopwords: bool = False,
        apply_stemming: bool = False,
        apply_lemmatization: bool = False,
        expand_contractions: bool = False,
        remove_special_characters: bool = False,
        remove_emoticons: bool = False,
    ):
        text = cls._lower(text)
        if remove_stopwords:
            text = cls._remove_stopwords(text)
        if apply_stemming:
            text = cls._stem(text)
        if apply_lemmatization:
            text = cls._lemmatize(text)
        if expand_contractions:
            text = cls._expand_contractions(text)
        if remove_special_characters:
            text = cls._remove_special_characters(text)
        if remove_emoticons:
            text = cls._remove_emoticons(text)
        return text

    @classmethod
    def _lower(cls, text: str) -> str:
        return text.lower()

    @classmethod
    def _remove_stopwords(cls, text: str) -> str:
        return text

    @classmethod
    def _stem(cls, text: str) -> str:
        return text

    @classmethod
    def _lemmatize(cls, text: str) -> str:
        return text

    @classmethod
    def _expand_contractions(cls, text: str) -> str:
        return text

    @classmethod
    def _remove_special_characters(cls, text: str) -> str:
        return text

    @classmethod
    def _remove_emoticons(cls, text: str) -> str:
        return text

