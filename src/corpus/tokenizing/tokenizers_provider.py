from string import Template
from typing import Callable
from corpus.tokenizing.custom_tokenizers import CustomTokenizer
import nltk


class TokenizersProvider:
    NO_SUCH_KEY_IN_TOKENIZERS_DICTIONARY_INFO_TEMPLATE = Template(
        "Given name '${name}' has not been found in the tokenizers dictionary. "
        "Default tokenizer will be returned"
    )
    DEFAULT_SENTENCE_TOKENIZER_NAME = "default_sentence_tokenizer"
    DEFAULT_WORD_TOKENIZER_NAME = "default_word_tokenizer"
    TREEBANK_WORD_TOKENIZER_NAME = "treebank_word_tokenizer"
    TOK_TOK_WORD_TOKENIZER = "tok_tok_tokenizer"
    CUSTOM_TOKENIZER_NAME = "custom_tokenizer"
    TOKENIZERS = {
        DEFAULT_SENTENCE_TOKENIZER_NAME: nltk.sent_tokenize,
        DEFAULT_WORD_TOKENIZER_NAME: nltk.word_tokenize,
        TREEBANK_WORD_TOKENIZER_NAME: nltk.TreebankWordTokenizer().tokenize,
        TOK_TOK_WORD_TOKENIZER: nltk.ToktokTokenizer().tokenize,
        CUSTOM_TOKENIZER_NAME: CustomTokenizer.tokenize,
    }

    @classmethod
    def get_sentence_tokenizer(cls, name: str) -> Callable:
        return cls._get_default_tokenizer(
            name=name, default_tokenizer_name=cls.DEFAULT_SENTENCE_TOKENIZER_NAME
        )

    @classmethod
    def get_word_tokenizer(cls, name: str) -> Callable:
        return cls._get_default_tokenizer(
            name=name, default_tokenizer_name=cls.DEFAULT_WORD_TOKENIZER_NAME
        )

    @classmethod
    def _get_default_tokenizer(cls, name: str, default_tokenizer_name: str):
        if name in cls.TOKENIZERS:
            return cls.TOKENIZERS.get(name)
        else:
            return cls.TOKENIZERS.get(default_tokenizer_name)
