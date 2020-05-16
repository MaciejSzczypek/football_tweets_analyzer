from enum import Enum

import nltk


class Tokenizer(Enum):
    default_sentence_tokenizer = nltk.sent_tokenize
    default_word_tokenizer = nltk.word_tokenize
    treebank_word_tokenizer = nltk.TreebankWordTokenizer().tokenize
    tok_tok_tokenizer = nltk.ToktokTokenizer().tokenize
