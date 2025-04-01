from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

from utils.printing import section_printing_decorator, new_line_appendix_decorator

# Constants
DEFAULT_GRAMMAR = r"NP: {<DT>? <JJ>* <NN.*>+}"
STOPWORDS = stopwords.words("english")
INVALID_CHUNK_TAG = "0"


class KeyPhraseExtractor:

    @staticmethod
    def _generate_ngrams(sequence, n_gram_length):
        return list(zip(*(sequence[index:] for index in range(n_gram_length))))

    @classmethod
    def _get_sorted_ngram_frequencies(cls, tokens, ngram_length, ngrams_limit):
        ngrams = cls._generate_ngrams(tokens, ngram_length)
        ngrams_freq_dist = FreqDist(ngrams)
        sorted_ngrams_freq = sorted(
            ngrams_freq_dist.items(), key=lambda ngram_tuple: ngram_tuple[1], reverse=True
        )[:ngrams_limit]
        return [(" ".join(text), freq) for text, freq in sorted_ngrams_freq]

    @classmethod
    def get_top_ngrams(cls, normalized_corpus, ngram_length=1, ngrams_limit=10):
        flattened_corpus = " ".join(" ".join(doc) for doc in normalized_corpus)
        tokens = word_tokenize(flattened_corpus)
        return cls._get_sorted_ngram_frequencies(tokens, ngram_length, ngrams_limit)

    @classmethod
    @new_line_appendix_decorator
    def print_ngrams_with_the_biggest_count(cls, corpus, ngram_length: int, n_top_ngrams: int):
        top_ngrams = cls.get_top_ngrams(
            corpus, ngram_length=ngram_length, ngrams_limit=n_top_ngrams
        )
        print(f"Top {n_top_ngrams} {ngram_length}-grams:")
        rows = [(ngram[0], ngram[1]) for ngram in top_ngrams]
        for index, (ngram, count) in enumerate(rows, start=1):
            print(f"\t{index}. '{ngram}' [{count}]")

    @classmethod
    @section_printing_decorator("TOP N-GRAMS")
    def print_top_ngrams(cls, corpus, n_top_ngrams: int = 10, longest_n_gram_size: int = 5):
        for i in range(longest_n_gram_size):
            cls.print_ngrams_with_the_biggest_count(corpus=corpus, ngram_length=i + 1, n_top_ngrams=n_top_ngrams)

    @staticmethod
    def print_top_words_for_all_topics(model, feature_names, n_top_words):
        top_words_for_topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[: -n_top_words - 1: -1]]
            print(f"Topic #{topic_idx}: {' '.join(top_words)}")
            top_words_for_topics.append(top_words)
