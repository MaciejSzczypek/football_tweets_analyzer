import itertools
import nltk
from nltk import word_tokenize
from gensim import corpora, models

STOPWORDS = nltk.corpus.stopwords.words("english")


def _get_ngrams(sequence, n_gram_length):
    return list(zip(*(sequence[index:] for index in range(n_gram_length))))


def get_top_ngrams(normalized_corpus, ngram_length=1, ngrams_limit=10):
    flattened_corpus = " ".join(" ".join(doc) for doc in normalized_corpus)
    tokens = word_tokenize(flattened_corpus)
    ngrams = _get_ngrams(tokens, ngram_length)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(
        ngrams_freq_dist.items(), key=lambda ngram_tuple: ngram_tuple[1], reverse=True
    )
    sorted_ngrams = sorted_ngrams_fd[:ngrams_limit]
    sorted_ngrams = [(" ".join(text), freq) for text, freq in sorted_ngrams]
    return sorted_ngrams


def get_chunks(
    sentences, grammar=r"NP: {<DT>? <JJ>* <NN.*>+}", stopwords=STOPWORDS,
):
    all_chunks = []
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    for sentence in sentences:
        _sentence = " ".join([word for word in sentence])
        tagged_sents = [nltk.pos_tag(nltk.word_tokenize(_sentence))]
        chunks = [chunker.parse(tagged_sent) for tagged_sent in tagged_sents]
        wtc_sents = [nltk.chunk.tree2conlltags(chunk) for chunk in chunks]
        flattened_chunks = list(itertools.chain.from_iterable(wtc_sents))
        valid_chunks_tagged = [
            (status, [wtc for wtc in chunk])
            for status, chunk in itertools.groupby(
                flattened_chunks, lambda word_pos_chunk: word_pos_chunk[2] != "0"
            )
        ]
        valid_chunks = [
            " ".join(
                word.lower()
                for word, tag, chunk in wtc_group
                if word.lower() not in stopwords
            )
            for status, wtc_group in valid_chunks_tagged
            if status
        ]
        all_chunks.append(valid_chunks)
    return all_chunks


def get_tfidf_weighted_keyphrases(
    sentences, grammar=r"NP: {<DT>? <JJ>* <NN.*>+}", top_n=15,
):
    valid_chunks = get_chunks(sentences, grammar)
    dictionary = corpora.Dictionary(valid_chunks)
    corpus = [dictionary.doc2bow(chunk) for chunk in valid_chunks]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    weighted_phrases = {
        dictionary.get(idx): value for doc in corpus_tfidf for idx, value in doc
    }
    weighted_phrases = sorted(
        weighted_phrases.items(),
        key=lambda phrase_tuple: phrase_tuple[1],
        reverse=True,
    )
    weighted_phrases = [(term, round(wt, 3)) for term, wt in weighted_phrases]
    return weighted_phrases[:top_n]
