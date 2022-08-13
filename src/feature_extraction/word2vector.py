from gensim.models import Word2Vec


def get_trained_word_2_vector_model(normalized_corpus):
    w2v_model = Word2Vec(min_count=4, window=5, size=300, alpha=0.001, negative=20,)
    w2v_model.build_vocab(sentences=normalized_corpus)
    w2v_model.train(
        normalized_corpus, total_examples=w2v_model.corpus_count, epochs=30,
    )
    w2v_model.init_sims()
    return w2v_model.wv
    # for i, word in enumerate(w2v_model.wv.vocab):
    #     print(word, [t[0] for t in w2v_model.wv.similar_by_word(word, 10)])
    # print(w2v_model.wv.vocab)
