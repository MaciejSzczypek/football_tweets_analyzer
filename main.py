from gensim.summarization.summarizer import summarize, summarize_corpus
from gensim.summarization import keywords, mz_keywords

from configs.config_loader import ConfigLoader
from corpus.cleaning.noisy_tweets_remover import NoisyTweetsRemover
from corpus.corpus_transformation_pipeline import CorpusTransformer
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME
from data.loaders import DataLoader
from gensim.models.lsimodel import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora
from data.paths import LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
from information_extraction.keyphrase_extraction import get_top_ngrams
from information_extraction.text_summarization import TweetsSummarizer
from feature_extraction.tfidf import create_df_with_tfidf_feature_vectors
from information_extraction.answer_basic_questions import QuestionsAnswerer
from information_extraction.keyphrase_extraction import get_tfidf_weighted_keyphrases
from corpus.tokenizing.custom_tokenizers import CustomTokenizer

4
import re
from feature_extraction.vectorizers import transform_matrix_with_count_vectorizer


def run_liverpool_watford_analysis():
    configs = ConfigLoader.load()
    initial_df = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
    )
    df = NoisyTweetsRemover.remove_noisy_tweets(initial_df)
    tweets = df[LIV_WAT_TEXT_COLUMN_NAME].to_numpy()
    corpus_transformation_result = CorpusTransformer.transform_twitter_corpus(
        corpus=tweets, hyper_parameters_config=configs.settings["setting_1"],
    )
    normalized_corpus = corpus_transformation_result.corpus
    df = df[~df.index.isin(df.iloc[corpus_transformation_result.indexes_of_removed_tweets].index)]
    flattened_and_normalized_corpus_text = []
    separate_and_normalized_tweets = []
    for tweet in normalized_corpus:
        flattened_and_normalized_corpus_text.extend(tweet)
        separate_and_normalized_tweets.append(" ".join(tweet))

    tokenized_tweets = [CustomTokenizer.tokenize(tweet) for tweet in tweets]
    question_answerer = QuestionsAnswerer(corpus=tokenized_tweets)
    answered_questions = question_answerer.answer_basic_questions()
    # words_occurences = count_word_occurences(flattened_and_normalized_corpus_text)
    # tweets_scores = score_tweets(tfidf_tweets=normalized_corpus, words_occurence=words_occurences)
    # print(tweets_scores)
    tfidf_df = create_df_with_tfidf_feature_vectors(corpus=separate_and_normalized_tweets,
                                                    maximum_number_of_features=100)
    # print(keywords(". ".join(flattened_and_normalized_corpus_text), scores=True))
    # TweetsSummarizer.generate_tweets_summary_based_on_page_rank_and_random_sample(
    #     tfidf_tweets=tfidf_df,
    #     df_before_transformation=df,
    #     sentences=normalized_corpus,
    # )
    # flattened_more = ". ".join(separate_and_normalized_tweets[:1000])
    # print(summarize(flattened_more, ratio=0.005, word_count=100))

    # # top_unigrams = get_top_ngrams(normalized_corpus, ngram_length=1, ngrams_limit=10)
    # top_bigrams = get_top_ngrams(normalized_corpus, ngram_length=2, ngrams_limit=5)
    # top_trigrams = get_top_ngrams(normalized_corpus, ngram_length=3, ngrams_limit=5)
    # top_quadgrams = get_top_ngrams(normalized_corpus, ngram_length=4, ngrams_limit=5)
    # top_pentagrams = get_top_ngrams(normalized_corpus, ngram_length=5, ngrams_limit=5)
    # top_hexagrams = get_top_ngrams(normalized_corpus, ngram_length=6, ngrams_limit=5)

    # print(top_unigrams)
    # print(top_bigrams)
    # print(top_trigrams)
    # print(top_quadgrams)
    # print(top_pentagrams)
    # print(top_hexagrams)

    # summary = TweetsSummarizer.generate_tweets_summary(
    #     tweets=normalized_corpus,
    #     top_n_grams=top_trigrams,
    #     threshold=100,
    # )
    # print(summary)

    # section topic modeling
    lda_dictionary = corpora.Dictionary(normalized_corpus)
    print(lda_dictionary.items())
    lda_prepared_corpus = [lda_dictionary.doc2bow(text) for text in normalized_corpus]
    print("before lda model")
    import random
    def show_top_words(model):
        for topic in model.print_topics(num_topics=5, num_words=15):
            words = re.findall("\".*?\"", topic[1])
            print([word.replace('"', '') for word in words])
            # print(topic[1])

    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from pprint import pprint
    from sklearn.preprocessing import normalize
    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
    data_samples = separate_and_normalized_tweets
    tfidf_vectorizer = TfidfVectorizer(token_pattern="\S+",
                                       stop_words='english',
                                       )
    tfidf = tfidf_vectorizer.fit_transform(data_samples)

    tf_vectorizer = CountVectorizer(
        token_pattern="\S+",
        stop_words='english'
    )
    tf = tf_vectorizer.fit_transform(data_samples)
    for i in range(10):
        max_iter = random.randint(1, 5000)
        n_of_topics = random.randint(2, 5)
        alpha = random.random()
        l1_ratio = random.random()
        print(f"NMF: alpha: {alpha}, l1_ratio: {l1_ratio}, max_iter: {max_iter}")
        nmf = NMF(
            n_components=n_of_topics,
            alpha=alpha,
            l1_ratio=l1_ratio,
        ).fit(tfidf)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        print_top_words(nmf, tfidf_feature_names, 15)
    # for i in range(10):
    #     n_of_topics = random.randint(1, 5)
    #     max_iter = random.randint(1, 5000)
    #     learning_offset = random.randint(1, 1000)
    #     evaluate_every = random.randint(1, 200)
    #     print(f"LDA: learning_offset: {learning_offset}, evaluate_every: {evaluate_every}, max_iter: {max_iter}")
    #     lda = LatentDirichletAllocation(
    #         n_components=n_of_topics,
    #         max_iter=5,
    #         learning_method='online',
    #         learning_offset=50.,
    #     )
    #     lda.fit(tf)
    #     tf_feature_names = tf_vectorizer.get_feature_names()
    #     print_top_words(lda, tf_feature_names, 15)
    # for i in range(10):
    #     n_iter = random.randint(1, 15000)
    #     tol = random.random()
    #     print(f"LSI: n_iter: {n_iter}, tol: {tol}")
    #     lsi = TruncatedSVD(
    #         n_components=2, algorithm='randomized', n_iter=n_iter, tol=tol)
    #     lsi.fit(tf)
    #     tf_feature_names = tf_vectorizer.get_feature_names()
    #     print_top_words(lsi, tf_feature_names, 15)

    # todo Pachinko allocation, aggregated tweets and LDA
    return

    # lsi = LsiModel(
    #     corpus=lda_prepared_corpus,
    #     num_topics=2,
    #     id2word=lda_dictionary,
    #     chunksize=10000,
    #     power_iters=75,
    #     onepass=False,
    #     # extra_samples=1000,
    #     # workers=6,
    # )
    # show_top_words(lsi)
    # return
    def show_top_words_with_randomized_parameters():
        # random_state = random.randint(1, 100)
        passes = random.randint(50, 80)
        iterations = random.randint(10, 40)
        chunksize = 10000  # random.randint(3200, 3800)
        update_every = 5  # random.randint(10, 100)
        print("===========")
        print(
            f"passes: {passes}, iterations: {iterations}, chunksize: {chunksize}, update_every: {update_every}, random_state: ")
        lda_model = LdaMulticore(
            corpus=lda_prepared_corpus,
            num_topics=2,
            id2word=lda_dictionary,
            passes=passes,
            iterations=iterations,
            chunksize=chunksize,
            # random_state=random_state,
            # random_state=1,
            # update_every=update_every
            # workers=6,
        )
        show_top_words(lda_model)

    for i in range(10):
        show_top_words_with_randomized_parameters()

    # priority todo's
    # todo who played, score, where, mvp
    # todo topic modeling
    # todo emotions, transfer learning

    # less relevant for now:
    # todo improve most relevant sentences
    # todo improve summarization


if __name__ == "__main__":
    run_liverpool_watford_analysis()
