from gensim.summarization.summarizer import summarize, summarize_corpus
from gensim.summarization import keywords, mz_keywords

from configs.config_loader import ConfigLoader
from corpus.cleaning.noisy_tweets_remover import NoisyTweetsRemover
from corpus.corpus_transformation_pipeline import CorpusTransformer
from data.column_names import LIV_WAT_TEXT_COLUMN_NAME
from data.loaders import DataLoader
from data.paths import LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
from information_extraction.keyphrase_extraction import get_top_ngrams
from information_extraction.text_summarization import TweetsSummarizer
from feature_extraction.tfidf import create_df_with_tfidf_feature_vectors
from information_extraction.answer_basic_questions import QuestionsAnswerer
from information_extraction.keyphrase_extraction import get_tfidf_weighted_keyphrases
from corpus.tokenizing.custom_tokenizers import CustomTokenizer


def run_liverpool_watford_analysis():
    configs = ConfigLoader.load()
    initial_df = DataLoader.from_csv(
        LIVERPOOL_VS_WATFORD_WITH_TWEET_SPECIFIC_NOISE_REMOVED_FILE_PATH
    )
    df = NoisyTweetsRemover.remove_noisy_tweets(initial_df)
    df.to_csv("/home/maciej_szczypek/PJATK/master_thesis/python_project/data/source_files/df_t")
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
    tfidf_df = create_df_with_tfidf_feature_vectors(corpus=separate_and_normalized_tweets)
    print(keywords(". ".join(flattened_and_normalized_corpus_text), scores=True))
    TweetsSummarizer.generate_tweets_summary_based_on_page_rank_and_random_sample(
        tfidf_tweets=tfidf_df,
        df_before_transformation=df,
        sentences=normalized_corpus,
    )
    flattened_more = ". ".join(separate_and_normalized_tweets[:1000])
    print(summarize(flattened_more, ratio=0.005, word_count=100))

    # top_unigrams = get_top_ngrams(normalized_corpus, ngram_length=1, ngrams_limit=10)
    top_bigrams = get_top_ngrams(normalized_corpus, ngram_length=2, ngrams_limit=5)
    top_trigrams = get_top_ngrams(normalized_corpus, ngram_length=3, ngrams_limit=5)
    top_quadgrams = get_top_ngrams(normalized_corpus, ngram_length=4, ngrams_limit=5)
    top_pentagrams = get_top_ngrams(normalized_corpus, ngram_length=5, ngrams_limit=5)
    top_hexagrams = get_top_ngrams(normalized_corpus, ngram_length=6, ngrams_limit=5)

    # print(top_unigrams)
    print(top_bigrams)
    print(top_trigrams)
    print(top_quadgrams)
    print(top_pentagrams)
    print(top_hexagrams)

    summary = TweetsSummarizer.generate_tweets_summary(
        tweets=normalized_corpus,
        top_n_grams=top_trigrams,
        threshold=100,
    )
    print(summary)
    # todo remove duplicated (probably bot-generated) tweets
    # todo who played, score, where, what happened
    # todo most relevant sentences
    # todo topic modeling
    # todo emotions, transfer learning


if __name__ == "__main__":
    run_liverpool_watford_analysis()
