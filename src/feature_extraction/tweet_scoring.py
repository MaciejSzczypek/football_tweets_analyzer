from typing import Dict, List


def score_tweets(tweets: List[List[str]], words_occurence: Dict[str, int]) -> Dict:
    tweets_scores = {}

    for tweet in tweets:
        tweet = tuple(tweet)
        words_count_in_tweet = len(tweets)
        tweet_score = 0
        for word in tweet:
            word_score = words_occurence.get(word)
            if word_score:
                tweet_score += word_score
        tweet_score = tweet_score / words_count_in_tweet
        if tweets_scores.get(tweet):
            tweets_scores[tweet] += tweet_score
        else:
            tweets_scores[tweet] = tweet_score

    return tweets_scores
