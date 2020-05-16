import re


class TweetTextExtractor:
    MENTIONED_USER_REGEX = r"@\w+ "
    TWEET_LINK_REGEX = r"https://t.co/\w+$"

    @classmethod
    def extract_text_from_tweet(cls, text: str) -> str:
        text = cls._remove_mentioned_users(text)
        text = cls._remove_reference_to_another_tweet(text)
        return text

    @classmethod
    def _remove_mentioned_users(cls, text: str) -> str:
        return re.sub(pattern=cls.MENTIONED_USER_REGEX, repl="", string=text)

    @classmethod
    def _remove_reference_to_another_tweet(cls, text: str) -> str:
        return re.sub(pattern=cls.TWEET_LINK_REGEX, repl="", string=text)
