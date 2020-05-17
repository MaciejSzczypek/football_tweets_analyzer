
class BasicSplitByWhitespaceTokenizer:

    @classmethod
    def tokenize(cls, text: str):
        return text.split()
