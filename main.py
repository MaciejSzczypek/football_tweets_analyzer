from gensim.models import Word2Vec

from data.loaders import DataLoader
from data.schemas import LIV_WAT_TEXT_COLUMN_NAME
from data.paths import (
    FINAL_LIVERPOOL_VS_WATFORD_WITH_RETWEETS_EXCLUDED_FILE_PATH,
)

df = DataLoader.from_csv(FINAL_LIVERPOOL_VS_WATFORD_WITH_RETWEETS_EXCLUDED_FILE_PATH)
tweets = df[LIV_WAT_TEXT_COLUMN_NAME]
print(tweets)

# w2v_model = Word2Vec(min_count=3, window=4, size=300, alpha=0.03, min_alpha=0.0007,)
# print(w2v_model)
