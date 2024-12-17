import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import random
import os
from configs.config_schema import PathsConfig


def _color_randomly(*args, **kwargs):
    return f"rgb({random.randint(0, 230)}, {random.randint(0, 230)}, {random.randint(0, 230)})"

def generate_word_cloud(text: str, paths_config: PathsConfig, normalize_plurals = True, show = True, save = True):
    mask = np.array(Image.open(paths_config.wordcloud_mask))

    wordcloud = WordCloud(
        background_color="white",
        mask=mask,
        random_state=1,
        normalize_plurals=normalize_plurals,
        color_func=_color_randomly,
    ).generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    if show:
        plt.show()
    if save:
        path = os.path.join(paths_config.results_dir, "wordcloud.png")
        wordcloud.to_file(path)
        print(f"Wordcloud saved in {path}")


