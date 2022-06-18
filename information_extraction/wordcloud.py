import numpy as np
import matplotlib.pyplot as plt

def show_word_cloud(text, normalize_plurals=True):
    from wordcloud import WordCloud
    from PIL import Image
    import random

    # Create and generate a word cloud image:
    football_mask = np.array(Image.open("data/images/football_mask.png"))
    mask = np.array(Image.open("data/images/footballer_mask.jpg"))

    def color_func(*args, **kwargs):
        return f"rgb({random.randint(0, 230)}, {random.randint(0, 230)}, {random.randint(0, 230)})"

    wordcloud = WordCloud(
        background_color="white", mask=mask, random_state=1, normalize_plurals=normalize_plurals #color_func=color_func,
    ).generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    wordcloud.to_file("data/images/wordcloud.png")

