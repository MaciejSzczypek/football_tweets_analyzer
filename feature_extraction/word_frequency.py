from typing import List, Dict


def count_word_occurences(text: List[str]) -> Dict:
    word_occurences = dict()
    for word in text:
        if word in word_occurences:
            word_occurences[word] += 1
        else:
            word_occurences[word] = 1
    return word_occurences
