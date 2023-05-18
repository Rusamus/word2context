import re
import nltk
from tqdm import tqdm
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from src.context_reverso import get_context

N_MOST_COMMON = 1000


def generate_word_context_pairs():
    nltk.download('brown')
    nltk.download('stopwords')
    nltk.download('punkt')

    words = brown.words()
    word_freq = FreqDist(words)
    stop_words = set(stopwords.words('english'))
    common_words = []

    for word, _ in tqdm(word_freq.most_common(), desc='Finding the most common words'):
        if len(common_words) >= N_MOST_COMMON:
            break
        if word.lower() not in stop_words and word.lower() not in common_words:
            common_words.append(word.lower())

    words = re.sub(r'[^\w\s]', '', ' '.join(common_words)).split()
    print(len(words))

    word_context_pairs = []
    for word in tqdm(words, desc='Generating word-context pairs'):
        contexts = None
        while contexts is None:
            try:
                contexts = get_context(word)
                word_context_pairs.extend([[word, ctx] for ctx in contexts])
            except KeyboardInterrupt:
                return word_context_pairs
            except:
                pass

    return word_context_pairs
