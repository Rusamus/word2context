from model import get_t5_model
from utils import generate_sentence
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("Enter 'q' to quit the program.")
    while True:
        input_word = input("Enter a word: ")
        if input_word.lower() == 'q':
            break

        tokenizer, model = get_t5_model(checkpoint='/ssd/r.musaev/NLP_project/word2context/results/checkpoint-2700')
        sentence = generate_sentence(input_word, model, tokenizer)
        print(f"Generated sentence: {sentence}\n")
