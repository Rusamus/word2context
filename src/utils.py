import pandas as pd
import csv
import torch

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    pairs = list(df[['word', 'context']].itertuples(index=False, name=None))
    return pairs

def generate_sentence(input_word, model, tokenizer):
    input_text = f"generate sentence with a word: {input_word}"
    input_encoding = tokenizer(input_text, return_tensors='pt')
    input_encoding['input_ids'] = input_encoding['input_ids'].cuda()
    input_encoding['attention_mask'] = input_encoding['attention_mask'].cuda()
    
    with torch.no_grad():
        output = model.generate(**input_encoding)
    
    generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_sentence

def save_word_context_pairs_to_csv(word_context_pairs, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["word", "context"])
        writer.writerows(word_context_pairs)

# def generate_sentence(input_word, model, tokenizer):
#     input_text = f"generate sentence with {input_word} in it"
#     input_tokenized = tokenizer.encode(input_text, return_tensors="pt")
#     output_tokenized = model.generate(input_tokenized, max_length=30, num_return_sequences=1)
#     output_text = tokenizer.decode(output_tokenized[0], skip_special_tokens=True)
#     return output_text
