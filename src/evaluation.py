import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load models
t5_small = T5ForConditionalGeneration.from_pretrained("/ssd/r.musaev/NLP_project/word2context/results_10k_samples/checkpoint-600").cuda()
t5_base = T5ForConditionalGeneration.from_pretrained("/ssd/r.musaev/NLP_project/word2context/results_base_10k_samples/checkpoint-600").cuda()
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")

models = {
    "T5-small": (t5_small, tokenizer_t5),
    "T5-base": (t5_base, tokenizer_t5)
}

rouge_evaluator = Rouge()

def generate_sentence(model, tokenizer, prompt):
    random_prompt = np.random.choice([
        f"Create a sentence using the word {prompt} that showcases its usage in a common context.",
        f"Write a sentence that uses the word {prompt} in everyday language.",
        f"Formulate a sentence with the word {prompt} to demonstrate its typical usage.",
        f"Construct a sentence that includes the word {prompt} in a familiar context.",
        f"Compose a sentence that features the word {prompt} in a general setting.",
    ])
    prompt = random_prompt

    inputs = tokenizer.encode(prompt, return_tensors="pt").cuda()
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_bleu(reference, prediction):
    reference = [reference.split()]
    prediction = prediction.split()
    smooth = SmoothingFunction().method1
    return sentence_bleu(reference, prediction, smoothing_function=smooth)

def evaluate_meteor(reference, prediction):
    reference = reference.split()
    prediction = prediction.split()
    return meteor_score([reference], prediction)

def evaluate_rouge(reference, prediction):
    reference = [reference]
    prediction = prediction.split()
    if not reference or not prediction:
        return 0.0
    if len(reference[0].split()) != len(prediction):
        return 0.0
    scores = rouge_evaluator.get_scores(' '.join(prediction), ' '.join(reference[0].split()))
    return scores["rouge-l"]["f"]

# Example validation set with reference sentences and keywords
# validation_data = [
#     {"keyword": "apple", "reference": "She offered him a apple."},
#     {"keyword": "dog", "reference": "The dog barked loudly at the strangers."},
# ]

df = pd.read_csv('/ssd/r.musaev/NLP_project/word_context_pairs_1k_words_10contexts.csv')
validation_data = [
    {"keyword": df.iloc[item].word, "reference": df.iloc[item].context} for item in range(1, len(df))
]

results = []

for model_name, (model, tokenizer) in models.items():
    total_bleu = 0
    total_meteor = 0
    total_rouge_l = 0
    n = len(validation_data)
    for item in tqdm(validation_data):
        prompt = item["keyword"]
        reference = item["reference"]
        prediction = generate_sentence(model, tokenizer, prompt)

        bleu = evaluate_bleu(reference, prediction)
        meteor = evaluate_meteor(reference, prediction)
        rouge_l = 0 #evaluate_rouge(reference, prediction)
        
        total_bleu += bleu
        total_meteor += meteor
        total_rouge_l += rouge_l

    avg_bleu = total_bleu / n
    avg_meteor = total_meteor / n
    avg_rouge_l = total_rouge_l / n

    results.append({
        "model": model_name,
        "average BLEU": avg_bleu,
        "average METEOR": avg_meteor,
        "average ROUGE-L": avg_rouge_l,
    })

for result in results:
    print(f"Results for {result['model']}:")
    print(f"  Average BLEU: {result['average BLEU']:.4f}")
    print(f"  Average METEOR: {result['average METEOR']:.4f}")
    print(f"  Average ROUGE-L: {result['average ROUGE-L']:.4f}")
    print()
