from transformers import T5Tokenizer, T5ForConditionalGeneration


def get_t5_model(checkpoint='t5-small'):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.cuda()
    return tokenizer, model
