from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer

def prepare_data_for_t5(pairs):
    t5_data = []
    for input_word, sentence in pairs:
        input_text = f"generate sentence with: {input_word}"
        target_text = sentence
        t5_data.append({"input_text": input_text, "target_text": target_text})
    return t5_data


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data[idx]['input_text']
        target_text = self.data[idx]['target_text']
        input_encoding = self.tokenizer(input_text, return_tensors='pt', padding='max_length', max_length=64, truncation=True)
        target_encoding = self.tokenizer(target_text, return_tensors='pt', padding='max_length', max_length=64, truncation=True)
        return {'input_ids': input_encoding['input_ids'][0],
                'attention_mask': input_encoding['attention_mask'][0],
                'labels': target_encoding['input_ids'][0]}