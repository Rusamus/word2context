import argparse
from dataset import prepare_data_for_t5, CustomDataset
from model import get_t5_model
from train import get_trainer
from utils import (load_data_from_csv,
                   save_word_context_pairs_to_csv)
from word_context_pairs import generate_word_context_pairs
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def main(args):
    if args.mode == "pipeline":
        # Generate and save word-context pairs
        word_context_pairs = generate_word_context_pairs()
        file_path = 'data/word_context_pairs_1k_words_10contexts.csv'
        save_word_context_pairs_to_csv(word_context_pairs, file_path)

    elif args.mode == "train":
        file_path = args.file_path

    else:
        print(f"Invalid mode: {args.mode}")
        return

    # Load data
    tokenizer, model = get_t5_model()
    pairs_data = load_data_from_csv(file_path)
    prepared_data = prepare_data_for_t5(pairs_data)
    train_data, val_data = train_test_split(prepared_data, test_size=0.1)
    train_dataset = CustomDataset(train_data, tokenizer)
    val_dataset = CustomDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Train the model
    trainer = get_trainer(model, train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mode selection')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['pipeline', 'train'],
                        help='Choose mode: "pipeline" or "train"')
    parser.add_argument('--file_path', type=str,
                        help='Path to the CSV file containing word-context pairs')
    args = parser.parse_args()

    main(args)
