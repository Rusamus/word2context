from transformers import Trainer, TrainingArguments

def get_trainer(model, train_dataset, val_dataset):
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results_10k_samples',
        num_train_epochs=10,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=100,
        report_to="wandb"
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    return trainer
