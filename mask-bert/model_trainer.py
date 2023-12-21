# model_trainer.py
import math
import logging

import torch
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer

def initialize_model_and_trainer(
        tokenized_datasets, data_collator, tokenizer,
        output_dir, model_checkpoint,
        batch_size, num_epochs
    ):

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    logging.info("=========================================================")
    logging.info(f"Using device: {device}")
    model.to(device)
    logging_steps = len(tokenized_datasets["train"]) // batch_size

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # Specify the output directory
        evaluation_strategy="epoch",
        num_train_epochs= num_epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,  # Set to False unless you also want to push to Hugging Face's Model Hub
        # fp16=True, # when we use cuda
        logging_steps=logging_steps,
        remove_unused_columns=False,  # In order to keep importance_weight column
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    return trainer

def downsample_dataset(tokenized_datasets, train_size=100):
    test_size = int(0.1 * train_size)
    downsampled_dataset = tokenized_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )
    return downsampled_dataset


def train_and_evaluate(trainer):
    # Perform training
    trainer.train()

    # Perform evaluation after training
    eval_results = trainer.evaluate()
    logging.info(f">>> Perplexity after training: {math.exp(eval_results['eval_loss']):.2f}")

    return eval_results
