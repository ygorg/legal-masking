# model_trainer.py
import math
import logging

import torch
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer
import idr_torch

def initialize_model_and_trainer(
        tokenized_datasets, data_collator, tokenizer,
        output_dir, model_checkpoint,
        batch_size, num_epochs, logging_dir
    ):

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        # Macos
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)

    logging.info("=========================================================")
    logging.info(f"Using device: {device}")
    model.to(device)
    logging_steps = len(tokenized_datasets["train"]) // batch_size
    logging_steps = max(1, logging_steps)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # Specify the output directory
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        logging_steps = 50,
        logging_dir = logging_dir,
        log_level='info',
        save_total_limit = 20,
        load_best_model_at_end = True,
        num_train_epochs= num_epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to="tensorboard",
        push_to_hub=False,  # Set to False unless you also want to push to Hugging Face's Model Hub
        fp16=True, # when we use cuda
        gradient_accumulation_steps = 16,
        # logging_steps=logging_steps,
        remove_unused_columns=False,  # In order to keep importance_weight column
        use_mps_device=str(model.device).startswith('mps')  # For MacOS
    )

    training_args.local_rank = idr_torch.local_rank

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
