# model_trainer.py
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer
import math
import torch

def initialize_model_and_trainer(tokenized_datasets, data_collator, tokenizer,version=1,batch_size=8, num_epochs=3, model_checkpoint="models/bert-base-uncased"):
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("=========================================================")
    print(f"Using device: {device}")
    model.to(device)
    model_name = model_checkpoint.split("/")[-1]
    output_dir = f"models/{model_name}-baseline-v{version}"     
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
    print(f">>> Perplexity after training: {math.exp(eval_results['eval_loss']):.2f}")

    return eval_results
