import logging
import argparse

from my_tokenize import initialize_tokenizer
from data_setup import load_dataset
from my_tokenize import tokenize_function, group_texts
from data_collator_setup import initialize_data_collator, demonstrate_data_collator, compute_token_importance
from model_trainer import initialize_model_and_trainer, downsample_dataset, train_and_evaluate


def main():

    # Set up argument parser
    # =============================================================================
    parser = argparse.ArgumentParser(description="Run the model training and evaluation")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to data (a directory containing {test, train, validation}.json (default: ./data)")
    parser.add_argument("--model-checkpoint", type=str, default="bert-base-uncased", help="Model name (or path) of huggingface model (default: bert-base-uncased)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training/Evaluation batch size (Default: 16)")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs (Default: 3)")
    # parser.add_argument("--scoring", type=str, default="", help="Scoring function to use for masking tokens (Default: None)")
    parser.add_argument("--version", type=int, default=1, help="Model version (Default: 1)")
    parser.add_argument("--output-dir", type=str, default=None, help="(default: models/{model_checkpoint}-baseline-v{version})")
    parser.add_argument("--num-example", type=int, default=None, help="Number of example to load (for debugging purposes) (default: None)")

    # parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    # 
    # Parse arguments
    args = parser.parse_args()
    # =============================================================================

    data_dir = args.data_path  # Specify your data directory
    model_checkpoint = args.model_checkpoint
    version = args.version
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_name = model_checkpoint.split("/")[-1]
        output_dir = f"models/{model_name}-baseline-v{version}"
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_example = args.num_example

    logging.basicConfig(level=logging.INFO)

    # =================================================================
    # Pre processing data
    # =================================================================
    datasets = load_dataset(data_dir, num_example)

    tokenizer = initialize_tokenizer(model_checkpoint=model_checkpoint)
    
    logging.info("====================================================================")
    logging.info("Tokenizing")
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        tokenized_datasets[split] = dataset.map(
            lambda examples: tokenize_function(tokenizer, examples),
            batched=True,
            remove_columns=['id', 'text', 'sector', 'descriptor', 'year', '__index_level_0__']
        )


    demonstrate_data_collator(
        data_collator, tokenized_datasets['train'],
        tokenizer, num_examples=3
    )



    # Group texts
    chunk_size = 128  # Specify the chunk size
    logging.info("====================================================================")
    logging.info("Split documents into {chunk_size} tokens")
    for split in tokenized_datasets.keys():
        tokenized_datasets[split] = tokenized_datasets[split].map(
            lambda examples: group_texts(chunk_size, examples),
            batched=True
        )

    # Example output
    logging.info("====================================================================")
    logging.info("Example of loaded document")
    logging.info("====================================================================")
    logging.info(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
    logging.info(tokenizer.decode(tokenized_datasets["train"][0]["labels"]))

    # Initialize and demonstrate Data Collator
    data_collator = initialize_data_collator(tokenizer)
    demonstrate_data_collator(data_collator, tokenized_datasets, tokenizer)


    # =================================================================
    # Training model
    # =================================================================
    # Initialize the trainer with the model, datasets, data collator, and command line parameters
    trainer = initialize_model_and_trainer(
        tokenized_datasets, 
        data_collator, 
        tokenizer, 
        model_checkpoint=model_checkpoint, 
        batch_size=batch_size, 
        num_epochs=num_epochs,
        output_dir=output_dir
    )
    # Add training code here if needed
    eval_results = train_and_evaluate(trainer)

if __name__ == "__main__":
    main()

