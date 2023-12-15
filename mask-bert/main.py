import logging
import argparse

from my_tokenize import initialize_tokenizer
from data_setup import load_dataset
from my_tokenize import tokenize_function, pretokenize_function, group_texts
from data_collator_setup import initialize_data_collator, demonstrate_data_collator, compute_token_importance
from model_trainer import initialize_model_and_trainer, downsample_dataset, train_and_evaluate
from data_collator_setup import create_tfidfscoring_function, create_idfscoring_function, create_termscoring_function

masking_strategies = {
    'tfidf': create_tfidfscoring_function,
    'idf': create_idfscoring_function,
    'term': create_termscoring_function,
    'default': None,
}

def main():

    # Set up argument parser
    # =============================================================================
    parser = argparse.ArgumentParser(description="Run the model training and evaluation")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to data (a directory containing {test, train, validation}.json (default: ./data)")
    parser.add_argument("--model-checkpoint", type=str, default="bert-base-uncased", help="Model name (or path) of huggingface model (default: bert-base-uncased)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training/Evaluation batch size (Default: 16)")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs (Default: 3)")
    parser.add_argument("--mask-strategy", type=str, choices=masking_strategies, default=None, help="Scoring function to use for masking tokens (Default: None)")
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
    mask_strategy = args.mask_strategy

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

    # Adds 'pretokenized' column
    pre_tokenized_documents = datasets['train'].map(
        lambda examples: pretokenize_function(tokenizer, examples),
        batched=True,
        remove_columns=['text', 'sector', 'descriptor', 'year', '__index_level_0__']
    )

    logging.info("====================================================================")
    logging.info("Train masking strategy")
    # Initialize and demonstrate Data Collator
    data_collator, score_token = initialize_data_collator(
        masking_strategies,
        strategy=mask_strategy,
        collator_kwargs={'tokenizer': tokenizer, 'score_column': 'importance_weight'},
        score_kwargs={'docs': pre_tokenized_documents['pretokenized'], 'path': 'path/to/terms'}
    )


    # score_token = lambda w: [.9] + [.1/(len(w)-1)] * (len(w)-1)


    logging.info("====================================================================")
    logging.info("Compute importance weights")
    # Compute words masking weights (stored in 'importance_weights')
    for split in tokenized_datasets.keys():
        tokenized_datasets[split] = tokenized_datasets[split].map(
            lambda example: compute_token_importance(example, tokenizer, score_token),
            batched=False,
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
            lambda examples: group_texts(examples, chunk_size),
            batched=True,
        )

    # TODO: what does group_texts do with the importance_score and the words etc... ?
    # Why don't they match because they matched before

    demonstrate_data_collator(
        data_collator, tokenized_datasets['train'],
        tokenizer, num_examples=3
    )

    exit()

    # Example output
    logging.info("====================================================================")
    logging.info("Example of loaded document")
    logging.info("====================================================================")
    logging.info(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
    logging.info(tokenizer.decode(tokenized_datasets["train"][1]["input_ids"]))


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

