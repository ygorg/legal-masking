import logging
import argparse
import os
from my_tokenize import initialize_tokenizer
from data_setup import load_custom_dataset
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

    def arguments():
        # Set up argument parser
        # =============================================================================
        parser = argparse.ArgumentParser(description="Run the model training and evaluation")
        parser.add_argument("--data-path", type=str, default="./data", help="Path to data (a directory containing {test, train, validation}.json (default: ./data)")
        parser.add_argument("--model-checkpoint", type=str, default="bert-base-uncased", help="Model name (or path) of huggingface model (default: bert-base-uncased)")
        parser.add_argument("--batch-size", type=int, default=16, help="Training/Evaluation batch size (Default: 16)")
        parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs (Default: 3)")
        parser.add_argument("--mask-strategy", type=str, choices=masking_strategies, default=None, help="Scoring function to use for masking tokens (Default: None)")
        # parser.add_argument("--version", type=int, default=1, help="Model version (Default: 1)")
        parser.add_argument("--cache_dir", type=str, default=None, help="Chemin du dossier de cache où les données temporaires seront stockées. Si non spécifié, utilise le cache par défaut.")
        parser.add_argument("--num_workers", type=int, default=1, help="Chemin du dossier de cache où les données temporaires seront stockées. Si non spécifié, utilise le cache par défaut.")
        parser.add_argument("--load_cache_dir", type=bool, default=True, help="Chemin du dossier de cache où les données temporaires seront stockées. Si non spécifié, utilise le cache par défaut.")
        parser.add_argument("--output-dir", type=str, default=None, help="(default: models/{model_checkpoint}-baseline-v{version})")
        parser.add_argument("--num-example", type=int, default=None, help="Number of example to load (for debugging purposes) (default: None)")
        

        # parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
        # 
        # Parse arguments
        return parser.parse_args()
    # =============================================================================

    args = arguments()

    data_dir = args.data_path  # Specify your data directory
    root_path = os.environ['DSDIR'] + '/HuggingFace_Models'
    model_checkpoint = root_path+'/' + args.model_checkpoint
    # version = args.version

    num_epochs = args.num_epochs
    num_example = args.num_example
    mask_strategy = args.mask_strategy
    batch_size = args.batch_size
    cache_dir = args.cache_dir
    load_cache_dir = args.load_cache_dir

    chunk_size = 500  # Split the documents every CHUNK_SIZE tokens

    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_name = model_checkpoint.split("/")[-1]
        output_dir = f"../saved_models/{model_name}-e{num_epochs}-b{batch_size}-{mask_strategy}"
        if num_example:
            output_dir += f"-DEBUG{num_example}"


    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        # Log to a file and to the terminal
        handlers=[logging.FileHandler("../masking_bert.log", mode='a'),
                  stream_handler],
    )

    # =================================================================
    # Pre processing data
    # =================================================================
    datasets = load_custom_dataset(data_dir, num_example)

    tokenizer = initialize_tokenizer(model_checkpoint=model_checkpoint)
    
    logging.info("====================================================================")
    logging.info("Pre-Tokenizing")

    # Pretokenize documents for training the masking strategy
    pre_tokenized_documents = datasets['train'].map(
        lambda examples: pretokenize_function(tokenizer, examples),
        batched=True,
        remove_columns=['text', 'sector', 'descriptor', 'year', '__index_level_0__'],
	cache_file_name = f"{cache_dir}/pre-tokenization/{model_name}-e{num_epochs}-b{batch_size}-{mask_strategy}-ex{num_example if num_example else '4GB'}-Train.arrow",
    load_from_cache_file= load_cache_dir,
    num_proc=args.num_workers
    )

    logging.info("====================================================================")
    logging.info("Train masking strategy (example: compute TfIdf on whole documents)")
    


    if mask_strategy != 'default':
        if mask_strategy in ['term']:
            if True:
                score_token = masking_strategies[mask_strategy](path = 'path/to/terms')

        else:
            if True:
                score_token = masking_strategies[mask_strategy](docs =  pre_tokenized_documents['pretokenized'])
    else:
        score_token = None
    print(f"Score:\n{score_token}")



    # Initialize and demonstrate Data Collator
    data_collator = initialize_data_collator(
        masking_strategies,
        strategy=mask_strategy,
        collator_kwargs={'tokenizer': tokenizer, 'score_column': 'importance_weight'}
       # score_kwargs={'docs': pre_tokenized_documents['pretokenized'], 'path': 'path/to/terms'}
    )

    logging.info("====================================================================")
    logging.info("Tokenizing")
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        tokenized_datasets[split] = dataset.map(
            lambda examples: tokenize_function(tokenizer, examples),
            batched=True,
            remove_columns=['id', 'text', 'sector', 'descriptor', 'year', '__index_level_0__'],
            cache_file_name = f"{cache_dir}/tokenization/{model_name}-e{num_epochs}-b{batch_size}-{mask_strategy}-ex{num_example if num_example else '4GB'}-{split}.arrow",
            load_from_cache_file= load_cache_dir,
            num_proc=args.num_workers

        )


    # =================================================================
    # Computing word importance scores
    # =================================================================


    if mask_strategy != 'default':
        logging.info("====================================================================")
        logging.info("Compute importance weights according to masking strategy")
        # Compute words masking weights (stored in 'importance_weights')
        for split in tokenized_datasets.keys():
            tokenized_datasets[split] = tokenized_datasets[split].map(
                lambda example: compute_token_importance(example, tokenizer, score_token),
                batched=False,
                cache_file_name = f"{cache_dir}/tokenization-strategy/{model_name}-e{num_epochs}-b{batch_size}-{mask_strategy}-ex{num_example if num_example else '4GB'}-{split}.arrow",
                load_from_cache_file= load_cache_dir,
                num_proc=args.num_workers
            )


    # =================================================================
    # Spliting documents
    # =================================================================
    logging.info("====================================================================")
    logging.info(f"Split documents every {chunk_size} tokens")
    for split in tokenized_datasets.keys():
        tokenized_datasets[split] = tokenized_datasets[split].map(
            lambda examples: group_texts(
                examples, chunk_size, split_importance_weights=mask_strategy != 'default'
            ),
            batched=True,
            cache_file_name = f"{cache_dir}/tokenization-split/{model_name}-e{num_epochs}-b{batch_size}-{mask_strategy}-ex{num_example if num_example else '4GB'}-{split}.arrow",
            load_from_cache_file= load_cache_dir,
            num_proc=args.num_workers
        )

    demonstrate_data_collator(
        data_collator, tokenized_datasets['train'],
        tokenizer, num_examples=3
    )

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

