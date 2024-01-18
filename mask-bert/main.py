import logging
import argparse

from my_tokenize import initialize_tokenizer
from data_setup import load_dataset
from my_tokenize import tokenize_function, pretokenize_function, group_texts
from data_collator_setup import initialize_data_collator, initialize_scoring_function, demonstrate_data_collator, compute_token_importance
from model_trainer import initialize_model_and_trainer, downsample_dataset, train_and_evaluate

masking_strategies = [
    'tfidf', 'idf', 'term', 'default'
]

def main():

    def arguments():
        # Set up argument parser
        # =============================================================================
        parser = argparse.ArgumentParser(description="Run the model training and evaluation")
        parser.add_argument("--data-path", type=str, default="./data", help="Path to data (a directory containing {test, train, validation}.json (default: ./data)")
        parser.add_argument("--model-checkpoint", type=str, default="bert-base-uncased", help="Model name (or path) of huggingface model (default: bert-base-uncased)")
        parser.add_argument("--batch-size", type=int, default=16, help="Training/Evaluation batch size (default: 16)")
        parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs (default: 3)")
        parser.add_argument("--mask-strategy", type=str, choices=masking_strategies, default='default', help="Scoring function to use for masking tokens (default: default)")
        parser.add_argument("--term-path", type=str, default=None, help="Path to list of terms (one term per line)")
        # parser.add_argument("--version", type=int, default=1, help="Model version (Default: 1)")
        parser.add_argument("--output-dir", type=str, default=None, help="Directory to save model's checkpoints (default: models/{model_name}-e{num_epochs}-b{batch_size}-{mask_strategy})")
        parser.add_argument("--num-example", type=int, default=None, help="Number of example to load (for debugging purposes) (default: all)")

        # parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
        # 
        # Parse arguments
        return parser.parse_args()
    # =============================================================================

    args = arguments()

    data_dir = args.data_path  # Specify your data directory
    model_checkpoint = args.model_checkpoint
    # version = args.version

    num_epochs = args.num_epochs
    num_example = args.num_example
    mask_strategy = args.mask_strategy
    batch_size = args.batch_size

    chunk_size = 128  # Split the documents every CHUNK_SIZE tokens

    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_name = model_checkpoint.split("/")[-1]
        output_dir = f"models/{model_name}-e{num_epochs}-b{batch_size}-{mask_strategy}"
        if num_example:
            output_dir += f"-DEBUG{num_example}"


    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        # Log to a file and to the terminal
        handlers=[logging.FileHandler("masking_bert.log", mode='a'),
                  stream_handler],
    )

    # =================================================================
    # Pre processing data
    # =================================================================
    datasets = load_dataset(data_dir, num_example)

    tokenizer = initialize_tokenizer(model_checkpoint=model_checkpoint)
    
    logging.info("====================================================================")
    logging.info("Pre-Tokenizing")

    # Pretokenize documents for training the masking strategy
    pre_tokenized_documents = datasets['train'].map(
        lambda examples: pretokenize_function(tokenizer, examples),
        batched=True,
        remove_columns=['text', 'sector', 'descriptor', 'year', '__index_level_0__']
    )

    logging.info("====================================================================")
    logging.info(f"Initialize data collator strategy ({mask_strategy})")
    # Initialize and demonstrate Data Collator

    data_collator = initialize_data_collator(
        mask_strategy, tokenizer, 'importance_weight',
    )

    logging.info("====================================================================")
    logging.info(f"Tokenizing")

    tokenized_datasets = {}
    for split, dataset in datasets.items():
        tokenized_datasets[split] = dataset.map(
            lambda examples: tokenize_function(tokenizer, examples),
            batched=True,
            remove_columns=['id', 'text', 'sector', 'descriptor', 'year', '__index_level_0__']
        )


    # =================================================================
    # Computing word importance scores
    # =================================================================


    if mask_strategy != 'default':
        def doc_generator(dataset, batch_size=10000):
            # HG datasets are always on the hard drive, and are in memory
            #  on the fly. For training tf-idf we use SKlearn, in order to
            #  pass the documents to Sklearn just writing
            #  `pre_tokenized_documents['pretokenized']` loads all the
            #  documents in the RAM.
            # This function acts as a buffer for loading only 10000
            #  documents at a time.
            dataset_ = dataset.to_iterable_dataset().iter(batch_size)
            for i, batch in enumerate(dataset_):
                logging.info(f'Loaded {batch_size * i} documents of {len(dataset)}')
                for doc in batch['pretokenized']:
                    yield doc

        score_token = initialize_scoring_function(
            mask_strategy, doc_generator(pre_tokenized_documents), term_path
        )

        logging.info("====================================================================")
        logging.info("Compute importance weights according to masking strategy")

        def apply_to_batch(fct, batch, kwargs_fct={}):
            """Apply a function that deals with only one example
            to a batch
            
            Args:
                fct (Callable[dict[str, Any] -> dict[str, Any]]): takes a row of a datasets.Dataset and returns a new row
                batch (dict[str, list[Any]]): a batch of examples
                kwargs_fct (dict[str, Any]): arguments for `fct`
            Returns:
                dict[str, list[Any]] a processed batch of rows from a datasets.Dataset

            """
            doc = {k: batch[k][0] for k in batch}
            new_doc = fct(doc, **kwargs_fct)

            new_batch = {k: [v] for k, v in new_doc.items()}

            batch_len = len(batch[list(batch.keys())[0]])
            for i in range(1, batch_len):
                doc = {k: batch[k][i] for k in batch}
                new_doc = fct(doc, **kwargs_fct)
                for k, v in new_doc.items():
                    new_batch[k].append(v)

            return new_batch

        # Compute words masking weights (stored in 'importance_weights')
        for split in tokenized_datasets.keys():
            tokenized_datasets[split] = tokenized_datasets[split].map(
                lambda examples: apply_to_batch(compute_token_importance, examples, kwargs_fct={'tokenizer': tokenizer, 'score_token': score_token}),
                batched=True,
                batch_size=10000,
                num_proc=4,
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

