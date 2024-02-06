import os
import logging
import argparse
import os
from my_tokenize import initialize_tokenizer
from data_setup import load_custom_dataset
from my_tokenize import tokenize_function, group_texts
from data_collator_setup import (
    initialize_data_collator, create_tfidfscoring_function,
    create_idfscoring_function, create_termscoring_function,
    demonstrate_data_collator
)
from model_trainer import initialize_model_and_trainer, downsample_dataset, train_and_evaluate
import torch

import idr_torch
import torch.distributed as dist

masking_strategies = [
    'tfidf', 'idf', 'term', 'default'
]

def doc_generator(dataset, column_name, batch_size=10000):
    # HG datasets are always on the hard drive, and are in memory on
    #  the fly. For training tf-idf we use SKlearn, in order to pass
    #  the documents to Sklearn just writing `dataset['column_name']`
    #  loads all the documents in the RAM.
    # This function acts as a buffer for loading only 10000
    #  documents at a time.

    for i, batch in enumerate(dataset.iter(batch_size)):
        logging.info(f'Loaded {batch_size * (i+1)} documents of {len(dataset)}')
        for doc in batch[column_name]:
            yield doc


def apply_to_batch(fct, batch, kwargs_fct={}):
    """Apply a function that deals with only one example to a batch
    
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


def main():

    def arguments():
        # Set up argument parser
        # =============================================================================
        parser = argparse.ArgumentParser(description="Run the model training and evaluation")
        parser.add_argument("--data-path", type=str, default="./data", help="Path to data (a directory containing {test, train, validation}.json (default: ./data)")
        parser.add_argument("--model-checkpoint", type=str, default="bert-base-uncased", help="Model name (or path) of huggingface model (default: bert-base-uncased)")

        # parser.add_argument("--batch-size", type=int, default=16, help="Training/Evaluation batch size (default: 16)")
        # parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs (default: 3)")
        parser.add_argument("--chunk-size", type=int, default=128, help="Split the documents into sequences of X tokens (default: 128)")

        parser.add_argument("--mask-strategy", type=str, choices=masking_strategies, default='default', help="Scoring function to use for masking tokens (default: default)")
        parser.add_argument("--term-path", type=str, default=None, help="Path to list of terms (one term per line)")

        parser.add_argument("--cache-dir", type=str, default=None, help="Directory to cache pretreatments (default: no cache)")
        parser.add_argument("--load-from-cache-file", choices=['true', 'false'], default='true', help="If False force recompute the cache (default: true)")
        parser.add_argument("--num-workers", type=int, default=1, help="Number of processes to use for pretreating data (default: 1)")
        # parser.add_argument("--output-dir", type=str, default=None, help="Directory to save model's checkpoints (default: models/{model_name}-e{num_epochs}-b{batch_size}-{mask_strategy})")
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

    # num_epochs = args.num_epochs
    num_example = args.num_example
    mask_strategy = args.mask_strategy
    # batch_size = args.batch_size

    term_path = args.term_path
    cache_dir = args.cache_dir
    load_from_cache_file = True if args.load_from_cache_file == 'true' else False
    num_workers = args.num_workers

    if mask_strategy != 'term':
        term_path = None
    if mask_strategy == 'term' and term_path == None:
        raise Exception('Masking strategy is "term" but not term list provided. Please provide a file to --term-path.')

    chunk_size = args.chunk_size  # Split the documents every CHUNK_SIZE tokens


    # Name of the masking strategy with the name of the term used if any
    mask_strat_for_cache = f'{mask_strategy}'
    if term_path is not None:
        mask_strat_for_cache += '_' + os.path.basename('.'.join(term_path.split('.')[:-1]))


    model_name = model_checkpoint.split("/")[-1]
    # if args.output_dir:
    #     output_dir = args.output_dir
    # else:
    #     output_dir = f"../saved_models/{model_name}-e{num_epochs}-b{batch_size}-c{chunk_size}-{mask_strat_for_cache}-ex{num_example if num_example else 'all'}"


    import datasets, os

    root_path = os.environ['DSDIR'] + '/HuggingFace_Models/'


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

    tokenizer = initialize_tokenizer(model_checkpoint= root_path + model_checkpoint if model_checkpoint == "bert-base-uncased" else model_checkpoint)

    # Making sure the cache directory exists
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(f"{cache_dir}/tokenization", exist_ok=True)
        os.makedirs(f"{cache_dir}/word_reconstructed", exist_ok=True)
        os.makedirs(f"{cache_dir}/tfidf", exist_ok=True)
        os.makedirs(f"{cache_dir}/tokenization-strategy", exist_ok=True)
        os.makedirs(f"{cache_dir}/tokenization-split", exist_ok=True)

        # Defining all cache file_name in order to check whether something was already done
        cache_fn_tfidf = f"{cache_dir}/tfidf/{model_name}-ex{num_example if num_example else 'all'}-train.joblib"
        cache_fn_tokenize = {}
        cache_fn_reconstructed = {}
        cache_fn_word_import = {}
        cache_fn_word_import_split = {}
        for split in datasets.keys():
            cache_fn_tokenize[split] = f"{cache_dir}/tokenization/{model_name}-ex{num_example if num_example else 'all'}-{split}.arrow"
            cache_fn_reconstructed[split] = f"{cache_dir}/word_reconstructed/{model_name}-ex{num_example if num_example else 'all'}-{split}.arrow"
            cache_fn_word_import[split] = f"{cache_dir}/tokenization-strategy/{model_name}-{mask_strat_for_cache}-ex{num_example if num_example else 'all'}-{split}.arrow"
            cache_fn_word_import_split[split] = f"{cache_dir}/tokenization-split/{model_name}-{mask_strat_for_cache}-c{chunk_size}-ex{num_example if num_example else 'all'}-{split}.arrow"
    else:
        logging.info('No cache will be used')
        cache_fn_tfidf = None
        cache_fn_tokenize = {split: None for split in datasets.keys()}
        cache_fn_reconstructed = {split: None for split in datasets.keys()}
        cache_fn_word_import = {split: None for split in datasets.keys()}
        cache_fn_word_import_split = {split: None for split in datasets.keys()}



    logging.info("====================================================================")
    logging.info(f"Initialize data collator strategy ({mask_strategy})")
    # Initialize and demonstrate Data Collator

    data_collator = initialize_data_collator(
        mask_strategy, tokenizer, 'importance_weight'
    )

    logging.info("====================================================================")
    logging.info(f"Tokenizing")

    tokenized_datasets = {}
    for split, dataset in datasets.items():
        tokenized_datasets[split] = dataset.map(
            lambda examples: tokenize_function(tokenizer, examples),
            batched=True,
            remove_columns=['id', 'text', 'sector', 'descriptor', 'year', '__index_level_0__'],
            cache_file_name=cache_fn_tokenize[split],
            load_from_cache_file=load_from_cache_file
        )
        logging.info(f'Cacheing to {cache_fn_tokenize[split]}')


    # =================================================================
    # Computing word importance scores
    # =================================================================


    if mask_strategy != 'default':

        logging.info("====================================================================")
        logging.info("Reconstructing words")

        def reconstruct_words(example, collator):
            # Apply the same treatment as in DataCollatorForWholeWordMasking

            # This function could cause problem for processing chinese with a tokenizer other than Bert
            ref_tokens = collator.get_ref_tokens(example)

            _, words = collator.reconstruct_words_from_subtokens(ref_tokens)
            example['reconstructed_words'] = words
            return example

        # Reconstruct words so we can compute weights of whole words instead of subtokens
        for split in tokenized_datasets.keys():
            tokenized_datasets[split] = tokenized_datasets[split].map(
                lambda examples: apply_to_batch(reconstruct_words, examples, kwargs_fct={'collator': data_collator}),
                batched=True,
                num_proc=num_workers,
                cache_file_name=cache_fn_reconstructed[split],
                load_from_cache_file=load_from_cache_file,
            )
            logging.info(f'Cacheing to {cache_fn_reconstructed[split]}')

        logging.info('Example of reconstructed document:')
        fst_row = tokenized_datasets['train'][0]
        logging.info(tokenizer.convert_ids_to_tokens(fst_row['input_ids'][:20]))
        logging.info(fst_row['reconstructed_words'][:10])

        if mask_strategy == 'tfidf':
            logging.info("====================================================================")
            logging.info(f"Fitting tf-idf matrix")
            score_token = create_tfidfscoring_function(
                doc_generator(tokenized_datasets['train'], 'reconstructed_words'),
                cache_file=cache_fn_tfidf,
                load_from_cache_file=load_from_cache_file,
            )
        elif mask_strategy == 'idf':
            logging.info("====================================================================")
            logging.info(f"Fitting tf-idf matrix")
            score_token = create_idfscoring_function(
                doc_generator(tokenized_datasets['train'], 'reconstructed_words'),
                cache_file=cache_fn_tfidf,
                load_from_cache_file=load_from_cache_file,
            )
        elif mask_strategy == 'term':
            logging.info("====================================================================")
            logging.info(f"Loading terms from {term_path}")
            score_token = create_termscoring_function(term_path)

        logging.info('Example of importance weights:')
        fst_row = tokenized_datasets['train'][0]
        logging.info(fst_row['reconstructed_words'][:10])
        logging.info(score_token(fst_row['reconstructed_words'][:10]))

        logging.info("====================================================================")
        logging.info("Compute importance weights according to masking strategy")

        def compute_token_importance(example, score_token):
            example['importance_weight'] = score_token(example['reconstructed_words'], normalize=False)
            return example

        # Compute words masking weights (stored in 'importance_weights')
        for split in tokenized_datasets.keys():
            tokenized_datasets[split] = tokenized_datasets[split].map(
                lambda examples: apply_to_batch(compute_token_importance, examples, kwargs_fct={'score_token': score_token}),
                batched=True,
                num_proc=num_workers,
                remove_columns=['reconstructed_words'],
                cache_file_name=cache_fn_word_import[split],
                load_from_cache_file=load_from_cache_file,
            )
            logging.info(f'Cacheing to {cache_fn_word_import[split]}')

        logging.info('Example of computed importance weights:')
        fst_row = tokenized_datasets['train'][0]
        logging.info(reconstruct_words(fst_row, data_collator)['reconstructed_words'][:10])
        logging.info(fst_row['importance_weight'][:10])

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
            num_proc=num_workers,
            cache_file_name=cache_fn_word_import_split[split],
            load_from_cache_file=load_from_cache_file,
        )
        logging.info(f'Cacheing to {cache_fn_word_import_split[split]}')

    demonstrate_data_collator(
        data_collator, tokenized_datasets['train'],
        tokenizer, num_examples=3
    )


    # for split, dataset in tokenized_datasets.items():
    #     dataset.save_to_disk(cache_fn_word_import_split[split])

    # tokenized_datasets.save_to_disk(cache_fn_word_import_split)

    # Example output
    logging.info("====================================================================")
    logging.info("Example of loaded document")
    logging.info("====================================================================")
    logging.info(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
    logging.info(tokenizer.decode(tokenized_datasets["train"][1]["input_ids"]))

    import random

    # Définir un seed pour la reproductibilité
    # Définir un seed pour la reproductibilité
    seed_value = 42

    # Mélanger le dataset
    shuffled_dataset = tokenized_datasets['train'].shuffle(seed=seed_value)

    # Répéter le processus de mélange si nécessaire
    for _ in range(2):  # Répéter deux fois de plus pour un total de trois mélanges
        shuffled_dataset = shuffled_dataset.shuffle(seed=seed_value)

    logging.info("====================================================================")
    logging.info("Shuffle documents")
    logging.info("====================================================================")
    logging.info(tokenizer.decode(shuffled_dataset[0]["input_ids"]))
    logging.info(tokenizer.decode(shuffled_dataset[1]["input_ids"]))
    logging.info("====================================================================")
    logging.info("End Preprocessing")
    logging.info("====================================================================")
 

if __name__ == "__main__":
    main()

