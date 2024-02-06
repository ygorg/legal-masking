import os
import logging
import argparse
import os
import random
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
from datasets import load_from_disk
from datasets import Dataset
from datasets import load_dataset
import datetime
import datasets, os



masking_strategies = [
    'tfidf', 'idf', 'term', 'default'
]

splits = ["train", "validation", "test"]


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
        parser.add_argument("--batch-size", type=int, default=32, help="Training/Evaluation batch size (default: 16)")
        parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs (default: 3)")
        parser.add_argument("--chunk-size", type=int, default=128, help="Split the documents into sequences of X tokens (default: 128)")
        parser.add_argument("--jean-zay-config", type=str, default="1", help="Configuration de jeanzay (node, gpu...)")

        # parser.add_argument("--logging-dir", type=str, default="logs", help="Configuration de jeanzay (node, gpu...)")

        parser.add_argument("--mask-strategy", type=str, choices=masking_strategies, default='default', help="Scoring function to use for masking tokens (default: default)")
        parser.add_argument("--term-path", type=str, default=None, help="Path to list of terms (one term per line)")

        # parser.add_argument("--cache-dir", type=str, default=None, help="Directory to cache pretreatments (default: no cache)")
        parser.add_argument("--load-from-cache-file", choices=['true', 'false'], default='true', help="If False force recompute the cache (default: true)")
        parser.add_argument("--num-workers", type=int, default=1, help="Number of processes to use for pretreating data (default: 1)")
        parser.add_argument("--output-dir", type=str, default=None, help="Directory to save model's checkpoints (default: models/{model_name}-e{num_epochs}-b{batch_size}-{mask_strategy})")
        parser.add_argument("--num-example", type=int, default=None, help="Number of example to load (for debugging purposes) (default: all)")


        # parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
        # 
        # Parse arguments
        return parser.parse_args()
    # =============================================================================

    dist.init_process_group(backend='nccl',
                        init_method='env://',
                        world_size=idr_torch.size,
                        rank=idr_torch.rank)

    args = arguments()

    data_dir = args.data_path  # Specify your data directory
    model_checkpoint = args.model_checkpoint
    jean_zay_config = args.jean_zay_config

    # logging_dir = args.logging_dir
    

    # version = args.version

    num_epochs = args.num_epochs
    num_example = args.num_example
    mask_strategy = args.mask_strategy
    batch_size = args.batch_size

    term_path = args.term_path
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
    if args.output_dir:
        output_dir = f"{args.output_dir}/{model_name}-jz{jean_zay_config}-e{num_epochs}-b{batch_size}-c{chunk_size}-{mask_strat_for_cache}-ex{num_example if num_example else 'all'}"
    else:
        output_dir = f"models/{model_name}-jz{jean_zay_config}-e{num_epochs}-b{batch_size}-c{chunk_size}-{mask_strat_for_cache}-ex{num_example if num_example else 'all'}"


    # current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.exists(f"{output_dir}/runs"):
        os.makedirs(f"{output_dir}/runs", exist_ok=True)
  


    root_path = os.environ['DSDIR'] + '/HuggingFace_Models/'


    model_checkpoint= root_path + model_checkpoint if model_checkpoint == "bert-base-uncased" else model_checkpoint


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

    tokenizer = initialize_tokenizer(model_checkpoint=model_checkpoint)


    full_path = {}
    if data_dir is not None:
        for split in splits:
            full_path[split] = f"{data_dir}/tokenization-split/{model_name}-{mask_strat_for_cache}-c{chunk_size}-ex{num_example if num_example else 'all'}-{split}.arrow"


    tokenized_datasets = {}
    for split, cache_dir in full_path.items():
        tokenized_datasets[split] = Dataset.from_file(cache_dir) #Dataset.load_from_disk(cache_dir)


    logging.info("====================================================================")
    logging.info("Example of loaded document")
    logging.info("====================================================================")
    logging.info(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
    logging.info(tokenizer.decode(tokenized_datasets["train"][1]["input_ids"]))


    # Définir un seed pour la reproductibilité
    seed_value = 42

    # Mélanger le dataset
    tokenized_datasets['train'] = tokenized_datasets['train'].shuffle(seed=seed_value)

    # Répéter le processus de mélange si nécessaire
    for _ in range(2):  # Répéter deux fois de plus pour un total de trois mélanges
        tokenized_datasets['train'] = tokenized_datasets['train'].shuffle(seed=seed_value)

    
    data_collator = initialize_data_collator(
        mask_strategy, tokenizer, 'importance_weight'
    )

    demonstrate_data_collator(
        data_collator, tokenized_datasets['train'],
        tokenizer, num_examples=3
    )

    # Example output
    logging.info("====================================================================")
    logging.info(f"{model_name}-{mask_strat_for_cache}-c{chunk_size}-ex{num_example if num_example else 'all'}")
    logging.info("====================================================================")


   

    logging.info("====================================================================")
    logging.info("Shuffle documents")
    logging.info("====================================================================")
    logging.info(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
    logging.info(tokenizer.decode(tokenized_datasets["train"][1]["input_ids"]))

    logging.info("====================================================================")
    logging.info("Start Training")
    logging.info("====================================================================")


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
        output_dir=output_dir,
        logging_dir= f"{output_dir}/runs"
    )
    # Add training code here if needed
    eval_results = train_and_evaluate(trainer)


    logging.info("====================================================================")
    logging.info("End Training")
    logging.info("====================================================================")


if __name__ == "__main__":
    main()

