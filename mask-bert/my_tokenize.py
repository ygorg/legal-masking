# my_tokenize.py
from transformers import AutoModelForMaskedLM, AutoTokenizer

from data_setup import create_small_datasets

def initialize_tokenizer(model_checkpoint="models/bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_checkpoint)

# =============================================================================
# def tokenize_datasets(datasets, tokenizer):
#     # ... [Your existing tokenize_function and tokenization logic]
#     tokenized_datasets = {}
#     for split, dataset in datasets.items():
#         tokenized_datasets[split] = dataset.map(
#             lambda examples: tokenizer(examples["text"]),
#             batched = True,
#             remove_columns=['id', 'text','sector', 'descriptor', 'year', '__index_level_0__']
#             )
#         if tokenizer.is_fast:
#             tokenized_datasets[split] = tokenized_datasets[split].map(
#                 lambda examples: {"word_ids": [tokenizer.word_ids(i) for i in range(len(examples["input_ids"]))]},
#                 batched= True)
#     return tokenized_datasets
# 
# def group_and_process_datasets(tokenized_datasets, chunk_size):
#     # ... [Your existing group_texts function and logic]
#     # Process each example (document) individually
#     result = {key: [] for key in tokenized_datasets.keys()}
#     for index in range(len(tokenized_datasets["input_ids"])):
#         # Extract single document
#         single_doc = {key: tokenized_datasets[key][index] for key in tokenized_datasets.keys()}
#         # Compute length
#         doc_length = len(single_doc["input_ids"])
#         # Adjust length to fit the chunk size
#         doc_length = (doc_length // chunk_size) * chunk_size
#         # Segment into chunks
#         for i in range(0, doc_length, chunk_size):
#             chunk = {key: single_doc[key][i:i + chunk_size] for key in single_doc.keys()}
#             # Add chunk to result
#             for key in chunk.keys():
#                 result[key].append(chunk[key])
#     # Duplicate input_ids to labels if needed
#     result["labels"] = result["input_ids"].copy()
#     return result
# =============================================================================

# my_tokenizer.py
def tokenize_function(tokenizer, examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(chunk_size, examples):
    # Process each example (document) individually
    result = {key: [] for key in examples.keys()}
    for index in range(len(examples["input_ids"])):
        # Extract single document
        single_doc = {key: examples[key][index] for key in examples.keys()}
        # Compute length
        doc_length = len(single_doc["input_ids"])
        # Adjust length to fit the chunk size
        doc_length = (doc_length // chunk_size) * chunk_size
        # Segment into chunks
        for i in range(0, doc_length, chunk_size):
            chunk = {key: single_doc[key][i:i + chunk_size] for key in single_doc.keys()}
            # Add chunk to result
            for key in chunk.keys():
                result[key].append(chunk[key])
    # Duplicate input_ids to labels if needed
    result["labels"] = result["input_ids"].copy()
    return result
