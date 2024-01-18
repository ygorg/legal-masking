# my_tokenize.py
from transformers import AutoModelForMaskedLM, AutoTokenizer


def initialize_tokenizer(model_checkpoint="models/bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_checkpoint)


# my_tokenizer.py
def tokenize_function(tokenizer, examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def pretokenize_function_for_just_bert(tokenizer, examples):
    """Returns text tokenized on words and not subwords"""
    pretokenized = []
    for txt in examples['text']:
        txt = tokenizer.backend_tokenizer.normalizer.normalize_str(txt)
        txt = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(txt)
        txt = [t for t, off in txt]
        pretokenized.append(txt)
    examples['pretokenized'] = pretokenized
    return examples

def pretokenize_function(tokenizer, examples):
    """Returns text tokenized on words and not subwords, adaptable for BERT and RoBERTa."""
    pretokenized = []
    tokenizer_type = type(tokenizer).__name__

    for txt in examples['text']:
        if tokenizer_type == "BertTokenizer":
            # Pour BERT
            txt = tokenizer.backend_tokenizer.normalizer.normalize_str(txt)
            txt = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(txt)
            txt = [t for t, off in txt]
        elif tokenizer_type == "RobertaTokenizer":
            # Pour RoBERTa
            encoded_input = tokenizer.encode_plus(txt, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded_input['offset_mapping']
            tokens = [txt[start:end] for (start, end) in offsets if start != end]
            txt = tokens
        else:
            # Gestion par défaut
            txt = tokenizer.tokenize(txt)

        pretokenized.append(txt)

    examples['pretokenized'] = pretokenized
    return examples

def group_texts(examples, chunk_size, split_importance_weights=True):
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
            chunk = {key: single_doc[key][i:i + chunk_size] for key in single_doc.keys() if key != "importance_weight"}

            # tokens   = A #B #C D E F #G H
            # word_ids = 0 0  0  1 2 3 3  4
            # scores   = 9       8 7 6    5
            # needs to be split as
            # A #B #C ; D E F ; #G H
            # 0 0  0  ; 1 2 3 ; 3  4
            # 9       ; 8 7 6 ; 6  5
            # So basically we use word_ids to know which importance score to get
            # + theres some None word_ids when the token is [CLASS] or [SEP]

            if split_importance_weights:
                # Find the first and last not None word_id and get the 
                i = 0
                while chunk['word_ids'][i] is None:
                    i += 1
                beg = chunk['word_ids'][i]
                i = -1
                while chunk['word_ids'][i] is None:
                    i -= 1
                end = chunk['word_ids'][i]
                chunk['importance_weight'] = single_doc['importance_weight'][beg:end+1]
            # Add chunk to result
            for key in chunk.keys():
                result[key].append(chunk[key])
    # Duplicate input_ids to labels if needed
    result["labels"] = result["input_ids"].copy()
    return result


# =============================================================================
"""
def tokenize_datasets(datasets, tokenizer):
    # ... [Your existing tokenize_function and tokenization logic]
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        tokenized_datasets[split] = dataset.map(
            lambda examples: tokenizer(examples["text"]),
            batched = True,
            remove_columns=['id', 'text','sector', 'descriptor', 'year', '__index_level_0__']
            )
        if tokenizer.is_fast:
            tokenized_datasets[split] = tokenized_datasets[split].map(
                lambda examples: {"word_ids": [tokenizer.word_ids(i) for i in range(len(examples["input_ids"]))]},
                batched= True)
    return tokenized_datasets

def group_and_process_datasets(tokenized_datasets, chunk_size):
    # ... [Your existing group_texts function and logic]
    # Process each example (document) individually
    result = {key: [] for key in tokenized_datasets.keys()}
    for index in range(len(tokenized_datasets["input_ids"])):
        # Extract single document
        single_doc = {key: tokenized_datasets[key][index] for key in tokenized_datasets.keys()}
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
"""
# =============================================================================

"""
# From https://github.com/huggingface/transformers/blob/ffa04def0e6765cdf1d9518546506325a72e2bbe/examples/pytorch/language-modeling/run_mlm.py#L534
# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
from itertools import chain
def group_texts(examples, max_seq_length):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
# remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
# might be slower to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/process#map
"""


"""def group_texts(examples, chunk_size):
    CHUNK_SIZE = 30
    for i in range(len(examples)):
        doc = examples[i]
        columns_to_split = ['input_ids', 'token_type_ids', 'attention_mask', 'word_ids']
        columns_extra = ['importance_weight']

        nb_chunks = len(examples[columns_to_split[0]]) // CHUNK_SIZE
        for i in range(nb_chunks):
            DEB = i * CHUNK_SIZE
            FIN = DEB + CHUNK_SIZE

            chunk = {k: v[DEB:FIN] for k, v in doc.items() if k in columns_to_split}
"""
