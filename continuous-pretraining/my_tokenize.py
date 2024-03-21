# my_tokenize.py
import re

from transformers import AutoTokenizer
from transformers import BertTokenizer, BertTokenizerFast
from transformers import RobertaTokenizer, RobertaTokenizerFast
from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast


def initialize_tokenizer(model_checkpoint="models/bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_function(tokenizer, examples):

    txt = examples["text"]

    # Some special token occur naturally in some documents ([CLS]), if
    #  they appear we split them into "[CLS ]"
    is_bert_tokenizer = isinstance(tokenizer, (BertTokenizer, BertTokenizerFast))
    is_roberta_tokenizer = isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
    is_xlm_roberta_tokenizer = isinstance(tokenizer, (XLMRobertaTokenizer, XLMRobertaTokenizerFast))
    if is_bert_tokenizer:
        split_spe_tokens = re.compile(r'(\[UNK|\[SEP|\[PAD|\[CLS|\[MASK)(\])')
    elif is_roberta_tokenizer or is_xlm_roberta_tokenizer:
        split_spe_tokens = re.compile(r'(<s|</s|<unk|<pad|<mask)(>)')
    txt = [split_spe_tokens.sub(r'\1 \2', t) for t in txt]

    result = tokenizer(txt)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


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
            chunk = {key: single_doc[key][i:i + chunk_size] for key in single_doc.keys() if key not in ["importance_weight"]}

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
                chunk['importance_weight'] = single_doc['importance_weight'][beg:end + 1]

            # Add chunk to result
            for key in chunk.keys():
                result[key].append(chunk[key])
    # Duplicate input_ids to labels if needed
    result["labels"] = result["input_ids"].copy()
    return result
