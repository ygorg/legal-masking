# data_collator_setup.py
import os
import logging

from transformers import DataCollatorForWholeWordMask
from custom_data_collator import DataCollatorForTermSpecificMasking, tolist
from transformers import BertTokenizer, BertTokenizerFast
from transformers import RobertaTokenizer, RobertaTokenizerFast
from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
from transformers import HerbertTokenizer, HerbertTokenizerFast

def initialize_data_collator(strategy, tokenizer, score_column):
    if strategy in ['tfidf', 'idf', 'terms']:
        collator = DataCollatorForTermSpecificMasking(
            tokenizer=tokenizer,
            return_tensors="pt",
            score_column=score_column
        )
    else:
        collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            return_tensors="pt",
        )
    return collator


def demonstrate_data_collator(data_collator, tokenized_datasets, tokenizer, num_examples=2):
    samples = [tokenized_datasets[i] for i in range(min(num_examples, len(tokenized_datasets)))]


    logging.info("====================================================================")
    logging.info("Example of masked document")
    logging.info("====================================================================")

    for s in samples:
        chunk = data_collator([s])["input_ids"][0]
        logging.info(f"{tokenizer.decode(chunk)}")


def compute_token_importance(example, tokenizer, score_token):
    """Computes importance score of words in a document.
    
    This function should be use in a `datasets.Dataset.map`. This
    function does what is done in the data collator. But for this
    experiment we need to precompute the importance scores.
    
    Args:
        example (dict): a row of a huggingface Dataset
        tokenizer (tokenizers.Tokenizer): tokenizer used to tokenize the example
        score_token (Callable[list(words) -> list(scores)]): a function that returns a score for every word in the example
    
    Returns:
        dict: the new row of the dataset
    """

    # From DataCollator.[torch,tf,numpy]_call
    ref_tokens = []
    for id in tolist(example["input_ids"]):
        token = tokenizer._convert_id_to_token(id)
        ref_tokens.append(token)

    input_tokens = ref_tokens # ===

    # From DataCollator.whole_word_mask
    cand_indexes = []
    words = []  # === Reconstruct words to give to scoring function

    special_tokens = [val for key, val in tokenizer.special_tokens_map.items()
                      if key not in ['unk_token', 'mask_token']]
    is_bert_tokenizer = isinstance(tokenizer, (BertTokenizer, BertTokenizerFast))
    is_roberta_tokenizer = isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
    is_xlm_roberta_tokenizer = isinstance(tokenizer, (XLMRobertaTokenizer, XLMRobertaTokenizerFast))
    for i, token in enumerate(input_tokens):
        if token in special_tokens:
            continue


        if is_bert_tokenizer:
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
                words[-1] += token[2:]  # ===
            else:
                cand_indexes.append([i])
                words.append(token)  # ===
        elif is_roberta_tokenizer:
            # If a token doesn't start with Ġ, it's part of the previous token
            if len(cand_indexes) >= 1 and not token.startswith("Ġ"):
                cand_indexes[-1].append(i)
                words[-1] += token.replace("Ċ", '')  # ===
            else:
                cand_indexes.append([i])
                words.append(token.replace("Ġ", '').replace("Ċ", ''))  # ===
        elif is_xlm_roberta_tokenizer:
            # If a token doesn't start with ▁, it's part of the previous token
            if len(cand_indexes) >= 1 and not token.startswith("▁"):
                cand_indexes[-1].append(i)
                words[-1] += token  # ===
            else:
                cand_indexes.append([i])
                words.append(token[1:])  # ===
        else:
            raise ValueError("Whole-word masking only implemented for BERT/RoBERTa/XLM-Roberta so far")

    if len(cand_indexes[-1]) == 0:
        cand_indexes = cand_indexes[:-1]

    example['importance_weight'] = score_token(words, normalize=False)
    return example


#====
def no_tokenize(x):
    return x

def fit_or_load_tfidf(docs, cache_file):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from joblib import dump, load

    if cache_file is not None and os.path.exists(cache_file):
        tfidf = load(cache_file)
        logging.info(f"Using cached Tfidf from {cache_file}")
    else:
        tfidf = TfidfVectorizer(analyzer=no_tokenize)
        tfidf.fit(docs)
        if cache_file:
            dump(tfidf, cache_file)
            logging.info(f"Cacheing Tfidf to {cache_file}")
    logging.info(f"Tfidf vocabulary size : {len(tfidf.get_feature_names_out())}")
    return tfidf

def create_tfidfscoring_function(docs, epsilon=0.00001, cache_file=None):
    # TODO : how to smooth in a more inteligent way ?
    # why is epsilon = 0.00001 ??

    # Compute tfidf from given documents
    tfidf = fit_or_load_tfidf(docs, cache_file)

    # Create actual scoring function that uses the computed IDF mapping
    def score_tfidf(words, normalize=True):
        # Give a score to each word according to its TF-IDF in the corpus
        scores = tfidf.transform([words]).toarray()[0]

        words_idx = [tfidf.vocabulary_.get(w, None) for w in words]
        weights = [scores[i] if i is not None else epsilon for i in words_idx]

        # softmax
        if normalize:
            total = sum(weights)
            weights = [w/total for w in weights]
        # print([(t, int(w*100)) for t, w in zip(words, weights)])
        return weights

    return score_tfidf


def create_idfscoring_function(docs, epsilon=0.00001, cache_file=None):
    # TODO : how to smooth in a more inteligent way ?
    # why is epsilon = 0.00001 ??

    # Compute tfidf from given documents
    tfidf = fit_or_load_tfidf(docs, cache_file)

    # Create word-IDF mapping
    idf_dict = {k: v for k, v in zip(
        tfidf.get_feature_names_out(),
        tfidf.idf_
    ) if v > 0 }

    # Create actual scoring function that uses the computed IDF mapping
    def score_idf(words, normalize=True):
        # Give a score to each word according to its IDF in the corpus
        weights = [idf_dict.get(w, epsilon) for w in words]

        # softmax
        if normalize:
            total = sum(weights)
            weights = [w/total for w in weights]
        # print([(t, int(w*100)) for t, w in zip(words, weights)])
        return weights

    return score_idf


# Return a scoring function according to a list of terms (a json file)
def create_termscoring_function(path, epsilon=0.1):
    # TODO : how to smooth in a more inteligent way ?
    # why is epsilon = 0.1 ??

    # Load stopwords
    import json
    from nltk.corpus import stopwords
    stop_words = [w.strip().lower() for w in stopwords.words('english')]

    # Load terms
    with open(path) as f:
        legal_terms = [l.strip() for l in f]

    # Convert multi-word terms to single words without stopwords
    # (['abandoned property', ...] to ['abandoned', 'property', ...])
    legal_words = set([
        word.strip().lower()
        for term in legal_terms
        for word in term.split(' ')
        if word and word not in stop_words
    ])

    # Create actual scoring function that uses the loaded list of words
    def score_lawterms(words, normalize=True):
        weights = [1 if w in legal_words else epsilon for w in words]
        
        #softmax
        if normalize:
            total = sum(weights)
            weights = [w/total for w in weights]
        # print([(t, int(w*100)) for t, w in zip(words, weights)])
        return weights

    return score_lawterms
