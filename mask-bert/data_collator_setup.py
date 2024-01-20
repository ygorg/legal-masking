# data_collator_setup.py
import os
import logging

from transformers import DataCollatorForWholeWordMask
from custom_data_collator import DataCollatorForTermSpecificMasking


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


#====
def no_tokenize(x):
    return x

def fit_or_load_tfidf(docs, cache_file, load_from_cache_file):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from joblib import dump, load

    if load_from_cache_file is True and cache_file is not None and os.path.exists(cache_file):
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

def create_tfidfscoring_function(docs, epsilon=0.00001, cache_file=None, load_from_cache_file=True):
    # TODO : how to smooth in a more inteligent way ?
    # why is epsilon = 0.00001 ??

    # Compute tfidf from given documents
    tfidf = fit_or_load_tfidf(docs, cache_file, load_from_cache_file)

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


def create_idfscoring_function(docs, epsilon=0.00001, cache_file=None, load_from_cache_file=True):
    # TODO : how to smooth in a more inteligent way ?
    # why is epsilon = 0.00001 ??

    # Compute tfidf from given documents
    tfidf = fit_or_load_tfidf(docs, cache_file, load_from_cache_file)

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
