# data_collator_setup.py
import logging

from custom_data_collator import DataCollatorForTermSpecificMasking,CustomDataCollatorForWholeWordMask, tolist
from transformers import DataCollatorForWholeWordMask

def initialize_data_collator(masking_strategies, strategy="default", collator_kwargs=None, score_kwargs=None):
    if strategy == 'default':
        tokenizer = collator_kwargs.get('tokenizer')
	#print("default strategy")
        # for adapt with BERT and RoBERTa models , be attention with Ygor code
        collator =CustomDataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            return_tensors="pt",
        )
        score_token = None
        #logging.info("default strategy")
    else:
        collator = DataCollatorForTermSpecificMasking(
            return_tensors="pt",
            **collator_kwargs
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
    # From DataCollator.pytorch_call
    ref_tokens = []
    for id in tolist(example["input_ids"]):
        token = tokenizer._convert_id_to_token(id)
        ref_tokens.append(token)

    # From DataCollator.whole_word_mask
    cand_indexes = []
    words = []  # === Reconstruct words to give to scoring function
    for i, token in enumerate(ref_tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue

        if len(cand_indexes) >= 1 and token.startswith("##"):
            cand_indexes[-1].append(i)
            words[-1] += token[2:]  # ===
        else:
            cand_indexes.append([i])
            words.append(token)  # ===

    example['importance_weight'] = score_token(words, normalize=False)
    return example


#====
def create_tfidfscoring_function(docs, epsilon=0.00001):
    # TODO : how to smooth in a more inteligent way ?
    # why is epsilon = 0.00001 ??

    # Compute tfidf from given documents
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    tfidf.fit(docs)

    # Create actual scoring function that uses the computed IDF mapping
    def score_tfidf(words, normalize=True):
        # Give a score to each word according to its TF-IDF in the corpus

        # Create word-TfIdf mapping
        tfidf_dict = {k: v for k, v in zip(
            tfidf.get_feature_names_out(),
            tfidf.transform([' '.join(words)]).toarray()[0]
        ) if v > 0 }
        weights = [tfidf_dict.get(w, epsilon) for w in words]

        # softmax
        if normalize:
            total = sum(weights)
            weights = [w/total for w in weights]
        # print([(t, int(w*100)) for t, w in zip(words, weights)])
        return weights

    return score_tfidf


def create_idfscoring_function(docs, epsilon=0.00001):
    # TODO : how to smooth in a more inteligent way ?
    # why is epsilon = 0.00001 ??

    # Compute tfidf from given documents
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    _ = tfidf.fit(docs)

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
    sw = [w.strip().lower() for w in stopwords.words('english')]

    # Load terms
    with open(path) as f:
        legal_terms = json.load(f)

    # Convert multi-word terms to single words without stopwords
    # (['abandoned property', ...] to ['abandoned', 'property', ...])
    legal_words = set([w.strip().lower() for t in legal_terms for w in t.split(' ') if w and w not in sw])

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
