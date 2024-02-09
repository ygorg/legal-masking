import os
import json
import logging

from tqdm import tqdm


class ScoringFunction:
    def __init__(self, epsilon=0.00001, load_from_cache_file=True):
        from string import punctuation
        import nltk.corpus

        self.epsilon = epsilon
        self.load_from_cache_file = load_from_cache_file

        self.stop_words = set(nltk.corpus.stopwords.words('english')) | set(punctuation) | set('—…“”’–')

    def normalize(self, lst):
        total = sum(lst)
        lst = [w / total for w in lst]
        return lst

    def score_sequence(self, normalize=True):
        raise NotImplementedError


def _no_tokenize(x):
    return x


class TfIdfScoring(ScoringFunction):
    def __init__(self, docs, epsilon=0.00001, cache_file_prefix=None, load_from_cache_file=True):
        super().__init__(epsilon, load_from_cache_file)
        self.cache_file_tfidf = None
        if cache_file_prefix:
            self.cache_file_tfidf = cache_file_prefix + f"-tfidf.pkl"
        self.tfidf = None
        self._fit_or_load_tfidf(docs)

    def _fit_or_load_tfidf(self, docs):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from joblib import dump, load

        if self.load_from_cache_file is True and self.cache_file_tfidf is not None and os.path.exists(self.cache_file_tfidf):
            logging.info(f"Using cached TfIdf from {self.cache_file_tfidf}")
            self.tfidf = load(self.cache_file_tfidf)

        else:
            logging.info(f"Computing TfIdf")
            self.tfidf = TfidfVectorizer(analyzer=_no_tokenize)
            self.tfidf.fit(docs)
            if self.cache_file_tfidf is not None:
                dump(self.tfidf, self.cache_file_tfidf)
                logging.info(f"Cacheing Tfidf to {self.cache_file_tfidf}")
        logging.info(f"Tfidf vocabulary size : {len(self.tfidf.get_feature_names_out())}")

    def score_sequence(self, words, normalize=True):
        # Give a score to each word according to its TF-IDF in the corpus
        scores = self.tfidf.transform([words]).toarray()[0]

        words_idx = [self.tfidf.vocabulary_.get(w, None) for w in words]
        weights = [scores[i] if i is not None else self.epsilon for i in words_idx]

        # softmax
        if normalize:
            weights = self.normalize(weights)
        return weights


class IdfScoring(TfIdfScoring):
    def __init__(self, docs, epsilon=0.00001, cache_file_prefix=None, load_from_cache_file=True):
        super().__init__(docs, epsilon, cache_file_prefix, load_from_cache_file)
        self.word2score = None
        self._compute_word2score()

    def _compute_word2score(self):
        self.word2score = {k: v for k, v in zip(
            self.tfidf.get_feature_names_out(),
            self.tfidf.idf_
        ) if v > 0}

    def score_sequence(self, words, normalize=True):
        # Give a score to each word according to its IDF in the corpus
        weights = [self.word2score.get(w, self.epsilon) for w in words]

        # softmax
        if normalize:
            weights = self.normalize(weights)
        return weights


class MetaDiscourseScoring(ScoringFunction):
    def __init__(self, docs, epsilon=0.00001, phrase_delimiter='ĊĊ', cache_file_prefix=None, load_from_cache_file=True):
        super().__init__(epsilon, load_from_cache_file)

        self.cache_file_word2score = None
        self.cache_file_phrasemodel = None
        if cache_file_prefix:
            self.cache_file_word2score = cache_file_prefix + f"-metadiscourse.json"
            self.cache_file_phrasemodel = cache_file_prefix + f"-phrasemodel.pkl"

        self.phrase_delimiter = phrase_delimiter

        self.word2score = None
        self.phrase_model = None

        if self.load_from_cache_file is False or self.cache_file_phrasemodel is None or not os.path.exists(self.cache_file_phrasemodel):    
            logging.info(f'Loading all files (this could take a while with a lot of documents (>100_000))')
            docs = list(docs)

        self._fit_or_load_phrasemodel(docs)
        self._fit_or_load_word2score(docs)

    def _fit_or_load_phrasemodel(self, docs):
        from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
 
        if self.load_from_cache_file is True and self.cache_file_phrasemodel is not None and os.path.exists(self.cache_file_phrasemodel):
            logging.info(f"Using cached phrase model from {self.cache_file_phrasemodel}")
            self.phrase_model = Phrases(
                [], min_count=5, threshold=0.5, delimiter=self.phrase_delimiter,
                connector_words=ENGLISH_CONNECTOR_WORDS, scoring="npmi"
            )
            self.phrase_model = self.phrase_model.load(self.cache_file_phrasemodel)
        else:
            logging.info(f"Computing phrase model")

            self.phrase_model = Phrases(
                docs, min_count=5, threshold=0.5, delimiter=self.phrase_delimiter,
                connector_words=ENGLISH_CONNECTOR_WORDS, scoring="npmi"
            )

            self.phrase_model.add_vocab(docs)  # add new sentences to model
            if self.cache_file_phrasemodel is not None:
                logging.info(f"Cacheing phrase model to {self.cache_file_phrasemodel}")
                self.phrase_model.save(self.cache_file_phrasemodel)

    def _fit_or_load_word2score(self, docs):
        if self.load_from_cache_file is True and self.cache_file_word2score is not None and os.path.exists(self.cache_file_word2score):
            logging.info(f"Using cached discourse weights from {self.cache_file_word2score}")
            with open(self.cache_file_word2score) as f:
                self.word2score = json.load(f)
        else:
            self._compute_word2score(docs)

    def _compute_word2score(self, docs):
        import statistics

        logging.info(f"Computing phrases for each document")
        docs_phrase = self.phrase_model[docs]
        logging.info(f"Computing tf and df")
        df, tfd, tf = self.df_itf_times_df_iD(docs_phrase)

        logging.info(f"Computing metadiscourse score")
        self.word2score = dict()
        N = len(docs_phrase)
        for t in tqdm(tf, desc='score'):
            if t.lower() in self.stop_words:
                continue

            if len(tfd[t]) == 1:
                tfd[t].append(tfd[t][0])  # to ensure StatisticsError: variance requires at least two data points
            self.word2score[t] = (df[t] / N) * (df[t] / tf[t]) * (1 - statistics.stdev(tfd[t]) / max(tfd[t]))

        if self.cache_file_word2score is not None:
            logging.info(f"Cacheing metadiscourse score to {self.cache_file_word2score}")
            with open(self.cache_file_word2score, 'w') as f:
                json.dump(self.word2score, f)

    def df_itf_times_df_iD(self, corpus):
      # df number of documents in which t occurs
      # D  number of documents in the corpus
      # tf number of term occurrences summing occurrences in all documents
      #  df/tf [0;1] the closer to 1 the result is the more the term is genre specific of a subset of documents ;
      # en fait serait plus juste de dire l'écart type est faible sur les documents où le termes apparait
      # df/D [0;1] the closer to 1 the result is the more the term is genre specific of the whole document collection ;
      # v1) 2df/(tf*D) se rapproche de ce que l'on veut
      # v2) (df/D)*(1-stddev_df/(max_len_d/2)) une alternative le nombre de mots dans le plus grand texte
      #  ou le plus grand nombre d'occurrence d'un même mot dans un document
      #  ou le max tfd, le max d'occ de ce terme dans un doc
      # mais cette formule donnerait des scores de genre specific pour la collection mais pas pour un sous ensemble
      # v3) df/tf*(1-stddev_df/(max_len_d/2)) 1 alternative
      df = dict() # for a given t, number of distinct d in which t occurs
      for d in tqdm(corpus, desc='df'):
        for t in set(d):
          if t in df: df[t] += 1
          else: df[t] = 1

      tfd = dict() # for a given t, list of number of occurrences when occurs in a document
      for d in tqdm(corpus, desc='tf'):
        tfds = dict() # the number of occurrences of each term of the current d
        for t in d:
          if t in tfds: tfds[t] += 1
          else: tfds[t] = 1
        # pour chaque mot compté dans le document, on ajoute une entrée pour le mot dans le compteur d'occurrences de chaque mot pour chaque document
        for t in tfds:
          if t in tfd:
            tfd[t].append(tfds[t])
          else:
            tfd[t] = list()
            tfd[t].append(tfds[t])

      tf = dict()  # for a given t, number total of occurrences of t in the whole corpus
      for t in tfd:
        tf[t] = sum(tfd[t])
      return df, tfd, tf

    def score_sequence(self, words, normalize=True):
        # Give a score to each word according to its TF-IDF in the corpus

        # Merge phrases
        phrases = self.phrase_model[words]
        # Score phrases
        tmp_weights = [(t, self.word2score.get(t, self.epsilon)) for t in phrases]
        # Un-merge phrases (the score of compound words is the same as the score of the phrase)
        # [("inĊĊconclusion", .5) -> [("in", .5), ("conclusion", .5)]
        weights = []
        for t, s in tmp_weights:
            weights += [s] * len(t.split(self.phrase_delimiter))

        # softmax
        if normalize:
            weights = self.normalize(weights)
        return weights


"""
# Compute tf from given documents
ct_vec = CountVectorizer(analyzer=no_tokenize)
X = ct_vec.fit_transform(docs)

# From sklearn.TfIdfTransformer compute df
# X.indices has the wordindex
df = np.bincount(X.indices, minlength=X.shape[1])
# df : number of documents in which t occurs
N = len(df)  # Total number of document
max_tfd = X.max(axis=0).toarray()[0]
# max_tfd : max number of occurences of term in all documents
tf = X.sum(axis=0).tolist()[0]
# tf : number of term occurrences summing occurrences in all documents

word2score = {}
X_ = X.tocsc()
for i, word in enumerate(ct_vec.get_feature_names_out()):
    tfd = X_[:,i].data
    # standard deviation of term occurences in all documents
    if len(tfd) == 1:
        # If there is only one element the std is 0
        std_tfd = 0
    else:
        std_tfd = np.std(tfd, ddof=1)

    word2score[word] = np.log((df[i]/N) (df[i]/tf[i]) * (1 - (std_tfd / max_tfd[i])))
"""


class TermScoring(ScoringFunction):
    def __init__(self, term_path, epsilon=0.00001, **kwargs):
        super().__init__(epsilon=epsilon, load_from_cache_file=None)
        self.word2score = None
        self._compute_word2score(term_path)

    def _compute_word2score(self, term_path):
        # Load terms
        logging.info(f'Loading terms from {term_path}')
        with open(term_path) as f:
            legal_terms = [l.strip() for l in f]

        # Convert multi-word terms to single words without stopwords
        # (['abandoned property', ...] to ['abandoned', 'property', ...])
        self.word2score = {
            word.strip().lower(): 1.
            for term in legal_terms
            for word in term.split(' ') if word and word not in self.stop_words
        }

    def score_sequence(self, words, normalize=True):
        # Give a score to each word according to its IDF in the corpus
        weights = [self.word2score.get(w, self.epsilon) for w in words]

        # softmax
        if normalize:
            weights = self.normalize(weights)
        return weights


masking_strategies = {
    'tfidf': TfIdfScoring, 'idf': IdfScoring,
    'metadiscourse': MetaDiscourseScoring,
    'term': TermScoring, 'default': None
}
