import nltk
import numpy as np
#import tensorflow_hub as hub
#import tensorflow as tf

from tokenizer import MyTokenizer
from collections import defaultdict
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.fasttext import FastText

#embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
wpt = nltk.WordPunctTokenizer()

class Vectorizer(object):
    def __init__(self, data):
        self.data = data
        self.tokenized_corpus = MyTokenizer().fit_transform(data['text'])

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        pass

class MeanEmbeddingVectorizer(Vectorizer):
    def __init__(self, data):
        super().__init__(data)
        # Set values for various parameters
        self.feature_size = 100    # Word vector dimensionality
        self.window_context = 10  # Context window size
        self.min_word_count = 1   # Minimum word count
        self.sample = 1e-3   # Downsample setting for frequent words

        w2v_model = word2vec.Word2Vec(self.tokenized_corpus, size=self.feature_size,
                                window=self.window_context, min_count=self.min_word_count,
                                sample=self.sample, iter=50)

        self.word2vec = dict(zip(w2v_model.wv.index2word, w2v_model.wv.syn0))
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(list(self.word2vec.values())[0])

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class FastTextVectorizer(Vectorizer):
    def __init__(self, data):
        super().__init__(data)
        # Set values for various parameters
        self.feature_size = 100    # Word vector dimensionality
        self.window_context = 10  # Context window size
        self.min_word_count = 1   # Minimum word count
        self.sample = 1e-3   # Downsample setting for frequent words

        ft_model = FastText(self.tokenized_corpus, size=self.feature_size,
                            window=self.window_context,
                    min_count=self.min_word_count,sample=self.sample, sg=1, iter=50)

        self.ft = dict(zip(ft_model.wv.index2word, ft_model.wv.vectors))
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(list(self.ft.values())[0])

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        return np.array([
            np.mean([self.ft[w] for w in words if w in self.ft]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfVectorizerStub(Vectorizer):
    def __init__(self, data=None, analyzer='word', ngram_range=(2,3)):
        # build tfidf model
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(analyzer=self.analyzer,
                                     ngram_range=self.ngram_range)

    def fit(self, X, y):
            self.tfidf.fit(X)
            return self

    def transform(self, X):
        # memory limit exception
        return np.array(self.tfidf.transform(X).todense())
        #return self.tfidf.transform(X)

class CountVectorizerStub(Vectorizer):
    def __init__(self, data=None):
        # build tfidf model
        self.cv = CountVectorizer()

    def fit(self, X, y):
            self.cv.fit(X)
            return self

    def transform(self, X):
        return np.array(self.cv.transform(X).todense())

class TfidfEmbeddingVectorizer(MeanEmbeddingVectorizer):
    def __init__(self, data):
        super().__init__(data)
        self.word2weight = None

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
