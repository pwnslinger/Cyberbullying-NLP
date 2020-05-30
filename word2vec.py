import nltk
import re
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp

from collections import defaultdict
from gensim.utils import simple_preprocess
from gensim.models import word2vec
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
np.seterr(divide='ignore', invalid='ignore')
pd.options.display.max_colwidth = 200

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatizer.lemmatize(token, pos='v'))
    return " ".join(result).strip()

def clean_tweets(df):
    URL = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    MENTION = r'@[A-Za-z0-9_]+'
    RT = r'RT '
    LINE = r'\n'
    AND = r'&amp;'
    TRUNCATED = r'[^\s]+…'
    EMOJI = r'\"\s|\s\"|&#\d+|@[A-Za-z0-9_]+|[:;#.&,!]|http'

    pattern_list = [URL, MENTION, LINE, AND, TRUNCATED, EMOJI, RT]

    for i, t in df.items():
        for p in pattern_list:
            pattern = re.compile(p, re.IGNORECASE)
            t = pattern.sub(" ", t).strip()
        # preprocessing text
        t = preprocess(t)
        df[i] = t


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfVectorizerStub(object):
    def __init__(self, tfidf):
        self.tfidf = tfidf

    def fit(self, X, y):
            self.tfidf.fit(X)
            return self

    def transform(self, X):
        return np.array(self.tfidf.transform(X).todense())


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

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

class Classifier(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.gs = GridSearchCV(self.model, self.param, cv=5, error_score=0, refit=True)

    def fit(self, X, y):
        return self.gs.fit(X, y)

    def predict(self, X):
        return self.gs.predict(X)

clf_models = {
    'MultinomialNB': MultinomialNB(),
    'NaiveBayes': GaussianNB(),
    'SVC': SVC(),
    'DecisionTree': DecisionTreeClassifier(),
    'Perceptron': MLPClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'LogisticRegression': LogisticRegression(),
    'AdaBoost': AdaBoostClassifier(),
    'MLPClassifier': MLPClassifier()
}

clf_params = {
    'NaiveBayes': {},
    'MultinomialNB': { 'alpha': [0.5, 1], 'fit_prior': [True, False] },
    'SVC': { 'kernel': ['linear'] },
    'DecisionTree': { 'min_samples_split': [2, 5] },
    'Perceptron': { 'alpha': [0.0001, 0.001], 'activation': ['tanh', 'relu'] },
    'GradientBoosting': { 'learning_rate': [0.05, 0.1], 'min_samples_split': [2, 5] },
    'MLPClassifier': { 'hidden_layer_sizes': [100,100]},
    'LogisticRegression': {},
    'AdaBoost': {},
    'RandomForest': {}
}


def exec_pipeline(vectorizer, cls_method, X_train, X_test, y_train, y_test):
    print("%s Classifier Started!"%cls_method)
    # Word2Vec
    '''
    clf = Pipeline([('Word2Vec vectorizer',
                     MeanEmbeddingVectorizer(vectorizer)), ('Classifier',
                                                            Classifier(clf_models[cls_method],
                                                                       clf_params[cls_method]))])
    '''

    # Word2Vec-TFIDF
    '''
    clf = Pipeline([('Word2Vec-TFIDF vectorizer',
                     TfidfEmbeddingVectorizer(vectorizer)), ('Classifier',
                                                            Classifier(clf_models[cls_method],
                                                                       clf_params[cls_method]))])
    '''

    # TFIDF
    clf = Pipeline([('TFIDF vectorizer', TfidfVectorizerStub(vectorizer)),
                    ('Classifier', Classifier(clf_models[cls_method],
                        clf_params[cls_method]))])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(cls_method, ':', clf_params[cls_method])
    print("Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \t\tF1: %1.3f\n" % (accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='macro'), recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro')))


if __name__ == '__main__':
    data = pd.read_csv('./DS2_clean.csv', encoding='utf-8')
    data['label'] = data['label'].apply(lambda label: 0 if label == False else 1)

    clean_tweets(data['text'])

    # build tfidf model
    tfidf = TfidfVectorizer()

    # build word2vec model
    wpt = nltk.WordPunctTokenizer()
    tokenized_corpus = [wpt.tokenize(document) for document in data['text']]

    # Set values for various parameters
    feature_size = 100    # Word vector dimensionality
    window_context = 10  # Context window size
    min_word_count = 1   # Minimum word count
    sample = 1e-3   # Downsample setting for frequent words

    w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size,
                            window=window_context, min_count=min_word_count,
                            sample=sample, iter=50)

    w2v = dict(zip(w2v_model.wv.index2word, w2v_model.wv.vectors))

    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'],
                                                        test_size=0.3, shuffle=True)

    procs = []

    for key in clf_models.keys():
        proc = mp.Process(target=exec_pipeline, args=(tfidf, key, X_train, X_test,
                                                y_train, y_test, ))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
