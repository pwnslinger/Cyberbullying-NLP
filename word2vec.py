import nltk
import re
import warnings
import string
import numpy as np
import pandas as pd
import multiprocessing as mp

from embedding import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from gensim.utils import simple_preprocess
from gensim.models import word2vec
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
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
stop=set(stopwords.words('english'))

def clean_tweets(data):
    data["text"] = data['text'].apply(lambda x : x.lower())

    def html_decode(text):
        return BeautifulSoup(text, 'lxml').get_text().decode('utf-8')

    data['text'] = data['text'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

    def remove_stopwords(text):
        if text is not None:
            return " ".join([x for x in word_tokenize(text) if x not in stop])
        else:
            return None

    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    def remove_mentions(text):
        mention = re.compile(r'(?:RT\s)@\S+', re.IGNORECASE)
        return mention.sub('', text)

    def remove_emoji(text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_punct(text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    def remove_illegal(text):
        illegal = re.compile(r'”|“|’|\d+\S+')
        return illegal.sub('', text)

    func_list = [remove_URL, remove_mentions, remove_emoji, remove_punct, remove_stopwords, remove_illegal]

    for f in func_list:
        data['text'] = data['text'].apply(lambda x: f(x))

    # remove extra spaces left
    data.text = data.text.replace('\s+', ' ', regex=True)


class Classifier(object):
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.gs = GridSearchCV(self.model, self.param, cv=5, error_score=0,
                               refit=True, verbose=2)

    def fit(self, X, y):
        return self.gs.fit(X, y)

    def predict(self, X):
        return self.gs.predict(X)

class VectorizerMixin(TransformerMixin):
    def __init__(self, model, data):
        self.model = model(data)

    def fit(self, X, y = None):
        return self.model.fit(X, y)

    def transform(self, X, y = None):
        return self.model.transform(X)

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

vec_cls = {
    'Word2Vec': MeanEmbeddingVectorizer,
    'Word2Vec-TFIDF': TfidfEmbeddingVectorizer,
    'Tfidf': TfidfVectorizerStub,
    'FastText': FastTextVectorizer
}


def exec_pipeline(vec_method, clf_method, data):

    print("%s Classifier with %s Vectorizer Started!"%(clf_method, vec_method))

    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'],
                                                        test_size=0.3, shuffle=True)

    clf = Pipeline([(vec_method, VectorizerMixin(vec_cls[vec_method], data)),
                    ('Classifier', Classifier(clf_models[clf_method],
                                              clf_params[clf_method]))])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(clf_method, ':', clf_params[clf_method])
    print("Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \t\tF1: %1.3f\n" % (accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='macro'), recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro')))


if __name__ == '__main__':
    data = pd.read_csv('./DS2_clean.csv', encoding='utf-8')
    data['label'] = data['label'].apply(lambda label: 0 if label == False else 1)

    clean_tweets(data)

    procs = []

    for clf_method in clf_models.keys():
        for vec_method in vec_cls.keys():
            proc = mp.Process(target=exec_pipeline, args=(vec_method, clf_method, data, ))
            procs.append(proc)
            proc.start()

    for proc in procs:
        proc.join()
