import nltk
import re
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix

nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

data = pd.read_csv('./DS2_clean.csv', encoding='utf-8')
data['label'] = data['label'].apply(lambda label: 0 if label == False else 1)

# extract spams and hams
offensive = data['text'].iloc[(data['label'] == 1).values]
normal = data['text'].iloc[(data['label'] == 0).values]

def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatizer.lemmatize(token, pos='v'))
    return " ".join(result).strip()

def clean_tweets(df):
    URL = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    MENTION = r'@[A-Za-z0-9_]+'
    LINE = r'\n'
    AND = r'&amp;'
    TRUNCATED = r'[^\s]+…'
    EMOJI = r'\"\s|\s\"|\"[@]|&#\d+|@[A-Za-z0-9_]+|[:;#.&,!]|http|#'

    pattern_list = [URL, MENTION, LINE, AND, TRUNCATED, EMOJI]

    for i, t in df.items():
        if "RT " in t:
            t = re.sub(r'RT ', "", t).strip()
        for p in pattern_list:
            pattern = re.compile(p, re.IGNORECASE)
            t = pattern.sub("", t).strip()
        # preprocessing text
        t = preprocess(t)
        df[i] = t

    #return df

clean_tweets(offensive)
clean_tweets(normal)


# most common words in spam and ham
offensive_tokens = []
for word in offensive:
    offensive_tokens += nltk.tokenize.word_tokenize(word)
normal_tokens = []
for word in normal:
    normal_tokens += nltk.tokenize.word_tokenize(word)

data, test_data = train_test_split(data, test_size=0.3)
print('Train-valid data length: {0}'.format(len(data)))
print('Test data length: {0}'.format(len(test_data)))

binary_vectorizer = CountVectorizer(binary=True)
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()


def feature_extraction(df, test=False):
    if not test:
        tfidf_vectorizer.fit(df['text'])
    X = np.array(tfidf_vectorizer.transform(df['text']).todense())
    return X

train_df, valid_df = train_test_split(data, test_size=0.2)

X_train = feature_extraction(train_df)
y_train = train_df['label'].values

X_valid = feature_extraction(valid_df, test=True)
y_valid = valid_df['label'].values

clfs = {
    'mnb': MultinomialNB(),
    'gnb': GaussianNB(),
    'svm1': SVC(kernel='linear'),
    #'svm2': SVC(kernel='rbf'),
    #'svm3': SVC(kernel='sigmoid'),
    #'mlp1': MLPClassifier(),
    #'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression()
}


f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    f1_scores[clf_name] = f1_score(y_pred, y_valid)

IPython.embed()
