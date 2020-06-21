# -*- coding: utf-8 -*-
import re
import string
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pycontractions import Contractions

cont = Contractions(api_key="glove-twitter-100")
stop=set(stopwords.words('english'))

def clean_tweets(data, exp_flag=False):
    '''
    Desc: clean tweets and returns the cleaned dataframe as a reference to
    dataframe

    input:
    data: DataFrame
    exp_flag: boolean

    return: None
    '''
    data['text'] = data['text'].apply(lambda x : x.lower())

    slang = pd.read_csv('twitter_moods/slang.txt',sep="-",header = None, error_bad_lines=False)
    slang.columns = ['short_form', 'long_form']
    slang = {str(k):str(v) for k, v in list(zip(slang.short_form, slang.long_form))}
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in slang.keys()) + r')\b')

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

    #replace short-form slangs with the exapnded one
    def expand_slangs(text):
        return pattern.sub(lambda x: slang[x.group()], text)

    def replace_question(text):
        single_quote = re.compile(r'(\w{1,4})\?(\w{1,2})')
        return single_quote.sub(r'\1\'\2', text)
        #return list(cont.expand_texts([text]))[0]

    func_list = [remove_URL, replace_question, remove_mentions, remove_emoji,
                    remove_stopwords, remove_illegal]

    if exp_flag:
        func_list.insert(0, expand_slangs)

    for f in func_list:
        data['text'] = data['text'].apply(lambda x: f(x))

    # remove extra spaces left
    data.text = data.text.replace('\s+', ' ', regex=True)
