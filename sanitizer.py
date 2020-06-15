import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
