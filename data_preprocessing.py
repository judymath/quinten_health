#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata
from textblob import TextBlob
from nltk.corpus import stopwords
from collections import Counter
import warnings; warnings.simplefilter('ignore')
import nltk
import string
from nltk import ngrams
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
import spacy

def preprocess_review(review):
    # Load the spaCy English language model
    nlp = spacy.load("en_core_web_sm")

    # Changing to lowercase
    review = review.lower()

    # Removing HTML entities (e.g., &amp;, &lt;, &gt;)
    review = re.sub(r'&\w+;', '', review)

    # Removing URLs
    review = re.sub(r'http\S+|www.\S+', '', review)

    # Removing special characters and punctuation
    review = re.sub(r'[^\w\s]', '', review)

    # Removing all non-ASCII characters
    review = re.sub(r'[^\x00-\x7F]+', '', review)

    # Removing leading and trailing whitespaces
    review = review.strip()

    # Replacing multiple spaces with a single space
    review = re.sub(r'\s+', ' ', review)

    # Replacing two or more dots with one
    review = re.sub(r'\.{2,}', ' ', review)

    # Lemmatization using spaCy
    doc = nlp(review)
    lemmatized_review = " ".join([token.lemma_ if token.lemma_ != "-PRON-" else token.text for token in doc])

    return lemmatized_review

def preprocess_data(data):
    # Apply the preprocessing function to the 'comment' column
    data['review_clean'] = data['comment'].apply(preprocess_review)

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    data['review_clean'] = data['review_clean'].apply(
        lambda x: ' '.join(word for word in x.split() if word not in stop_words)
    )

    # Sentiment analysis and polarity
    data['sentiment'] = data['comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data['sentiment_clean'] = data['review_clean'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Other text-based features
    data['count_word'] = data["review_clean"].apply(lambda x: len(str(x).split()))
    data['count_unique_word'] = data["review_clean"].apply(lambda x: len(set(str(x).split())))
    data['count_letters'] = data["review_clean"].apply(lambda x: len(str(x)))
    data['count_punctuations'] = data["comment"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    data['count_words_upper'] = data["comment"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    data['count_words_title'] = data["comment"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    data['count_stopwords'] = data["comment"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
    data['mean_word_len'] = data["review_clean"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    return data

