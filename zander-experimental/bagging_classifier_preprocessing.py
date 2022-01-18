#!/usr/bin/env python3

# IMPORTS

# Standard imports
import string
import random
import regex as re

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Constants and helper methods
target_names = ['legitimate', 'misinformation', 'irrelevant']

max_length = 96

stop = stopwords.words('english')

def clean_text(row):
    # Lower case
    row = row.lower()

    # Remove URLs
    row = re.sub('http\S+|www.\S+', '', row)

    # Remove @mentions
    row = re.sub('@[A-Za-z0-9]+', '', row)

    # Remove non-standard characters
    row = row.encode("ascii", "ignore").decode()

    # Remove punctuation
    row = row.translate(str.maketrans('', '', string.punctuation))

    # Remove stop words
    pat = r'\b(?:{})\b'.format('|'.join(stop))
    row = row.replace(pat, '')
    row = row.replace(r'\s+', ' ')

    # Remove extraneous whitespace
    row = row.strip()

    # Lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    w_tokenization = nltk.word_tokenize(row)
    final = ""
    for w in w_tokenization:
        final = final + " " + wordnet_lemmatizer.lemmatize(w)

    return final


# Preprocessing for miscov19_p.csv

raw_df = pd.read_csv('miscov19_p.csv')
df = raw_df[['text','label']]
df.dropna()
df['text'] = df['text'].astype(str)
df.tail()
df['text'] = df['text'].apply(clean_text)
df.to_csv('processed_for_bagging_miscov19_p.csv')

raw_df = pd.read_csv('combined_data.csv')
df = raw_df[['text','label']]
df.dropna()
df['text'] = df['text'].astype(str)
df.tail()
df['text'] = df['text'].apply(clean_text)
df.to_csv('processed_for_bagging_combined_data.csv')
