from tqdm import tqdm
import os
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from pymorphy2 import MorphAnalyzer


vectorizer = TfidfVectorizer()
morph = MorphAnalyzer()


def preprocess(text):
    stop_words = set(stopwords.words('russian'))
    punctuation = set(string.punctuation).union({'...', '–', '—', "''", '»', '«', '``'})

    np.char.lower(text)
    tokens = word_tokenize(text)

    lemmas = []
    for t in tokens:
        if t not in stop_words and t not in punctuation:
            if t[-1] == '…':
                t = t[:-1]
            analysis = morph.parse(t)
            lemma = analysis[0].normal_form
            lemmas.append(lemma)
    return " ".join(lemmas)


def make_corpus(corpus_dir):
    corpus = []

    for root, dirs, files in os.walk(corpus_dir):
        for name in tqdm(files):
            fpath = os.path.join(root, name)
            with open(fpath, 'r', encoding='utf-8-sig') as f:
                text = f.read()
            corpus.append(preprocess(text))
    return corpus


def make_index(corpus_dir):
    corpus = make_corpus(corpus_dir)
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer
