from tqdm import tqdm
import os
import string
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from pymorphy2 import MorphAnalyzer

vectorizer = CountVectorizer(analyzer='word')
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
    if 'resource' not in os.listdir():
        os.mkdir(os.path.join(os.getcwd(), 'resource'))

    _, folder = os.path.split(corpus_dir)
    pickle_path = os.path.join(os.getcwd(), 'resource', folder, 'corpus.pickle')

    if folder not in os.listdir(path=os.path.join(os.getcwd(), 'resource')):
        print("Обработка корпуса...")
        corpus = []
        for root, dirs, files in os.walk(corpus_dir):
            for name in tqdm(files):
                fpath = os.path.join(root, name)
                with open(fpath, 'r', encoding='utf-8-sig') as f:
                    text = f.read()
                corpus.append(preprocess(text))

        os.mkdir(os.path.join(os.getcwd(), 'resource', folder))
        with open(pickle_path, "wb") as fp:
            pickle.dump(corpus, fp)
    else:
        print("Загрузка корпуса...")
        with open(pickle_path, "rb") as fp:
            corpus = pickle.load(fp)
    return corpus


def make_title_list(corpus_dir):
    titles = []
    for root, dirs, files in os.walk(corpus_dir):
        for name in files:
            titles.append(name)
    return titles


def make_index_matrix(corpus):
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer


def make_index_dict(corpus, titles):
    index = {}
    for text, title in zip(corpus, titles):
        words = text.split()
        for word in words:
            if word not in index.keys():
                index[word] = {"count": 0,
                               "docs": []}
            if title not in index[word]["docs"]:
                index[word]["docs"].append(title)
            index[word]["count"] += 1
    return index
