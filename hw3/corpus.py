import json
import string
import os
import pickle
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

from pymorphy2 import MorphAnalyzer
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


def make_corpus(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = list(f)[:50000]

    corpus = []

    for doc in tqdm(data):
        answers = json.loads(doc)['answers']
        answers.sort(key=lambda x: x['author_rating']['value'], reverse=True)
        if answers:
            text = answers[0]['text']
            corpus.append(preprocess(text))
    return corpus


def bm25_vectorization(corpus, k=2, b=0.75):
    count_vectorizer = CountVectorizer()
    count = count_vectorizer.fit_transform(corpus)
    tf = count

    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf = tfidf_vectorizer.fit_transform(corpus)

    idf = tfidf_vectorizer.idf_
    idf = sparse.csr_matrix(np.expand_dims(idf, axis=0))

    len_d = tf.sum(axis=1)
    avdl = len_d.mean()

    A = idf.multiply(tf) * (k + 1)
    B_1 = (k * (1 - b + b * len_d / avdl))
    for i, j in zip(*A.nonzero()):
        A[i, j] /= (tf[i, j] + B_1[i, 0])

    return A, count_vectorizer


def save_corpus(filename):
    os.mkdir(os.path.join(os.getcwd(), 'resource', filename))
    corpus = make_corpus(filename)
    matrix, vectorizer = bm25_vectorization(corpus)
    pickle.dump(vectorizer, open(os.path.join(os.getcwd(), 'resource', filename, 'vectorizer.pickle'), 'wb'))
    sparse.save_npz(os.path.join(os.getcwd(), 'resource', filename, 'matrix.npz'), matrix)
    return matrix, vectorizer


def load_corpus(filename):
    matrix = sparse.load_npz(os.path.join(os.getcwd(), 'resource', filename, 'matrix.npz'))
    vectorizer = pickle.load(open(os.path.join(os.getcwd(), 'resource', filename, 'vectorizer.pickle'), 'rb'))
    return matrix, vectorizer


def make_index(path):
    if 'resource' not in os.listdir():
        os.mkdir(os.path.join(os.getcwd(), 'resource'))
    _, filename = os.path.split(path)
    if filename not in os.listdir(path=os.path.join(os.getcwd(), 'resource')):
        print("Обработка корпуса текстов...")
        matrix, vectorizer = save_corpus(filename)
    else:
        print("Загрузка данных...")
        matrix, vectorizer = load_corpus(filename)
    return matrix, vectorizer
