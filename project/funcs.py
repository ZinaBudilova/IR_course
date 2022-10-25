import os
import pickle
import torch
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
from scipy import sparse
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()


def load_title_list():
    with open((os.path.join(os.getcwd(), 'data', 'title_list.pickle')), 'rb') as fp:
        title_list = pickle.load(fp)
    return title_list


def load_matrices():
    tfidf_matrix = np.load(os.path.join(os.getcwd(), 'data', 'tf-idf', 'matrix.npy'), allow_pickle=True)
    tfidf_matrix = tfidf_matrix.item()
    bm25_matrix = sparse.load_npz(os.path.join(os.getcwd(), 'data', 'bm25', 'matrix.npz'))
    bert_matrix = torch.load(os.path.join(os.getcwd(), 'data', 'bert', 'tensor.pt'))
    matrices = {"tf-idf": tfidf_matrix,
                "bm25": bm25_matrix,
                "bert": bert_matrix}
    return matrices


def load_tools():
    tfidf_vectorizer = pickle.load(open(os.path.join(os.getcwd(), 'data', 'tf-idf', 'vectorizer.pickle'), 'rb'))
    bm25_vectorizer = pickle.load(open(os.path.join(os.getcwd(), 'data', 'bm25', 'vectorizer.pickle'), 'rb'))
    bert_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    bert_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    tools = {"tf-idf": tfidf_vectorizer,
             "bm25": bm25_vectorizer,
             "bert_tokenizer": bert_tokenizer,
             "bert_model": bert_model}
    return tools


def load_all():
    titles = load_title_list()
    matrices = load_matrices()
    tools = load_tools()
    return titles, matrices, tools


def choose_matrix(method, matrices):
    if method == "tf-idf":
        matrix = matrices["tf-idf"]
    elif method == "bm25":
        matrix = matrices["bm25"]
    else:  # bert
        matrix = matrices["bert"]
    return matrix


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


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


def query_vector(query, method, tools):
    if method == "tf-idf":
        vectorizer = tools["tf-idf"]
        query = preprocess(query)
        vector = vectorizer.transform([query]).toarray()
    elif method == "bm25":
        vectorizer = tools["bm25"]
        query = preprocess(query)
        vector = vectorizer.transform([query])
    else:  # bert
        tokenizer = tools["bert_tokenizer"]
        model = tools["bert_model"]
        encoded_q = tokenizer(query, padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_q)
        vector = mean_pooling(model_output, encoded_q['attention_mask'])
    return vector


def get_x_max(x, items):
    result = []
    for _ in range(x):
        max_value = max(items, key=itemgetter(1))
        items.remove(max_value)
        result.append(max_value)
    return result


def get_result(vector, titles):
    result = []
    for title, rate in zip(titles, vector):
        if rate:
            result.append((title, rate))
    # result = sorted(result, key=itemgetter(1), reverse=True)[:10]
    result = get_x_max(10, result)
    return result


def calc_similarity(matrix, qu_vec, method, titles):
    if method == "tf-idf":
        vec = []
        for doc in matrix:
            rate = cosine_similarity(qu_vec, doc)[0][0]
            vec.append(rate)
    elif method == "bm25":
        vec = np.dot(matrix, qu_vec.T)
    else:  # bert
        cos = torch.nn.CosineSimilarity(dim=1)
        vec = cos(matrix, qu_vec)
    res = get_result(vec, titles)
    return res


def search(query, method, matrices, titles, tools):
    matrix = choose_matrix(method, matrices)
    qu_vec = query_vector(query, method, tools)
    res = calc_similarity(matrix, qu_vec, method, titles)
    return res


titles, matrices, tools = load_all()
