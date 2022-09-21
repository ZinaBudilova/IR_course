import os
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity


def make_title_list(corpus_dir):
    titles = []
    for root, dirs, files in os.walk(corpus_dir):
        for name in files:
            titles.append(name)
    return titles


def sim_vector(matrix, qu_vec):
    vector = []
    for doc in matrix:
        rate = cosine_similarity(qu_vec, doc)[0][0]
        vector.append(rate)
    return vector


def get_result(vector, titles):
    result = []
    for title, rate in zip(titles, vector):
        result.append((title, rate))
    result = sorted(result, key=itemgetter(1), reverse=True)
    return result


def calc_similarity(matrix, qu_vec, corpus_dir):
    titles = make_title_list(corpus_dir)
    vec = sim_vector(matrix, qu_vec)
    res = get_result(vec, titles)
    return res


def print_res(res):
    for item in res:
        print(item[0])
