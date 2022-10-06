import json
import numpy as np
from operator import itemgetter


def make_title_list(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = list(f)[:50000]
    titles = []
    for doc in data:
        q = json.loads(doc)['question']
        titles.append(q)
    return titles


def sim_vector(matrix, qu_vec):
    return np.dot(matrix, qu_vec.T)


def get_result(vector, titles):
    result = []
    for title, rate in zip(titles, vector):
        if rate:
            result.append((title, rate))
    result = sorted(result, key=itemgetter(1), reverse=True)[:10]
    return result


def calc_similarity(matrix, qu_vec, filename):
    titles = make_title_list(filename)
    vec = sim_vector(matrix, qu_vec)
    res = get_result(vec, titles)
    return res


def print_res(res):
    for item in res:
        print(item[0])
