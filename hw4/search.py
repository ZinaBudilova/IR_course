import json
from operator import itemgetter
from torch import nn


def make_title_list(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = list(f)[:50000]
    titles = []
    for doc in data:
        q = json.loads(doc)['question']
        titles.append(q)
    return titles


def sim_vector(corpus, qu_vec):
    cos = nn.CosineSimilarity(dim=1)
    return cos(corpus, qu_vec)


def get_result(vector, titles):
    result = []
    for title, rate in zip(titles, vector):
        if rate:
            result.append((title, rate))
    result = sorted(result, key=itemgetter(1), reverse=True)[:10]
    return result


def calc_similarity(corpus, qu_vec, filename):
    titles = make_title_list(filename)
    vec = sim_vector(corpus, qu_vec)
    res = get_result(vec, titles)
    return res


def print_res(res):
    for item in res:
        print(item[0])
