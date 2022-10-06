import argparse
from corpus import make_index
from query import query_vector
from search import calc_similarity, print_res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query', action='append', required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    args = parser.parse_args()
    return args.query, args.data


def main():
    queries, corpus_dir = parse_args()

    matrix, vectorizer = make_index(corpus_dir)

    for query in queries:
        print("\nТоп-10 результатов по запросу '" + query + "':")
        vec = query_vector(query, vectorizer)
        res = calc_similarity(matrix, vec, corpus_dir)
        print_res(res)


if __name__ == '__main__':
    main()
