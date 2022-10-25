import argparse
from index import make_corpus, make_title_list, make_index_matrix, make_index_dict
import funcs as f


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True)
    args = parser.parse_args()
    return args.data


def main():
    corpus_dir = parse_args()
    corpus = make_corpus(corpus_dir)
    print("Строим обратный индекс в формате матрицы...")
    matrix, vectorizer = make_index_matrix(corpus)
    words = vectorizer.get_feature_names_out()
    word_count = list(matrix.toarray().sum(axis=0))
    print("Наиболее частотное слово:", f.matrix_most_frequent(word_count, words))
    print("Наименее частотное слово:", f.matrix_least_frequent(word_count, words))
    print("Слова, встретившиеся во всех сериях:", ", ".join(f.matrix_in_all(matrix, words)))
    print("Наиболее популярный персонаж:", f.matrix_character(word_count, words))

    print("\nСтроим обратный индекс в формате словаря...")
    titles = make_title_list(corpus_dir)
    index_dict = make_index_dict(corpus, titles)
    print("Наиболее частотное слово:", f.dict_most_freq(index_dict))
    print("Наименее частотное слово:", f.dict_least_freq(index_dict))
    print("Слова, встретившиеся во всех сериях:", ", ".join(f.dict_in_all(index_dict)))
    print("Наиболее популярный персонаж:", f.dict_character(index_dict))


if __name__ == '__main__':
    main()
