from collections import Counter


def dict_most_freq(index_dict):
    most_frequent = max(index_dict.items(), key=lambda x: x[1]["count"])
    return most_frequent[0]


def dict_least_freq(index_dict):
    least_frequent = min(index_dict.items(), key=lambda x: x[1]["count"])
    return least_frequent[0]


def dict_in_all(index_dict):
    in_all_episodes = []
    for item in index_dict.items():
        if len(item[1]["docs"]) == 165:  # кол-во эпизодов
            in_all_episodes.append(item[0])
    return in_all_episodes


def dict_character(index_dict):
    characters = {"Моника": 0,
                  "Рэйчел": 0,
                  "Чендлер": 0,
                  "Фиби": 0,
                  "Росс": 0,
                  "Джоуи": 0}
    for item in index_dict.items():
        if item[0] in ["моника", "мон"]:
            characters["Моника"] += item[1]["count"]
        elif item[0] in ["рэйчел", "рейч"]:
            characters["Рэйчел"] += item[1]["count"]
        elif item[0] in ["чендлер", "чэндлер", "чен"]:
            characters["Чендлер"] += item[1]["count"]
        elif item[0] in ["фиби", "фибс"]:
            characters["Фиби"] += item[1]["count"]
        elif item[0] == "росс":
            characters["Росс"] += item[1]["count"]
        elif item[0] in ["джоуи", "джои", "джо"]:
            characters["Джоуи"] += item[1]["count"]
    return max(characters.items(), key=lambda x: x[1])[0]


def matrix_most_frequent(word_count, words):
    max_freq = max(word_count)
    max_ind = word_count.index(max_freq)
    return words[max_ind]


def matrix_least_frequent(word_count, words):
    min_freq = min(word_count)
    min_ind = word_count.index(min_freq)
    return words[min_ind]


def matrix_in_all(matrix, words):
    in_all_episodes = []
    for item in dict(Counter(list(matrix.nonzero()[1]))).items():
        if item[1] == 165:
            in_all_episodes.append(words[item[0]])
    return in_all_episodes


def matrix_character(word_count, words):
    characters = {"Моника": 0,
                  "Рэйчел": 0,
                  "Чендлер": 0,
                  "Фиби": 0,
                  "Росс": 0,
                  "Джоуи": 0}
    characters["Моника"] += word_count[list(words).index("моника")] + list(words).index("мон")
    characters["Рэйчел"] += word_count[list(words).index("рэйчел")] + list(words).index("рейч")
    characters["Чендлер"] += word_count[list(words).index("чендлер")] + list(words).index("чэндлер") + list(
        words).index("чен")
    characters["Фиби"] += word_count[list(words).index("фиби")] + list(words).index("фибс")
    characters["Росс"] += word_count[list(words).index("росс")]
    characters["Джоуи"] += word_count[list(words).index("джоуи")] + list(words).index(
        "джо")  # "джои" is not in the list
    return max(characters.items(), key=lambda x: x[1])[0]
