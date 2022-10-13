import json
import os
from transformers import AutoTokenizer, AutoModel
import torch


def make_corpus(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = list(f)[:50000]

    corpus = []

    for doc in data:
        answers = json.loads(doc)['answers']
        answers.sort(key=lambda x: x['author_rating']['value'], reverse=True)
        if answers:
            text = answers[0]['text']
            corpus.append(text)
    return corpus


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def save_corpus(filename):
    os.makedirs(os.path.join(os.getcwd(), 'resource', filename))
    corpus = make_corpus(filename)

    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

    encoded_input = tokenizer(corpus, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    corpus_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    torch.save(corpus_embeddings, os.path.join(os.getcwd(), 'resource', filename, 'tensor.pt'))
    return corpus_embeddings


def load_corpus(filename):
    corpus_embeddings = torch.load(os.path.join(os.getcwd(), 'resource', filename, 'tensor.pt'))
    return corpus_embeddings


def make_index(path):
    if 'resource' not in os.listdir():
        os.mkdir(os.path.join(os.getcwd(), 'resource'))
    _, filename = os.path.split(path)
    if filename not in os.listdir(path=os.path.join(os.getcwd(), 'resource')):
        print("Обработка корпуса текстов...")
        tensor = save_corpus(filename)
    else:
        print("Загрузка данных...")
        tensor = load_corpus(filename)
    return tensor
