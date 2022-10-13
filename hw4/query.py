from transformers import AutoTokenizer, AutoModel
import torch
from corpus import mean_pooling


def query_vector(query):
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

    encoded_q = tokenizer(query, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_q)
    vector = mean_pooling(model_output, encoded_q['attention_mask'])
    return vector
