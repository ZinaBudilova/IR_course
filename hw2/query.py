from corpus import preprocess


def query_vector(query, vectorizer):
    query = preprocess(query)
    vector = vectorizer.transform([query]).toarray()
    return vector
