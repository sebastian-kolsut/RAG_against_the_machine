import bm25s


def get_best_matches(query: str):
    retriever = bm25s.BM25.load("index", load_corpus=True)
    tokenized_query = bm25s.tokenize(query)

    ids, scores = retriever.retrieve(tokenized_query, k=5)

    matched_ids = ids[0]
    matched_scores = scores[0]

    for doc_idx, score in zip(matched_ids, matched_scores):
        print(doc_idx['content'])
    print(scores)
