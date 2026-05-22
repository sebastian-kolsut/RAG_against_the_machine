import bm25s
import chromadb
import ollama
import time


def get_best_matches(query: str, k: int):
    doc_matches = _get_best_matches_doc(query, k)
    before = time.time()
    code_matches = _get_best_matches_code(query, k)
    print(time.time() - before)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("temporary")

    _add_chunks_to_collection(doc_matches, code_matches, collection)
    vector_query = ollama.embeddings(
        model="unclemusclez/jina-embeddings-v2-base-code", prompt=query)
    result = collection.query(query_embeddings=vector_query['embedding'],
                              n_results=k)

    return result


def _add_chunks_to_collection(doc_matches, code_matches, collection):
    vectors = []
    docs = []

    for chunk in doc_matches['documents'][0]:
        res = ollama.embeddings(
            model="unclemusclez/jina-embeddings-v2-base-code", prompt=chunk)
        vectors.append(res['embedding'])
        docs.append(chunk)

    for chunk in code_matches:
        res = ollama.embeddings(
            model="unclemusclez/jina-embeddings-v2-base-code",
            prompt=chunk['content'])
        vectors.append(res['embedding'])
        docs.append(chunk['content'])

    metadatas = []
    metadatas.extend(doc_matches['metadatas'][0])

    for chunk in code_matches:
        metadatas.append(chunk['source'])

    ids = [f"temp_chunk-{i}" for i in range(len(vectors))]

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=vectors,
        metadatas=metadatas
        )


def _get_best_matches_doc(query: str, k: int):
    client = chromadb.PersistentClient(path="./md_vector_db")
    collection = client.get_or_create_collection(name="markdown_chunks")

    vector_query = ollama.embeddings(
            model="jina/jina-embeddings-v2-base-en", prompt=query)
    return collection.query(query_embeddings=vector_query['embedding'],
                            n_results=k)


def _get_best_matches_code(query: str, k: int):
    retriever = bm25s.BM25.load("index", load_corpus=True)
    tokenized_query = bm25s.tokenize(query)

    ids, _ = retriever.retrieve(tokenized_query, k=k)

    return ids[0]
