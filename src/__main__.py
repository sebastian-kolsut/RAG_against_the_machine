import fire
from .chunkers import PythonChunker
from uuid import uuid4
import time
import bm25s


def chunk():
    chunker = PythonChunker()
    chunker.chunk_code()


def retrive() -> None:
    retriever = bm25s.BM25.load("index_bm25", load_corpus=True)

    query_tokens = bm25s.tokenize("sccache compilation launcher")
    doc_ids, scores = retriever.retrieve(query_tokens, k=1)

    print(doc_ids[0][0]['content'])


def test():
    import os

    walk = os.walk("vllm-0.10.1")
    for x in walk:
        print(x)


if __name__ == "__main__":
    fire.Fire()
