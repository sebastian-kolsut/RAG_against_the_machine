import fire
from .chunkers import chunk_code, chunk_markdown
from .chunkers.test_vdb import TestChunker
from uuid import uuid4
import time
import bm25s
import chromadb
from .answering import get_best_matches


def chunk():
    chunk_code()
    chunk_markdown()


def answer(query: str) -> None:
    get_best_matches(query)


def retrive() -> None:
    client = chromadb.PersistentClient(path="./test_vector_bd")
    collection = client.get_or_create_collection(name="python_code_chunks")

    # Example: get a specific chunk by ID
    result = collection.get(ids=["chunk-13"])
    print(result['documents'][0])


def test():
    pass

if __name__ == "__main__":
    fire.Fire()
