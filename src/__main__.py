import fire
from .chunkers import chunk_code, chunk_markdown
import chromadb
from .search import get_best_matches


def chunk():
    chunk_code()
    chunk_markdown()


def search(query: str, k: int) -> None:
    print(get_best_matches(query, k))


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
