from tqdm import tqdm
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownTextSplitter
from .PythonChunker import store_data
import os
import chromadb
import ollama


def chunk_markdown() -> None:
    chunks_dict = _split_all_md_files()
    docs = [chunk['content'] for chunk in chunks_dict]

    vectors = []

    for chunk in tqdm(docs, desc="Mardown Ingestion", colour="blue"):
        res = ollama.embeddings(
            model="jina/jina-embeddings-v2-base-en", prompt=chunk)
        vectors.append(res['embedding'])

    client = chromadb.PersistentClient(path="./md_vector_db")
    collection = client.get_or_create_collection(name="markdown_chunks")
    ids = [f"md_chunk-{i}" for i in range(len(chunks_dict))]

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=vectors,
        metadatas=_extract_metadata(chunks_dict)
        )
    print(ids)


def _extract_metadata(
        chunks_dict: List[Dict[str, Any]]) -> List[Dict[str, str | int]]:
    metadatas: List[Dict[str, str | int]] = []

    for chunk in chunks_dict:
        temp_dict: Dict[str, str | int] = {
            "file_path": str(chunk['source']['file_path']),
            "first_character_index": int(
                chunk['source']['first_character_index']),
            "last_character_index": int(
                chunk['source']['last_character_index'])
            }
        metadatas.append(temp_dict)

    return metadatas


def _split_all_md_files() -> List[Dict]:
    splitter = MarkdownTextSplitter(chunk_size=2000,
                                    chunk_overlap=400)
    all_files = [
        (path, dirs, files) for path, dirs, files in
        tqdm(os.walk("vllm-0.10.1"))
        ]

    chunks: List[str] = []
    chunks_dict: List[Dict] = []

    for path, _, files in tqdm(all_files, desc="Chunking", colour="red"):
        for file in files:
            chunks = []
            if not file.endswith(".md"):
                continue

            with open(path + "/" + file, "r") as f:
                doc = f.read()
            file_chunks = splitter.split_text(doc)

            for chunk in file_chunks:
                chunks.append(chunk)

            store_data(chunks, f"{path}/{file}"[12:], chunks_dict, doc)

    return chunks_dict
