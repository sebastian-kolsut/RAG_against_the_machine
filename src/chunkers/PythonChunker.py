from langchain_text_splitters import PythonCodeTextSplitter
from typing import List, Dict
from .source_models import Chunk, MinimalSource
from tqdm import tqdm
import bm25s
import os


def chunk_code() -> None:
    splitter = PythonCodeTextSplitter(chunk_size=2000,
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
            if not file.endswith(".py"):
                continue

            with open(path + "/" + file, "r") as f:
                doc = f.read()
            file_chunks = splitter.split_text(doc)

            for chunk in file_chunks:
                chunks.append(chunk)

            store_data(chunks, f"{path}/{file}"[12:],
                       chunks_dict, doc)

    _save_all_data(chunks_dict)


def store_data(
        chunks: List[str], source: str, chunks_dict, doc: str) -> None:
    ready_chunks = []

    for chunk in chunks:
        first_char = doc.find(chunk)
        chunk_source = MinimalSource(
            file_path=source,
            first_character_index=first_char,
            last_character_index=first_char + len(chunk) - 1,
        )
        ready_chunks.append(Chunk(content=chunk, source=chunk_source))

    for c in ready_chunks:
        chunks_dict.append(c.dict())


def _save_all_data(chunks_dict) -> None:
    retriever = bm25s.BM25()

    docs = [c['content'] for c in chunks_dict]
    tokens = bm25s.tokenize(
        tqdm(docs, colour="yellow", desc="Tokenizing"),
        show_progress=False
        )
    retriever.index(tokens, show_progress=True)
    retriever.save("index", chunks_dict, show_progress=True)
