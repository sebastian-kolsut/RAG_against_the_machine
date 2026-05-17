from langchain_text_splitters import PythonCodeTextSplitter
from typing import List, Dict
from .source_models import Chunk, MinimalSource
from tqdm import tqdm
import bm25s
import os


class PythonChunker:
    def __init__(self) -> None:
        self.splitter = PythonCodeTextSplitter(chunk_size=2000,
                                               chunk_overlap=400)

    def chunk_code(self) -> None:
        all_files = os.walk("vllm-0.10.1")
        chunks: List[str] = []
        chunks_dict: List[Dict] = []
        retriever = bm25s.BM25()

        for path, _, files in tqdm(all_files, colour="blue", desc="outer loop"):
            for file in files:
                chunks = []
                if not file.endswith(".py"):
                    continue
                with open(path + "/" + file, "r") as f:
                    doc = f.read()
                file_chunks = self.splitter.split_text(doc)

                for chunk in file_chunks:
                    chunks.append(chunk)

                    self._store_data(chunks, "setup.py", chunks_dict)

        tokens = bm25s.tokenize(
            [c['content'] for c in chunks_dict], show_progress=True)
        retriever.index(tokens, show_progress=True)
        retriever.save("index", chunks_dict, show_progress=True)

    @staticmethod
    def _store_data(
            chunks: List[str], source: str, chunks_dict) -> None:
        ready_chunks = []

        for chunk in chunks:
            chunk_source = MinimalSource(
                file_path=source,
                first_character_index=1,
                last_character_index=2,
            )
            ready_chunks.append(Chunk(content=chunk, source=chunk_source))

        for c in ready_chunks:
            chunks_dict.append(c.dict())
