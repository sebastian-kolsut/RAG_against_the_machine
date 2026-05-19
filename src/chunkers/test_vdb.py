from langchain_text_splitters import PythonCodeTextSplitter
from typing import List
from tqdm import tqdm
import ollama
import chromadb
import os
import subprocess
import torch


class TestChunker:
    def __init__(self) -> None:
        self.splitter = PythonCodeTextSplitter(chunk_size=2000,
                                               chunk_overlap=400)

    def create_vector_db(self):
        print(self._ollama_runtime_status())
        print(torch.cuda.get_device_name())
        chunks = self.chunk_code()

        vectors = []

        for chunk in tqdm(chunks):
            res = ollama.embeddings(model="qwen3-embedding:0.6b", prompt=chunk)
            vectors.append(res['embedding'])

        client = chromadb.PersistentClient(path="./test_vector_bd")
        collection = client.get_or_create_collection(name="python_code_chunks")
        ids = [f"chunk-{i}" for i in range(len(chunks))]

        collection.add(ids=ids, documents=chunks, embeddings=vectors)
        print(ids)

    def _ollama_runtime_status(self) -> str:
        try:
            ps = ollama.ps()
            return f"ollama ps: {ps}"
        except Exception:
            try:
                result = subprocess.run(
                    ["ollama", "ps"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = result.stdout.strip() or result.stderr.strip()
                return output or "ollama ps returned no output"
            except Exception as exc:
                return f"ollama ps unavailable: {exc}"

    def chunk_code(self) -> List[str]:
        all_files = os.walk("vllm-0.10.1")
        chunks: List[str] = []

        for path, _, files in tqdm(
                all_files, colour="blue", desc="outer loop"):
            for file in files:
                chunks = []
                if not file.endswith(".py"):
                    continue
                with open("vllm-0.10.1/setup.py", "r") as f:
                    doc = f.read()
                file_chunks = self.splitter.split_text(doc)

                for chunk in file_chunks:
                    chunks.append(chunk)

                return chunks
        return ["node"]
