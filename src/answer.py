import ollama
import dspy
from search import get_best_matches


def answer_query(query: str, k: int):
    matches = get_best_matches(query, k)

    model = dspy.LM(model="ollama/qwen3:0.6b")
    model.
