"""See embeddings and semantic similarity with real numbers from the local model.

    python -m scripts.embedding_demo
    python -m scripts.embedding_demo "first sentence" "second sentence" "third sentence"

Runs entirely on your machine; nothing is stored.
"""

import sys

import numpy as np

from app.config import ConfigError, load_settings
from app.embeddings import Embedder, EmbeddingError

EXAMPLES = [
    "How long do I have to postpone an exam?",
    "Students may defer an examination within 7 days of the exam date.",
    "Genetic algorithms evolve timetables using crossover and mutation.",
    "Evolutionary methods improve schedules by recombining candidate solutions.",
    "The timetable satisfies all hard constraints.",
    "The timetable violates all hard constraints.",
    "The library opens at 9 am on weekdays.",
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sentences = sys.argv[1:] or EXAMPLES
    if len(sentences) < 2:
        print("Give at least two sentences to compare.")
        return 1

    try:
        embedder = Embedder(load_settings().embedding_model)
        vectors = np.array(embedder.embed_documents(sentences))
    except (ConfigError, EmbeddingError) as exc:
        print(f"FAILED: {exc}")
        return 1

    first = vectors[0]
    print(f'S1 = "{sentences[0]}"')
    print(f"  becomes {len(first)} numbers. The first eight: {np.round(first[:8], 3).tolist()}")
    print(f"  Vector length: {np.linalg.norm(first):.3f} (normalized, so cosine similarity = dot product)")
    print("  No single number means anything on its own; only the direction of the whole vector does.\n")

    for number, sentence in enumerate(sentences, start=1):
        print(f"  S{number}  {sentence}")

    similarity = vectors @ vectors.T  # every pair's dot product = cosine similarity
    count = len(sentences)
    print("\nCosine similarity of every pair (1.00 = same direction):\n")
    print("      " + "".join(f"{f'S{j}':>6}" for j in range(1, count + 1)))
    for i in range(count):
        print(f"  S{i + 1:<3}" + "".join(f"{similarity[i, j]:6.2f}" for j in range(count)))

    pairs = sorted(((similarity[i, j], i, j) for i in range(count) for j in range(i + 1, count)), reverse=True)
    print("\nPairs, most to least similar:")
    for score, i, j in pairs:
        print(f"  {score:.3f}  S{i + 1} + S{j + 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
