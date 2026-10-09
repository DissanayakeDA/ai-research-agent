"""Turn text into embedding vectors with a local sentence-transformers model.

An embedding is a list of numbers (384 for bge-small) that represents a piece of text.
The model was trained on millions of text pairs that belong together - a question and its
answer, a sentence and its paraphrase - and learned to give each pair vectors pointing in
similar directions, while unrelated texts point elsewhere. So "postpone an exam" and
"defer an examination" land close together even though they share no words.

That closeness is learned from statistics, not understanding. It can miss negation
("satisfies" vs "violates"), exact numbers and codes, and text unlike its training data
(this model was trained on English).
"""

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """The embedding model could not be loaded, or failed to embed some text."""


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        # Imported here rather than at the top of the file: loading PyTorch and transformers
        # takes several seconds, and code that never embeds anything shouldn't pay for it.
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()  # hides the "Loading weights: 100%" bar on every load
        self.model_name = model_name
        self.batch_size = batch_size
        started = time.perf_counter()
        try:
            # Use the copy cached on disk if there is one: loads in under a second, works offline.
            self._model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except OSError:
            logger.info("Downloading embedding model %s (first run only)...", model_name)
            try:
                self._model = SentenceTransformer(model_name, device=device)
            except Exception as exc:
                raise EmbeddingError(f"Could not load embedding model '{model_name}': {exc}") from exc

        self.dimension: int = self._model.get_embedding_dimension()
        self.max_tokens: int = self._model.max_seq_length  # text beyond this is silently ignored
        logger.info(
            "Loaded %s in %.1fs: %d dimensions, max %d tokens",
            model_name, time.perf_counter() - started, self.dimension, self.max_tokens,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks for storage."""
        return self._encode(self._model.encode_document, texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a user's question for searching.

        Some models expect questions and passages to be marked differently, and
        encode_query/encode_document apply whatever the model defines. bge-small
        defines nothing, so here both produce identical vectors.
        """
        return self._encode(self._model.encode_query, [text])[0]

    def count_tokens(self, texts: list[str]) -> list[int]:
        """How many tokens the model would read for each text (before truncation)."""
        return [len(ids) for ids in self._model.tokenizer(texts, verbose=False)["input_ids"]]

    def _encode(self, encode: Callable, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            # normalize -> every vector has length 1, so cosine similarity is just a dot product
            vectors = encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return vectors.tolist()
