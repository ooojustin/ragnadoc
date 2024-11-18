from ragnadoc.embeddings import EmbeddingProvider, EmbeddingInput, EmbeddingOutput
from typing import List, Optional
import numpy as np
import openai
import logging


class OpenAIEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        batch_size: int = 8,
        timeout: int = 30,
    ):
        self.client = openai.OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.batch_size = batch_size
        self._dimension = 1536  # ada-002 dimension

    def _embed_single_batch(self, batch: List[EmbeddingInput]) -> List[EmbeddingOutput]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[input.text for input in batch]
            )

            return [
                EmbeddingOutput(
                    embedding=np.array(emb.embedding),
                    metadata=batch[i].metadata or {}
                )
                for i, emb in enumerate(response.data)
            ]
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise

    def embed_batch(self, inputs: List[EmbeddingInput], batch_size: Optional[int] = None) -> List[EmbeddingOutput]:
        batch_size = batch_size or self.batch_size
        all_embeddings = []

        # process in batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            embeddings = self._embed_single_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        return np.array(response.data[0].embedding)

    @property
    def dimension(self) -> int:
        return self._dimension
