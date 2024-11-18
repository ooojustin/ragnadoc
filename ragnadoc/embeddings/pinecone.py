from typing import List, Optional
import numpy as np
import logging
from pinecone.grpc import PineconeGRPC
from ragnadoc.embeddings.base import EmbeddingProvider, EmbeddingInput, EmbeddingOutput


class PineconeEmbeddingProvider(EmbeddingProvider):

    PASSAGE_PARAMS = {"input_type": "passage", "truncate": "END"}
    QUERY_PARAMS = {"input_type": "query"}

    def __init__(
        self,
        api_key: str,
        model: str = "multilingual-e5-large",
        batch_size: int = 8,
    ):
        self.client = PineconeGRPC(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self._dimension = 1024

    def embed_batch(
        self,
        inputs: List[EmbeddingInput],
        batch_size: Optional[int] = None
    ) -> List[EmbeddingOutput]:
        try:
            batch_size = batch_size or self.batch_size
            all_embeddings = []
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                texts = [input.text for input in batch]
                response = self.client.inference.embed(  # type: ignore
                    model=self.model,
                    inputs=texts,
                    parameters=self.PASSAGE_PARAMS
                )

                for j, embedding_data in enumerate(response):
                    embedding = np.array(embedding_data['values'])
                    metadata = batch[j].metadata or {}

                    all_embeddings.append(EmbeddingOutput(
                        embedding=embedding,
                        metadata=metadata
                    ))

                logging.debug(
                    f"Processed batch {i//batch_size + 1}, "
                    f"size: {len(batch)}"
                )

            return all_embeddings

        except Exception as e:
            logging.error(f"Error generating batch embeddings: {str(e)}")
            raise

    def embed(self, query: str) -> np.ndarray:
        try:
            response = self.client.inference.embed(  # type: ignore
                model=self.model,
                inputs=[query],
                parameters=self.QUERY_PARAMS
            )

            # Extract the embedding vector from response
            embedding = np.array(response[0]['values'])
            return embedding

        except Exception as e:
            logging.error(f"Error generating query embedding: {str(e)}")
            raise

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings (1024 for multilingual-e5-large)."""
        return self._dimension
