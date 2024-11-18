from typing import List, Dict, Optional, Type, TypeVar, Generic, Sequence
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from ragnadoc.content import Content
from ragnadoc.embeddings import EmbeddingProvider

T = TypeVar("T", bound=Content)


@dataclass
class VectorSearchResult(Generic[T]):
    documents: Sequence[T]
    vectors: Sequence[np.ndarray]
    scores: Sequence[float]


class VectorStore(ABC, Generic[T]):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        document_class: Type[T],
        show_progress: bool = True
    ):
        self.embedding_provider = embedding_provider
        self.document_class = document_class
        self.show_progress = show_progress

    @abstractmethod
    def update(self, docs: List[T], batch_size: int = 100) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        top_k: int = 4
    ) -> VectorSearchResult[T]:
        pass
