from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import numpy as np
from pydantic import BaseModel, Field


class EmbeddingInput(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True


class EmbeddingOutput(BaseModel):
    embedding: np.ndarray
    metadata: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True


class EmbeddingProvider(ABC):

    @abstractmethod
    def embed_batch(self, inputs: List[EmbeddingInput], batch_size: int = 8) -> List[EmbeddingOutput]:
        pass

    @abstractmethod
    def embed(self, query: str) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
