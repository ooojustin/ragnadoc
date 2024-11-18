from typing import Type, TypeVar
from ragnadoc.vectors.base import VectorStore
from ragnadoc.vectors.pinecone import PineconeVectorStore
from ragnadoc.config import RagnadocConfig
from ragnadoc.content import Content
from ragnadoc.embeddings import EmbeddingProvider

T = TypeVar("T", bound=Content)


def create_vector_store(
    config: RagnadocConfig,
    embedding_provider: EmbeddingProvider,
    document_class: Type[T]
) -> VectorStore[T]:
    store_type = config.vector_store.type
    store_config = config.vector_store.config

    if store_type == "pinecone":
        assert config.pinecone_api_key
        return PineconeVectorStore(
            pinecone_api_key=config.pinecone_api_key,
            embedding_provider=embedding_provider,
            document_class=document_class,
            cloud=store_config.get("cloud", "aws"),
            region=store_config.get("region", "us-east-1"),
            index_name=store_config.get("index_name", "ragnadoc")
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
