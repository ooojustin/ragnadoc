from ragnadoc.embeddings.base import EmbeddingProvider
from ragnadoc.embeddings.openai import OpenAIEmbeddingProvider
from ragnadoc.embeddings.pinecone import PineconeEmbeddingProvider
from ragnadoc.config import RagnadocConfig


def create_embedding_provider(config: RagnadocConfig) -> EmbeddingProvider:
    provider_type = config.embedding_provider.type
    provider_config = config.embedding_provider.config

    if provider_type == "pinecone":
        assert config.pinecone_api_key
        return PineconeEmbeddingProvider(
            api_key=config.pinecone_api_key,
            batch_size=provider_config.get("batch_size", 8),
        )
    elif provider_type == "openai":
        assert config.openai_api_key
        return OpenAIEmbeddingProvider(
            api_key=config.openai_api_key,
            model=provider_config.get("model", "text-embedding-ada-002"),
            batch_size=provider_config.get("batch_size", 8),
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider type: {provider_type}")
