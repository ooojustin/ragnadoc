from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path
import yaml


class OpenAIEmbeddingConfig(BaseModel):
    type: Literal["openai"]
    config: dict = Field(default_factory=lambda: {
        "model": "text-embedding-ada-002",
        "batch_size": 8,
        "dimension": 1536
    })


class PineconeEmbeddingConfig(BaseModel):
    type: Literal["pinecone"]
    config: dict = Field(default_factory=lambda: {
        "model": "multilingual-e5-large",
        "batch_size": 8,
        "dimension": 1024
    })


class FixedSizeChunkingConfig(BaseModel):
    type: Literal["fixed_size"]
    config: dict = Field(default_factory=lambda: {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": [" ", "\n", ""]
    })


class HeaderBasedChunkingConfig(BaseModel):
    type: Literal["header_based"]
    config: dict = Field(default_factory=lambda: {
        "max_chunk_size": 2000,
        "min_chunk_size": 100,
        "header_patterns": [
            "^#{1,6}\\s+.+$",  # ATX headers (# Heading)
            "^.+\\n[=\\-]+\\s*$"  # Setext headers (Heading\n=====)
        ]
    })


class PineconeVectorStoreConfig(BaseModel):
    type: Literal["pinecone"]
    config: dict = Field(default_factory=lambda: {
        "cloud": "aws",
        "region": "us-east-1"
    })


class RagnadocConfig(BaseModel):
    # api keys
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    github_token: str

    # repository configuration
    repos: list[str] = Field(default_factory=list)

    # vector store config
    vector_store: PineconeVectorStoreConfig = Field(
        default_factory=lambda: PineconeVectorStoreConfig(type="pinecone")
    )

    # embedding provider configuration
    embedding_provider: Union[OpenAIEmbeddingConfig, PineconeEmbeddingConfig]

    # chunking configuration
    chunking_strategy: Union[FixedSizeChunkingConfig, HeaderBasedChunkingConfig] = Field(
        default_factory=lambda: FixedSizeChunkingConfig(
            type="fixed_size"
        )
    )

    # query configuration
    max_results: int = Field(default=4, ge=1, le=10)
    min_relevance_score: float = Field(default=0.7, ge=0.0, le=1.0)

    @validator("repos")
    def validate_repos(cls, v):
        if not v:
            raise ValueError("At least one repository URL must be provided")
        return v

    @validator("embedding_provider")
    def validate_provider_keys(cls, v, values):
        if v.type == "openai" and not values.get("openai_api_key"):
            raise ValueError(
                "OpenAI API key is required when using OpenAI embedding provider")
        if v.type == "pinecone" and not values.get("pinecone_api_key"):
            raise ValueError(
                "Pinecone API key is required when using Pinecone embedding provider")
        return v

    @validator("vector_store")
    def validate_vector_store(cls, v, values):
        if v.type == "pinecone" and not values.get("pinecone_api_key"):
            raise ValueError(
                "Pinecone API key is required when using Pinecone vector store")
        return v

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RagnadocConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as f:
            config_dict = yaml.safe_load(f)

        return cls.parse_obj(config_dict)

    def to_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with path.open("w") as f:
            yaml.dump(self.dict(), f)
