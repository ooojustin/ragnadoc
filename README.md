# Ragnadoc

Ragnadoc is a powerful Python library for building RAG (Retrieval-Augmented Generation) systems specifically designed for technical documentation. It enables you to index, search, and query documentation from multiple sources using state-of-the-art language models and vector search capabilities.

## Features

- **Documentation Indexing**: Automatically fetch and index documentation from GitHub repositories
- **Smart Chunking**: Multiple chunking strategies including fixed-size and header-based document splitting
- **Flexible Embedding**: Support for multiple embedding providers (OpenAI, Pinecone)
- **Vector Search**: Efficient similarity search using Pinecone's vector database
- **Query Engine**: Built-in query engine using OpenAI's GPT-4 for generating accurate responses
- **Multiple Interfaces**: CLI, API, and Python library interfaces
- **Streaming Support**: Real-time streaming of generated responses
- **Rich Metadata**: Maintain document metadata including source, repository, and version information

## Installation

`pip install ragnadoc`

## Quick Start

1. Create a configuration file (config.yaml):

```yaml
# API Keys
openai_api_key: "your-openai-api-key"
pinecone_api_key: "your-pinecone-api-key"
github_token: "your-github-token"

# Repositories to index
repos:
 - "https://github.com/owner/repo/tree/main/docs"

# Embedding configuration
embedding_provider:
 type: "openai"
 config:
   model: "text-embedding-ada-002"
   batch_size: 8

# Vector store configuration
vector_store:
 type: "pinecone"
 config:
   cloud: "aws"
   region: "us-east-1"
   index_name: "ragnadoc"

# Chunking configuration
chunking_strategy:
 type: "header_based"
 config:
   max_chunk_size: 2000
   min_chunk_size: 100
```

2. Index your documentation:

```bash
ragnadoc index --config config.yaml
```

3. Query your documentation:

```bash
ragnadoc query --config config.yaml
```

## Usage

### As a CLI Tool

```bash
# Index documentation
ragnadoc index --config config.yaml

# Interactive query mode
ragnadoc query --config config.yaml

# Query with specific repository filter
ragnadoc query --config config.yaml --repo "owner/repo"

# Stream responses
ragnadoc query --config config.yaml --stream
```

### As an API Server

```bash
# Start the API server
ragnadoc api --config config.yaml --port 8000
```

The API server provides endpoints for querying documentation:

```bash
curl -X POST http://localhost:8000/query \
 -H "Content-Type: application/json" \
 -d '{"question": "How do I use the chunking feature?", "repo": "owner/repo"}'
```

### As a Python Library

```python
from ragnadoc.config import RagnadocConfig
from ragnadoc.core import RagnadocCore

# Load configuration
config = RagnadocConfig.from_yaml("config.yaml")

# Initialize core
core = RagnadocCore(config)

# Index repositories
core.index_repositories()

# Query documentation
result = core.query(
   question="How do I use the chunking feature?",
   repo="owner/repo",
   top_k=5,
   min_score=0.7,
   stream=False
)

print(result.answer)
for doc, score in zip(result.query_result.documents, result.query_result.scores):
   print(f"Source: {doc.id} (score: {score:.4f})")
```

## Architecture

Ragnadoc consists of several key components:

- **Content Processing**: Handles document chunking with support for different strategies
- **Embedding**: Manages document and query embedding using various providers
- **Vector Store**: Handles vector storage and similarity search
- **Query Engine**: Coordinates retrieval and generation for answering queries

### Content Processing

The library supports multiple chunking strategies:
- `FixedSizeChunking`: Splits documents into chunks of fixed size
- `HeaderBasedChunking`: Splits documents based on header sections

### Embedding Support

Currently supported embedding providers:
- OpenAI (`text-embedding-ada-002`)
- Pinecone (`multilingual-e5-large`)

### Vector Search

Uses Pinecone for efficient vector similarity search with support for:
- Metadata filtering
- Relevance scoring
- Batch operations

## License

Licensed under the MIT License. See the [LICENSE](https://github.com/ooojustin/ragnadoc/blob/main/LICENSE) file for details.