from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pinecone.core.openapi.data.models import QueryResponse
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Dict, Optional
from ragnadoc.docs import DocEmbedding
import numpy as np
import logging

ADA_002_DIM = 1536  # ada-002 dimensions


class VectorStore:

    def __init__(
        self,
        pinecone_api_key: str,
        openai_api_key: str,
        index_name: str,
        dimension: int = ADA_002_DIM,
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        self.logger = logging.getLogger(__name__)

        # initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,  # type: ignore
            model="text-embedding-ada-002"
        )

        # initialize pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name

        # create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )

        self.index = self.pc.Index(self.index_name)

    def _create_metadata(self, doc: Document) -> dict:
        return {
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "repo": doc.metadata.get("repo"),
            "sha": doc.metadata.get("sha"),
            "last_modified": doc.metadata.get("last_modified", "N/A"),
            "author": doc.metadata.get("author", "N/A")
        }

    def update(self, docs: List[Document]):
        try:
            # create vectors with metadata
            vectors = []
            for i, doc in enumerate(docs):
                # generate embedding
                embedding = self.embeddings.embed_documents(
                    [doc.page_content])[0]

                # create vector tuple (id, values, metadata)
                doc_id = f"{doc.metadata['repo']}/{doc.metadata['source']}#{i}"
                metadata = self._create_metadata(doc)
                vectors.append((doc_id, embedding, metadata))

            # upsert vectors in batches of 100
            self.logger.info(f"upserting {len(vectors)} vectors")
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            self.logger.info(f"upsert done")

        except Exception as e:
            self.logger.error(f"Error updating vectors: {str(e)}")
            raise

    def query(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        top_k: int = 4
    ) -> List[DocEmbedding]:
        try:
            # generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # query pinecone
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_values=True,
                include_metadata=True
            )

            assert isinstance(response, QueryResponse)
            matches = response.matches

            # convert to DocEmbedding objects
            results = [
                DocEmbedding(
                    id=match.id,
                    vector=np.array(match.values),
                    text=match.metadata["text"],
                    metadata={
                        k: v for k, v in match.metadata.items()
                        if k != "text"
                    },
                    distance=match.score
                )
                for match in matches
            ]
            return results
        except Exception as e:
            self.logger.error(f"Error querying vector store: {str(e)}")
            raise
