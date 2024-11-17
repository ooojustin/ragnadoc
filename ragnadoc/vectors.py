from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pinecone.core.openapi.data.models import QueryResponse
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Dict, Optional
from ragnadoc.docs import DocEmbedding
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, SpinnerColumn
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
        region: str = "us-east-1",
        show_progress: bool = True
    ):
        self.logger = logging.getLogger(__name__)
        self.show_progress = show_progress

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
            if self.show_progress:
                self.logger.info(f"Creating new Pinecone index: {index_name}")
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
            if not docs:
                self.logger.warning("No documents provided for update")
                return

            progress = None
            embedding_task = None
            upsert_task = None

            try:
                if self.show_progress:
                    progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn(
                            "[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeRemainingColumn(),
                    )
                    progress.start()

                    self.logger.info(
                        f"Processing {len(docs)} documents for vectorization")
                    embedding_task = progress.add_task(
                        "Generating embeddings...",
                        total=len(docs)
                    )

                vectors = []
                for i, doc in enumerate(docs):
                    # generate embedding
                    doc_content = [doc.page_content]
                    embedding = self.embeddings.embed_documents(doc_content)[0]

                    # create vector tuple (id, values, metadata)
                    source_path = f"{doc.metadata['repo']}/{doc.metadata['source']}"
                    doc_id = f"{source_path}#{i}"
                    metadata = self._create_metadata(doc)
                    vectors.append((doc_id, embedding, metadata))

                    if progress and embedding_task is not None:
                        progress.update(
                            embedding_task,
                            advance=1,
                            description=f"Embedding: {source_path}"
                        )

                        progress.console.log(
                            f"[green]Embedded:[/green] {doc_id}"
                        )

                # upsert vectors in batches
                batch_size = 100
                total_batches = (len(vectors) + batch_size - 1) // batch_size

                if self.show_progress and progress:
                    upsert_task = progress.add_task(
                        "Upserting vectors...",
                        total=total_batches
                    )

                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    batch_num = i // batch_size + 1

                    if progress and upsert_task is not None:
                        progress.update(
                            upsert_task,
                            advance=1,
                            description=f"Upserting batch {batch_num}/{total_batches}"
                        )

                    self.index.upsert(vectors=batch)

                    if progress:
                        progress.console.log(
                            f"[blue]Upserted:[/blue] Batch {batch_num}/{total_batches} ({len(batch)} vectors)"
                        )

            finally:
                if progress:
                    progress.console.log(
                        f"[bold green]✓[/bold green] Vector store update completed successfully"
                    )
                    progress.stop()

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
            if self.show_progress:
                self.logger.debug(
                    f"Generating embedding for query: {query[:50]}...")

            # generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            if self.show_progress:
                self.logger.debug(
                    f"Querying Pinecone index: {self.index_name}")

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

            if self.show_progress:
                self.logger.debug(f"Found {len(matches)} matches")

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
