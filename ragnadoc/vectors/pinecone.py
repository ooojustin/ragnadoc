from typing import List, Dict, Optional, Type, TypeVar
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC
from pinecone.core.openapi.data.models import QueryResponse as PCQueryResponse
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, SpinnerColumn
import logging
import numpy as np
from ragnadoc.vectors.base import VectorStore, VectorSearchResult
from ragnadoc.embeddings import EmbeddingProvider, EmbeddingInput
from ragnadoc.content import Content

T = TypeVar("T", bound=Content)


class PineconeVectorStore(VectorStore[T]):
    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str,
        embedding_provider: EmbeddingProvider,
        document_class: Type[T],
        cloud: str = "aws",
        region: str = "us-east-1",
        show_progress: bool = True
    ):
        super().__init__(embedding_provider, document_class, show_progress)
        self.logger = logging.getLogger(__name__)

        self.pc = PineconeGRPC(api_key=pinecone_api_key)
        self.index_name = index_name

        if self.index_name not in self.pc.list_indexes().names():
            if self.show_progress:
                self.logger.info(f"Creating new Pinecone index: {index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_provider.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )

        self.index = self.pc.Index(self.index_name)

    def update(self, docs: List[T], batch_size: int = 100, show_progress: Optional[bool] = True) -> None:
        if not docs:
            self.logger.warning("No documents provided for update")
            return

        progress = None
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn(
                    "[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            )
            progress.start()

        try:

            embedding_inputs = [
                EmbeddingInput(
                    text=doc.text,
                    metadata=doc.to_metadata_dict()
                )
                for doc in docs
            ]

            embedding_task = None
            if progress:
                embedding_task = progress.add_task(
                    "Generating embeddings...",
                    total=len(docs)
                )

            embeddings = self.embedding_provider.embed_batch(
                embedding_inputs,
                batch_size=batch_size
            )

            if progress and embedding_task is not None:
                progress.update(embedding_task, completed=len(docs))

            vectors = [
                (doc.id or f"doc_{i}",
                 emb.embedding.tolist(),
                 doc.get_sanitized_metadata())
                for i, (doc, emb) in enumerate(zip(docs, embeddings))
            ]

            total_batches = (len(vectors) + batch_size - 1) // batch_size
            upsert_task = None
            if progress:
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
                        f"[blue]Upserted:[/blue] Batch {batch_num}/{total_batches} "
                        f"({len(batch)} vectors)"
                    )

        except Exception as e:
            self.logger.error(f"Error updating vectors: {str(e)}")
            from rich.traceback import Traceback
            from ragnadoc.main import console
            console.print(Traceback())
            raise

        finally:
            if progress:
                progress.stop()

    def search(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        top_k: int = 4
    ) -> VectorSearchResult[T]:
        try:
            query_embedding = self.embedding_provider.embed(query)

            response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                filter=filter_dict,
                include_values=True,
                include_metadata=True
            )

            assert isinstance(response, PCQueryResponse)

            documents = []
            for match in response.matches:
                text = match.metadata.pop("text")
                doc = self.document_class(
                    text=text,
                    metadata=match.metadata,
                    id=match.id
                )
                documents.append(doc)

            vectors = [np.array(match.values) for match in response.matches]
            scores = [match.score or 0.0 for match in response.matches]

            return VectorSearchResult(
                documents=documents,
                vectors=vectors,
                scores=scores
            )

        except Exception as e:
            self.logger.error(f"Error searching vectors: {str(e)}")
            raise
