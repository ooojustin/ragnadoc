from typing import List, Optional
from rich.console import Console
from ragnadoc.gh import GitHubClient
from ragnadoc.query.engine import QueryEngine
from ragnadoc.content import ChunkingProcessor, GitContent
from ragnadoc.content.factory import create_chunking_strategy
from ragnadoc.embeddings.factory import create_embedding_provider
from ragnadoc.vectors.factory import create_vector_store
from ragnadoc.config import RagnadocConfig
import logging

logger = logging.getLogger(__name__)


class RagnadocCore:
    """Core functionality of Ragnadoc, independent of CLI/API interfaces."""

    def __init__(self, config: RagnadocConfig, console: Optional[Console] = None):
        self.config = config
        self.console = console or Console()
        self.query_engine = self._init_query_engine()

    def _init_query_engine(self) -> QueryEngine:
        """Initialize the query engine with configured providers."""
        assert self.config.openai_api_key  # required until other LLMs are supported
        embedding_provider = create_embedding_provider(self.config)
        vs = create_vector_store(
            config=self.config,
            embedding_provider=embedding_provider,
            document_class=GitContent
        )
        engine = QueryEngine[GitContent](
            vector_store=vs,
            openai_api_key=self.config.openai_api_key,
            model="gpt-4"
        )
        return engine

    def index_repositories(self) -> None:
        """Index all configured repositories."""
        # setup content processing
        chunking_strategy = create_chunking_strategy(self.config)
        content_processor = ChunkingProcessor[GitContent](chunking_strategy)

        # setup GitHub client and fetch docs
        github_client = GitHubClient(self.config.github_token)
        documents = github_client.fetch_docs(self.config.repos)
        self.console.print(f"[blue]→[/blue] Found {len(documents)} documents")

        # process and index documents
        for doc in documents:
            logger.info(f"Processing {doc.source} from {doc.repo}")
            try:
                chunks = content_processor.process(doc)
                if chunks:
                    self.console.print(
                        f"[blue]Processing:[/blue] {
                            doc.source} ({len(chunks)} chunks)"
                    )
                    self.query_engine.vector_store.update(chunks)
                else:
                    self.console.print(
                        f"[yellow]Skipping:[/yellow] {
                            doc.source} (no chunks created)"
                    )
            except Exception as e:
                logger.error(f"Error processing document {
                             doc.source}: {str(e)}")
                self.console.print(
                    f"[red]Error:[/red] Failed to process {doc.source}")
                continue

    def query(
        self,
        question: str,
        repo: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.7,
        stream: bool = False
    ):
        """Query the documentation."""
        filter_dict = {"repo": repo} if repo else None
        return self.query_engine.query(
            question=question,
            filter_dict=filter_dict,
            top_k=top_k,
            min_relevance_score=min_score,
            stream=stream
        )
