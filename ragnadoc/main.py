from typing import Optional
from rich.console import Console
from ragnadoc.gh import GitHubClient
from ragnadoc.query.engine import QueryEngine
from ragnadoc.content import ChunkingProcessor, GitContent
from ragnadoc.content.factory import create_chunking_strategy
from ragnadoc.embeddings.factory import create_embedding_provider
from ragnadoc.vectors.factory import create_vector_store
from ragnadoc.config import RagnadocConfig
from ragnadoc.logging import initialize_logging
import logging
import click

console = Console()


def load_config(config_path: str) -> RagnadocConfig:
    """Load configuration from YAML file."""
    return RagnadocConfig.from_yaml(config_path)


@click.group()
def cli():
    initialize_logging()


@cli.command()
@click.option("--config", required=True, help="Path to configuration file")
def index(config: str):
    """Index documentation from configured repositories."""
    try:
        cfg = load_config(config)
        logger = logging.getLogger(__name__)

        # setup embedding provider using factory
        embedding_provider = create_embedding_provider(cfg)

        assert cfg.pinecone_api_key

        # setup vector store
        vs = create_vector_store(
            config=cfg,
            embedding_provider=embedding_provider,
            document_class=GitContent
        )

        # setup content processing
        chunking_strategy = create_chunking_strategy(cfg)
        content_processor = ChunkingProcessor[GitContent](chunking_strategy)
        logging.warning(chunking_strategy)
        logging.warning(content_processor)

        # setup GitHub client
        github_client = GitHubClient(cfg.github_token)

        # fetch documents
        documents = github_client.fetch_docs(cfg.repos)
        console.print(f"[blue]→[/blue] Found {len(documents)} documents")

        # process and index documents
        for doc in documents:
            logger.info(f"Processing {doc.source} from {doc.repo}")

            try:
                chunks = content_processor.process(doc)
                if chunks:
                    console.print(
                        f"[blue]Processing:[/blue] {doc.source} ({len(chunks)} chunks)"
                    )
                    vs.update(chunks)
                else:
                    console.print(
                        f"[yellow]Skipping:[/yellow] {doc.source} (no chunks created)"
                    )

            except Exception as e:
                logger.error(
                    f"Error processing document {doc.source}: {str(e)}")
                console.print(
                    f"[red]Error:[/red] Failed to process {doc.source}")
                continue

        console.print(
            "[bold green]✓[/bold green] Indexing completed successfully")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.option("--config", required=True, help="Path to configuration file")
@click.option("--repo", help="Filter by repository")
@click.option("--top-k", default=5, help="Number of results to return")
@click.option("--min-score", default=0.7, help="Minimum relevance score")
def query(config: str, repo: Optional[str], top_k: int, min_score: float):
    """Query the documentation in interactive mode."""
    try:
        cfg = load_config(config)

        # Setup embedding provider using factory
        embedding_provider = create_embedding_provider(cfg)

        assert cfg.pinecone_api_key
        assert cfg.openai_api_key

        # setup vector store
        vs = create_vector_store(
            config=cfg,
            embedding_provider=embedding_provider,
            document_class=GitContent
        )

        # setup query engine
        engine = QueryEngine[GitContent](
            vector_store=vs,
            openai_api_key=cfg.openai_api_key,
            model="gpt-4o"
        )

        console.print("[bold blue]Interactive Query Mode[/bold blue]")
        console.print("Enter your questions below (type 'quit' to exit)")

        while True:
            question = console.input("\n[bold green]Question:[/bold green] ")
            if question.lower() in ("quit", "exit", ":q"):
                if question.lower() == ":q":
                    console.print("They should call you Vim Diesel.")
                break

            filter_dict = {"repo": repo} if repo else None

            with console.status("[bold yellow]Generating answer..."):
                result = engine.query(
                    question=question,
                    filter_dict=filter_dict,
                    top_k=top_k,
                    min_relevance_score=min_score
                )

            console.print("\n[bold]Answer:[/bold]")
            console.print(result.answer)

            console.print("\n[bold]Sources:[/bold]")
            for doc, score in zip(result.query_result.documents, result.query_result.scores):
                commit_msg = doc.metadata.get(
                    "commit_message", "").split("\n")[0]
                console.print(
                    f"- {doc.id} "
                    f"[dim](repo: {doc.repo}, score: {score:.4f})"
                    f"{f', commit: {commit_msg}' if commit_msg else ''}[/dim]"
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


if __name__ == "__main__":
    cli()
