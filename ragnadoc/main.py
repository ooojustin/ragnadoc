from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from ragnadoc.gh import GitHubClient
from ragnadoc.docs import DocumentProcessor
from ragnadoc.vectors import VectorStore
from ragnadoc.query import QueryEngine
from ragnadoc.logging import initialize_logging
import logging
import click
import yaml

console = Console()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    initialize_logging()


@cli.command()
@click.option('--config', required=True, help='Path to configuration file')
def index(config: str):

    try:
        cfg = load_config(config)
        logger = logging.getLogger(__name__)

        github_client = GitHubClient(cfg["github_token"])
        dp = DocumentProcessor()
        vs = VectorStore(
            pinecone_api_key=cfg["pinecone_api_key"],
            openai_api_key=cfg["openai_api_key"],
            index_name=cfg["index_name"],
        )

        # fetch documents from github
        docs = github_client.fetch_docs(cfg["repos"])
        console.print(
            f"[blue]⋲[/blue] Documents: {len(docs)}")

        for doc in docs:
            logger.info(f"Processing {doc.path} from {doc.repo_name}")

            try:
                # extract chunks
                chunks = dp.process(doc)
                vs.update(chunks)
            except Exception as e:
                logger.error(f"Error indexing documentation: {str(e)}")
                raise

        console.print(
            "[bold green]✓[/bold green] Indexing completed successfully")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.option('--config', required=True, help='Path to configuration file')
@click.option('--repo', help='Filter by repository')
def query(config: str, repo: Optional[str]):

    # TODO: add cli args for these
    top_k = 5
    min_relevance_score = 0.7

    try:
        cfg = load_config(config)

        vs = VectorStore(
            pinecone_api_key=cfg["pinecone_api_key"],
            openai_api_key=cfg["openai_api_key"],
            index_name=cfg["index_name"]
        )

        engine = QueryEngine(
            vector_store=vs,
            openai_api_key=cfg["openai_api_key"]
        )

        console.print("[bold blue]Interactive Query Mode[/bold blue]")
        console.print("Enter your questions below (type 'quit' to exit)")

        while True:
            question = console.input("\n[bold green]Question:[/bold green] ")
            if question.lower() in ('quit', 'exit'):
                break

            filter_dict = {"repo": repo} if repo else None

            with console.status("[bold yellow]Generating answer..."):
                result = engine.query(
                    question=question,
                    filter_dict=filter_dict,
                    top_k=top_k,
                    min_relevance_score=min_relevance_score
                )

            console.print("\n[bold]Answer:[/bold]")
            console.print(result.answer)

            console.print("\n[bold]Sources:[/bold]")
            for source in result.sources:
                console.print(
                    f"- {source['repo']}/{source['path']} "
                    f"(relevance: {source.get('relevance', 'N/A')})"
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


if __name__ == "__main__":
    cli()
