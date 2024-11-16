from rich.console import Console
from ragadoc.gh import GitHubClient
from ragadoc.docs import DocumentProcessor
from ragadoc.vectors import VectorStore
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

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

if __name__ == "__main__":
    cli()