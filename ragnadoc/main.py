from typing import Optional
from rich.console import Console
from rich.live import Live
from ragnadoc.config import RagnadocConfig
from ragnadoc.core import RagnadocCore
from ragnadoc.logging import initialize_logging
from ragnadoc.query.models import StreamChatQueryResult
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
        core = RagnadocCore(cfg, console)
        core.index_repositories()
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
@click.option("--stream", "-s", is_flag=True, default=False, help="Stream the response")
def query(config: str, repo: Optional[str], top_k: int, min_score: float, stream: bool):
    """Query the documentation in interactive mode."""
    try:
        cfg = load_config(config)
        core = RagnadocCore(cfg, console)

        console.print("[bold blue]Interactive Query Mode[/bold blue]")
        console.print("Enter your questions below (type 'quit' to exit)")

        while True:
            question = console.input("\n[bold green]Question:[/bold green] ")
            if question.lower() in ("quit", "exit", ":q"):
                if question.lower() == ":q":
                    console.print("They should call you Vim Diesel.")
                break

            if stream:
                with console.status("[bold yellow]Searching documentation..."):
                    result = core.query(
                        question=question,
                        repo=repo,
                        top_k=top_k,
                        min_score=min_score,
                        stream=True
                    )

                prefix = "[bold yellow]Generating answer...[/bold yellow]\n\n[bold]Answer:[/bold]"
                out = []

                assert isinstance(result, StreamChatQueryResult)
                with Live(prefix, console=console, refresh_per_second=4) as live:
                    for chunk in result.answer_stream:
                        out.append(chunk)
                        combined_output = f"{prefix}\n" + "".join(out)
                        live.update(combined_output)
                    live.update("\n[bold]Answer:[/bold]\n" + "".join(out))
            else:
                with console.status("[bold yellow]Generating answer...[/bold yellow]"):
                    result = core.query(
                        question=question,
                        repo=repo,
                        top_k=top_k,
                        min_score=min_score,
                        stream=False
                    )
                console.print("\n[bold]Answer:[/bold]")
                console.print(result.answer)

            console.print("\n[bold]Sources:[/bold]")
            for doc, score in zip(result.query_result.documents, result.query_result.scores):
                console.print(
                    f"- {doc.id} "
                    f"[dim](repo: {doc.repo}, score: {score:.4f})"
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.option("--config", required=True, help="Path to configuration file")
@click.option("--host", default="0.0.0.0", help="Host to bind the server to")
@click.option("--port", default=8000, help="Port to run the server on")
def api(config: str, host: str, port: int):
    """Run the documentation API server."""
    try:
        cfg = load_config(config)
        core = RagnadocCore(cfg, console)

        from ragnadoc.api import RagnadocAPI
        console.print(f"[bold blue]Starting API server on {
                      host}:{port}[/bold blue]")
        api_server = RagnadocAPI(core.query_engine)
        api_server.run(host=host, port=port)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


if __name__ == "__main__":
    cli()
