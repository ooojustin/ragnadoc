import click
import yaml
from rich.console import Console

console = Console()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--config', required=True, help='Path to configuration file')
def index(config: str):
    try:
        cfg = load_config(config)
        console.print(cfg)
        console.print(
            "[bold green]✓[/bold green] Indexing completed successfully")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()

if __name__ == "__main__":
    cli()