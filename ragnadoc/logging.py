from datetime import datetime
from rich.logging import RichHandler
from rich.text import Text
import logging


class CustomRichHandler(RichHandler):
    LEVEL_STYLES = {
        "debug": "dim cyan",
        "info": "blue",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold red reverse",
    }

    def render_message(self, record, message):
        """Render the log message with colored time."""
        log_time = datetime.fromtimestamp(
            record.created).strftime("%H:%M:%S")

        time = Text(f"[{log_time}]", style="dim cyan")
        logger_name = Text(f"[{record.name}]", style="magenta")

        level_style = CustomRichHandler.LEVEL_STYLES.get(
            record.levelname.lower(), "blue")
        level = Text(f"[{record.levelname}]",
                     style=level_style)

        # ex: [10:00:00] [WARNING] [ragnadoc.main] hello world
        formatted_message = Text.assemble(
            time, " ", level, " ",
            logger_name, " ", record.message
        )

        return formatted_message


def initialize_logging():

    # configure the customized RichHandler
    rich_handler = CustomRichHandler(
        show_time=False,
        show_level=False,
        rich_tracebacks=True,
        markup=True
    )

    logging.basicConfig(
        level=logging.WARNING,
        handlers=[rich_handler]
    )

    # adjust levels for specific loggers
    logger_levels = {
        "pinecone_plugin_interface.logging": logging.WARNING,
        "httpx": logging.WARNING,
    }
    for name, level in logger_levels.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
