from typing import Any, Dict
from ragnadoc.config import RagnadocConfig
from ragnadoc.content.chunking import (
    ChunkingStrategy,
    FixedSizeChunking,
    HeaderBasedChunking
)


def create_chunking_strategy(config: RagnadocConfig) -> ChunkingStrategy:
    strategy_type = config.chunking_strategy.type
    strategy_config = config.chunking_strategy.config

    if strategy_type == "fixed_size":
        return FixedSizeChunking(
            chunk_size=strategy_config.get("chunk_size", 1000),
            chunk_overlap=strategy_config.get("chunk_overlap", 200),
            separators=strategy_config.get("separators", [" ", "\n", ""])
        )
    elif strategy_type == "header_based":
        return HeaderBasedChunking(
            max_chunk_size=strategy_config.get("max_chunk_size", 2000),
            min_chunk_size=strategy_config.get("min_chunk_size", 100),
            header_patterns=strategy_config.get("header_patterns")
        )
    else:
        raise ValueError(
            f"Unsupported chunking strategy type: {strategy_type}")
