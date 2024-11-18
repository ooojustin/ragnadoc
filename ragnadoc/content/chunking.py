from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragnadoc.content.models import Content
import re
import inspect

T = TypeVar('T', bound=Content)


class ChunkingStrategy(ABC):

    @abstractmethod
    def split(self, text: str) -> List[str]:
        pass

    def clean_text(self, text: str) -> str:
        cleaned = text.replace('\r\n', '\n')
        return '\n'.join(line.rstrip() for line in cleaned.split('\n')).strip()


class FixedSizeChunking(ChunkingStrategy):

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or [" ", "\n", ""]
        )

    def split(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)
        return self.splitter.split_text(cleaned)


class HeaderBasedChunking(ChunkingStrategy):

    def __init__(
        self,
        max_chunk_size: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
        header_patterns: Optional[List[str]] = None
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.header_patterns = header_patterns or [
            r"^#{1,6}\s+.+$",  # ATX headers
            r"^.+\n[=\-]+\s*$"  # Setext headers
        ]

    def split(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)
        chunks = []
        current_chunk = []

        for line in cleaned.split('\n'):
            # check if line is a header
            is_header = any(re.match(pattern, line, re.MULTILINE)
                            for pattern in self.header_patterns)

            if is_header and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                if self._is_valid_chunk_size(chunk_text):
                    chunks.append(chunk_text)
                current_chunk = []

            current_chunk.append(line)

        # add the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if self._is_valid_chunk_size(chunk_text):
                chunks.append(chunk_text)

        return chunks

    def _is_valid_chunk_size(self, text: str) -> bool:
        if not text:
            return False
        if self.min_chunk_size and len(text) < self.min_chunk_size:
            return False
        if self.max_chunk_size and len(text) > self.max_chunk_size:
            return False
        return True


class ChunkingProcessor(Generic[T]):

    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy

    def process(self, content: T) -> List[T]:
        chunks = self.strategy.split(content.text)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # create new content using the same class as the input
            chunk_metadata = content.metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

            # use the same constructor pattern as the input type
            if hasattr(content, "create"):
                # for types like GitContent that have a create method
                create_method = getattr(content.__class__, "create")
                valid_params = inspect.signature(
                    create_method).parameters.keys()

                kwargs = {
                    k: v
                    for k, v in chunk_metadata.items()
                    if k in valid_params
                }
                if "text" in valid_params:
                    kwargs["text"] = chunk

                chunk_content = content.__class__.create(  # type: ignore
                    **kwargs)
            else:
                # for base Content types
                chunk_id = f"{content.id}#chunk{i}" if content.id else None
                chunk_content = content.__class__(
                    text=chunk,
                    metadata=chunk_metadata,
                    id=chunk_id
                )

            processed_chunks.append(chunk_content)

        return processed_chunks
