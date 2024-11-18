from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class Content(BaseModel):
    """Base class for all content types."""
    id: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_metadata_dict(self) -> Dict[str, Any]:
        metadata = {"text": self.text}
        metadata.update(self.metadata)
        return metadata

    def sanitize_metadata_value(self, value: Any) -> Union[str, int, float, bool, List, Dict]:
        if isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (list, tuple)):
            return [self.sanitize_metadata_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self.sanitize_metadata_value(v) for k, v in value.items()}
        else:
            return str(value)

    def get_sanitized_metadata(self) -> Dict[str, Any]:
        sanitized = {
            k: self.sanitize_metadata_value(v)
            for k, v in self.to_metadata_dict().items()
        }
        return sanitized


class GitContent(Content):
    """Content sourced from Git repositories."""
    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create(
        cls,
        text: str,
        source: str,
        repo: str,
        sha: Optional[str] = None,
        last_modified: Optional[datetime] = None,
        author: Optional[str] = None,
        chunk_index: Optional[int] = None,
        total_chunks: Optional[int] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> "GitContent":
        """Factory method to create a GitContent with computed metadata."""
        metadata = {
            "source": source,
            "repo": repo,
            "sha": sha,
            "last_modified": last_modified,
            "author": author,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        if id is None:
            id = f"{repo}/{source}#{chunk_index}" if chunk_index is not None else f"{repo}/{source}"

        return cls(
            text=text,
            metadata=metadata,
            id=id
        )

    @property
    def source(self) -> str:
        return self.metadata["source"]

    @property
    def repo(self) -> str:
        return self.metadata["repo"]

    @property
    def sha(self) -> Optional[str]:
        return self.metadata.get("sha")

    @property
    def last_modified(self) -> Optional[datetime]:
        lm = self.metadata.get("last_modified")
        if lm:
            try:
                return datetime.fromisoformat(lm)
            except (ValueError, TypeError):
                return None
        return None

    @property
    def author(self) -> Optional[str]:
        return self.metadata.get("author")

    @property
    def chunk_index(self) -> Optional[int]:
        return self.metadata.get("chunk_index")

    @property
    def total_chunks(self) -> Optional[int]:
        return self.metadata.get("total_chunks")
