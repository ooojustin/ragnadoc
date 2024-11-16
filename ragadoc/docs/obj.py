from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np

@dataclass
class DocInfo:
    path: str
    content: str
    sha: str
    repo_name: str
    last_modified: Optional[datetime] = None
    author: Optional[str] = None

@dataclass
class DocEmbedding:
    """Represents a document chunk with its embedding."""
    id: str  # format: {repo_name}/{file_path}#{chunk_number}
    vector: np.ndarray
    text: str
    metadata: dict
    distance: Optional[float] = None
