from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class DocInfo:
    path: str
    content: str
    sha: str
    repo_name: str
    last_modified: Optional[datetime] = None
    author: Optional[str] = None