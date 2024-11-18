from typing import Optional, Any, Dict, Sequence, TypeVar, Generic, List
from pydantic import BaseModel, Field, ConfigDict
from ragnadoc.content import Content, GitContent

T = TypeVar("T", bound=Content)


class QueryResult(BaseModel, Generic[T]):
    documents: Sequence[T]
    scores: Sequence[float]
    query: str
    total_found: int
    filter_dict: Optional[Dict[str, Any]] = None
    query_time_ms: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def best_match(self) -> Optional[tuple[T, float]]:
        return (self.documents[0], self.scores[0]) if self.documents else None

    def filter_by_score(self, min_score: float) -> 'QueryResult[T]':
        filtered = [(doc, score) for doc, score in zip(self.documents, self.scores)
                    if score >= min_score]
        docs, scores = zip(*filtered) if filtered else ([], [])

        return QueryResult[T](
            documents=list(docs),
            scores=list(scores),
            query=self.query,
            total_found=len(docs),
            filter_dict=self.filter_dict,
            query_time_ms=self.query_time_ms
        )


class GitQueryResult(QueryResult[GitContent]):
    @property
    def sources(self) -> List[str]:
        """Returns list of unique source documents."""
        return list(set(doc.source for doc in self.documents))

    @property
    def repositories(self) -> List[str]:
        return list(set(doc.repo for doc in self.documents))

    def get_matches_by_repo(self, repo: str) -> List[tuple[GitContent, float]]:
        return [(doc, score) for doc, score in zip(self.documents, self.scores)
                if doc.repo == repo]


class BaseQueryResult(BaseModel):
    answer: str
    raw_response: Any = Field(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChatQueryResult(BaseQueryResult, Generic[T]):
    query_result: QueryResult[T]

    @property
    def sources(self) -> List[T]:
        return list(self.query_result.documents)

    @property
    def source_scores(self) -> Dict[str, float]:
        return {
            doc.id: score
            for doc, score in zip(self.query_result.documents, self.query_result.scores)
            if doc.id is not None
        }

    @classmethod
    def create(
        cls,
        answer: str,
        query_result: QueryResult[T],
        raw_response: Any = None
    ) -> 'ChatQueryResult[T]':
        return cls(
            answer=answer,
            query_result=query_result,
            raw_response=raw_response
        )
