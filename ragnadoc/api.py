from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ragnadoc.query.engine import QueryEngine
from ragnadoc.content import GitContent
import uvicorn


class QueryRequest(BaseModel):
    question: str
    repo: Optional[str] = None
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.7
    stream: Optional[bool] = False


class SourceInfo(BaseModel):
    id: str
    repo: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]


class RagnadocAPI:
    def __init__(self, query_engine: QueryEngine[GitContent]):
        self.app = FastAPI(title="Ragnadoc API", version="1.0.0")
        self.query_engine = query_engine
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            """Health check"""
            return {"status": "ok", "service": "ragnadoc"}

        @self.app.post("/query", response_model=QueryResponse)
        async def query(request: QueryRequest) -> Dict[str, Any]:
            """Query the documentation"""
            try:
                filter_dict = {"repo": request.repo} if request.repo else None

                result = self.query_engine.query(
                    question=request.question,
                    filter_dict=filter_dict,
                    top_k=request.top_k or 5,
                    min_relevance_score=request.min_score or 0.7,
                    stream=request.stream or False
                )

                sources = [
                    SourceInfo(
                        id=doc.id or "unknown",
                        repo=doc.repo,
                        score=score
                    )
                    for doc, score in zip(
                        result.query_result.documents,
                        result.query_result.scores
                    )
                ]

                return {
                    "answer": result.answer,
                    "sources": sources
                }

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing query: {str(e)}"
                )

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server."""
        uvicorn.run(self.app, host=host, port=port)
