from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging
from openai import OpenAI
from ragnadoc.docs import DocEmbedding
from ragnadoc.vectors import VectorStore


@dataclass
class QueryResult:
    answer: str
    sources: List[Dict[str, Any]]
    raw_response: Any


class QueryEngine:
    """Engine for querying documentation and generating answers."""

    DEFAULT_SYSTEM_PROMPT = """
    You are a technical documentation assistant. Your role is to:
    1. Answer questions based solely on the provided documentation
    2. Cite specific sources for your information
    3. Admit when you can't find relevant information
    4. Keep responses clear and concise
    5. Use markdown formatting for code and technical terms
    
    Format code examples using triple backticks with the language specified.
    """

    DEFAULT_QUERY_PROMPT = """
    Answer the following question using only the provided documentation excerpts.
    If the documentation doesn't contain enough information to answer fully, say so.
    
    Documentation excerpts:
    {context}
    
    Question: {question}
    
    Provide your answer, citing relevant sources.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        openai_api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        query_prompt: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.query_prompt = query_prompt or self.DEFAULT_QUERY_PROMPT
        self.logger = logging.getLogger(__name__)

    def _format_context(self, docs: List[DocEmbedding]) -> str:
        context_parts = []
        for doc in docs:
            source = f"{doc.metadata['repo']}/{doc.metadata['source']}"
            # include relevance score if, available
            score_info = f" (relevance: {doc.distance:.2f})" if doc.distance is not None else ""
            context_parts.append(
                f"Source: {source}{score_info}\n"
                f"Content: {doc.text}\n"
            )
        return "\n---\n".join(context_parts)

    def _create_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.query_prompt.format(
                    context=context,
                    question=question
                )
            }
        ]

    def query(
        self,
        question: str,
        filter_dict: Optional[Dict] = None,
        top_k: int = 4,
        min_relevance_score: float = 0.5
    ) -> QueryResult:
        ""
        try:
            # retrieve relevant documents
            docs = self.vector_store.query(
                query=question,
                filter_dict=filter_dict,
                top_k=top_k
            )

            # filter by relevance score if specified
            if min_relevance_score is not None:
                docs = [
                    doc for doc in docs
                    if doc.distance is None or doc.distance >= min_relevance_score
                ]

            if not docs:
                return QueryResult(
                    answer="I couldn't find any relevant documentation to answer your question.",
                    sources=[],
                    raw_response=None
                )

            # format context from documents
            context = self._format_context(docs)

            # create messages for chat completion
            messages = self._create_messages(question, context)

            # generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # extract sources from used documents
            sources = [
                {
                    "repo": doc.metadata["repo"],
                    "path": doc.metadata["source"],
                    "relevance": doc.distance,
                    "last_modified": doc.metadata.get("last_modified"),
                    "author": doc.metadata.get("author")
                }
                for doc in docs
            ]

            return QueryResult(
                answer=response.choices[0].message.content or "",
                sources=sources,
                raw_response=response
            )

        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            raise
