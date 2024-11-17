from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging
from openai import OpenAI
from openai.types.beta import Thread, Assistant
from openai.types.beta.threads import TextContentBlock
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

        # create the chat assistant/thread
        self.assistant = self._create_assistant()
        self.thread = self._create_thread()

    def _format_context(self, docs: List[DocEmbedding]) -> str:
        context_parts = []
        for doc in docs:
            source = f"{doc.metadata['repo']}/{doc.metadata['source']}"
            score_info = f" (relevance: {doc.distance:.2f})" if doc.distance is not None else ""
            context_parts.append(
                f"Source: {source}{score_info}\n"
                f"Content: {doc.text}\n"
            )
        return "\n---\n".join(context_parts)

    def query(
        self,
        question: str,
        filter_dict: Optional[Dict] = None,
        top_k: int = 4,
        min_relevance_score: float = 0.5
    ) -> QueryResult:
        """
        Query the documentation using a persistent thread and assistant.
        """
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
                message_str = "I couldn't find any relevant documentation to answer your question."
                message = self.client.beta.threads.messages.create(
                    thread_id=self.thread.id,
                    role="user",
                    content=message_str
                )
                return QueryResult(
                    answer=message_str,
                    sources=[],
                    raw_response=message
                )

            context = self._format_context(docs)
            formatted_query = self.query_prompt.format(
                context=context,
                question=question
            )

            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=formatted_query
            )
            self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
            )

            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id,
                order="desc",
                limit=1
            )

            response = messages.data[0]
            content = response.content[0]
            if isinstance(content, TextContentBlock):
                answer = content.text.value
            else:
                answer = "Unable to interpret response as text."

            # extract sources from used documents
            sources = [
                {
                    "id": doc.id,
                    "repo": doc.metadata["repo"],
                    "path": doc.metadata["source"],
                    "relevance": doc.distance,
                    "last_modified": doc.metadata.get("last_modified"),
                    "author": doc.metadata.get("author")
                }
                for doc in docs
            ]

            return QueryResult(
                answer=answer,
                sources=sources,
                raw_response=response
            )

        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            raise

    def _create_assistant(self) -> Assistant:
        return self.client.beta.assistants.create(
            name="Documentation Assistant",
            instructions=self.system_prompt,
            model=self.model,
            tools=[]
        )

    def _create_thread(self) -> Thread:
        self.logger.debug("Created new conversation thread")
        return self.client.beta.threads.create()

    def reset_conversation(self):
        self.logger.debug("Resetting QueryEngine conversation")
        self.thread = self._create_thread()

    def __del__(self):
        try:
            if hasattr(self, "assistant") and self.assistant:
                self.client.beta.assistants.delete(self.assistant.id)
        except Exception as e:
            self.logger.error(f"Error cleaning up assistant: {str(e)}")
