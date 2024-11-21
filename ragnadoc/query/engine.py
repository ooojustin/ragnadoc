from typing import Dict, Optional, Generic, Union, TypeVar, Iterator
from openai import OpenAI
from openai.types.beta import Thread, Assistant
from openai.types.beta.threads import TextContentBlock, TextDeltaBlock
from openai.types.beta.assistant_stream_event import ThreadMessageDelta
from ragnadoc.content import Content
from ragnadoc.vectors import VectorStore
from ragnadoc.query.models import QueryResult, ChatQueryResult, StreamChatQueryResult
import time
import logging

T = TypeVar("T", bound=Content)


class QueryEngine(Generic[T]):
    """Engine for querying documentation and generating answers."""

    DEFAULT_SYSTEM_PROMPT = """
    You are a technical documentation assistant. Your role is to:
    1. Answer questions based solely on the provided documentation
    2. Cite specific sources for your information (excluding the source content)
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
    
    Provide your answer, citing relevant sources (excluding the source content).
    """

    def __init__(
        self,
        vector_store: VectorStore[T],
        openai_api_key: str,
        model: str = "gpt-4",
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

        self.assistant = self._create_assistant()
        self.thread = self._create_thread()

    def search(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        top_k: int = 4,
        min_relevance_score: Optional[float] = None
    ) -> QueryResult[T]:
        start_time = time.perf_counter()

        search_result = self.vector_store.search(
            query=query,
            filter_dict=filter_dict,
            top_k=top_k
        )

        result = QueryResult[T](
            documents=search_result.documents,
            scores=search_result.scores,
            query=query,
            total_found=len(search_result.documents),
            filter_dict=filter_dict,
            query_time_ms=(time.perf_counter() - start_time) * 1000
        )

        if min_relevance_score is not None:
            result = result.filter_by_score(min_relevance_score)

        return result

    def _format_context(self, result: QueryResult[T]) -> str:
        context_parts = []
        for doc, score in zip(result.documents, result.scores):
            source_info = f"{doc.id}" if doc.id else "Unknown Source"
            score_info = f" (relevance: {score:.2f})" if score is not None else ""
            context_parts.append(
                f"Source: {source_info}{score_info}\n"
                f"Content: {doc.text}\n"
            )
        return "\n---\n".join(context_parts)

    def query(
        self,
        question: str,
        filter_dict: Optional[Dict] = None,
        top_k: int = 4,
        min_relevance_score: float = 0.5,
        stream: bool = False
    ) -> Union[ChatQueryResult[T], StreamChatQueryResult[T]]:
        """Query method for QueryEngine class."""
        try:
            result = self.search(
                query=question,
                filter_dict=filter_dict,
                top_k=top_k,
                min_relevance_score=min_relevance_score
            )

            if not result.documents:
                message_str = "I couldn't find any relevant documentation to answer your question."
                message = self.client.beta.threads.messages.create(
                    thread_id=self.thread.id,
                    role="user",
                    content=message_str
                )
                return ChatQueryResult.create(
                    answer=message_str,
                    query_result=result,
                    raw_response=message
                )

            context = self._format_context(result)
            formatted_query = self.query_prompt.format(
                context=context,
                question=question
            )

            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=formatted_query
            )

            if stream:
                def stream_generator() -> Iterator[str]:
                    stream = self.client.beta.threads.runs.create(
                        thread_id=self.thread.id,
                        assistant_id=self.assistant.id,
                        stream=True
                    )
                    for i in stream:
                        if isinstance(i, ThreadMessageDelta):
                            content = i.data.delta.content
                            assert content
                            block = content[0]
                            if isinstance(block, TextDeltaBlock):
                                if block.text:
                                    yield str(block.text.value)
                return StreamChatQueryResult.create_streaming(
                    stream_generator=stream_generator,
                    query_result=result,
                    raw_response=None
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

            try:
                response = messages.data[0]
                content = response.content[0]
                assert isinstance(content, TextContentBlock)
                answer = content.text.value
            except (AssertionError, AttributeError, IndexError):
                answer = "Unable to interpret response as text."

            return ChatQueryResult.create(
                answer=answer,
                query_result=result,
                raw_response=messages.data[0]
            )

        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            from rich.traceback import Traceback
            from ragnadoc.main import console
            console.print(Traceback())
            raise

    def _create_assistant(self) -> Assistant:
        assistant = self.client.beta.assistants.create(
            name="Documentation Assistant",
            instructions=self.system_prompt,
            model=self.model,
            tools=[]
        )
        self.logger.debug(f"Created new assistant ({assistant.id})")
        return assistant

    def _create_thread(self) -> Thread:
        thread = self.client.beta.threads.create()
        self.logger.debug(f"Created new conversation thread ({thread.id})")
        return thread
