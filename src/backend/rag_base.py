import logging
import json
import os
import time
from typing import List, Dict, Any, Optional
from prompt_flow_client import PromptFlowClient
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from aiohttp import web
import instructor
from openai import AsyncAzureOpenAI
from grounding_retriever import GroundingRetriever
from models import (
    AnswerFormat,
    SearchConfig,
    GroundingResult,
    GroundingResults,
)
from processing_step import ProcessingStep
from typing_extensions import TypedDict

logger = logging.getLogger("multimodal-rag")


# Type definitions
class GroundingResults(TypedDict, total=False):
    """Results from the grounding retriever."""

    passages: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    references: List[Dict[str, Any]]
    search_queries: List[str]


class SearchConfig(TypedDict, total=False):
    """Configuration for the search."""

    query: str
    use_prompt_flow: bool


class ProcessingStep(TypedDict):
    """A step in the processing pipeline."""

    title: str
    type: str
    content: Any


class MessageType(Enum):
    ANSWER = "answer"
    CITATION = "citation"
    LOG = "log"
    ERROR = "error"
    END = "[END]"
    ProcessingStep = "processing_step"
    INFO = "info"


class RagBase(ABC):
    def __init__(
        self,
        openai_client: AsyncAzureOpenAI,
        chatcompletions_model_name: str,
        prompt_flow_client: PromptFlowClient = None,
    ):
        self.openai_client = openai_client
        self.chatcompletions_model_name = chatcompletions_model_name
        self.prompt_flow_client = prompt_flow_client

    async def _handle_request(self, request: web.Request):
        request_params = await request.json()
        search_text = request_params.get("query", "")
        chat_thread = request_params.get("chatThread", [])
        config_dict = request_params.get("config", {})
        search_config = SearchConfig(
            openai_api_mode=config_dict.get("openai_api_mode", "chat_completions"),
        )
        request_id = request_params.get("request_id", str(int(time.time())))
        response = await self._create_stream_response(request)
        try:
            await self._process_request(
                request_id, response, search_text, chat_thread, search_config
            )
        except Exception as e:
            print(e)
            logger.error(f"Error processing request: {str(e)}")
            await self._send_error_message(request_id, response, str(e))

        await self._send_end(response)
        return response

    @abstractmethod
    async def _process_request(
        self,
        request_id: str,
        response: web.StreamResponse,
        search_text: str,
        chat_thread: list,
        search_config: SearchConfig,
    ):
        pass

    async def _formulate_response(
        self,
        request_id: str,
        response: web.StreamResponse,
        messages: list,
        grounding_retriever: GroundingRetriever,
        grounding_results: GroundingResults,
        search_config: SearchConfig,
        msg_id: str = None,
        parsed_pf_response: Any = None,
    ) -> None:
        """Formulate a response to send back to the client based on the response from the model."""
        if msg_id is None:
            msg_id = str(uuid.uuid4())

        if parsed_pf_response is not None:
            # Adapt to the new Prompt Flow output structure, using parsed_pf_response
            final_answer = ""
            pf_citations = []

            # Get response and citations field names from the PromptFlow client
            response_field = self.prompt_flow_client.response_field_name
            citations_field = self.prompt_flow_client.citations_field_name

            logger.info(
                f"Looking for response in field '{response_field}' and citations in field '{citations_field}'"
            )

            # The response structure from PromptFlow is:
            # {
            #   "chat_output": {
            #     "citations": [...],
            #     "output": "Answer text"
            #   }
            # }
            if "chat_output" in parsed_pf_response and isinstance(
                parsed_pf_response["chat_output"], dict
            ):
                chat_output = parsed_pf_response["chat_output"]
                logger.info(
                    f"Found chat_output structure with keys: {list(chat_output.keys())}"
                )

                # Extract the answer
                if response_field in chat_output:
                    final_answer = chat_output[response_field]
                    logger.info(f"Found response in chat_output.{response_field}")
                else:
                    logger.warning(
                        f"Could not find response field '{response_field}' in chat_output, available fields: {list(chat_output.keys())}"
                    )

                # Extract citations
                if citations_field in chat_output:
                    pf_citations = chat_output[citations_field]
                    logger.info(
                        f"Found {len(pf_citations)} citations in chat_output.{citations_field}"
                    )
                else:
                    logger.warning(
                        f"Could not find citations field '{citations_field}' in chat_output"
                    )
            else:
                logger.warning(
                    f"Could not find expected chat_output structure in PromptFlow response. Keys: {list(parsed_pf_response.keys())}"
                )

                # Try to find the response in the top level
                if response_field in parsed_pf_response:
                    final_answer = parsed_pf_response[response_field]
                    logger.info(f"Found response in top-level {response_field} field")

            # Convert PromptFlow citations to the format expected by the frontend
            text_citations = []
            image_citations = []

            for citation in pf_citations:
                try:
                    # Check if this is a valid citation
                    if isinstance(citation, dict) and "content" in citation:
                        citation_item = {
                            "content": citation.get("content", ""),
                            "title": citation.get("title", ""),
                            "filepath": citation.get("filepath", ""),
                            "url": citation.get("url", ""),
                        }
                        text_citations.append(citation_item)
                except Exception as e:
                    logger.error(f"Error processing citation: {e}", exc_info=True)

            # Send answer message with type "answer" and include citations
            await self._send_answer_message(
                request_id,
                response,
                msg_id,
                final_answer,
                text_citations=text_citations,
                image_citations=image_citations,
            )

            # Don't call _extract_and_send_citations when using prompt flow
            return

        # Handle non-PromptFlow responses
        complete_response = {}

        logger.info("Streaming chat completion")
        chat_stream_response = instructor.from_openai(
            self.openai_client,
        ).chat.completions.create_partial(
            stream=True,
            model=self.chatcompletions_model_name,
            response_model=AnswerFormat,
            messages=messages,
        )

        async for stream_response in chat_stream_response:
            if stream_response.answer is not None:
                await self._send_answer_message(
                    request_id, response, msg_id, stream_response.answer
                )
                complete_response = stream_response.model_dump()
        if len(complete_response.keys()) == 0:
            raise ValueError("No response received from chat completion stream.")

        # Extract citations if we have any
        if (
            grounding_retriever
            and "text_citations" in complete_response
            and "image_citations" in complete_response
        ):
            await self._extract_and_send_citations(
                request_id,
                response,
                grounding_retriever,
                grounding_results.get("references", []),
                complete_response.get("text_citations") or [],
                complete_response.get("image_citations") or [],
            )

    async def _extract_and_send_citations(
        self,
        request_id: str,
        response: web.StreamResponse,
        grounding_retriever: GroundingRetriever,
        grounding_results: List[GroundingResult],
        text_citation_ids: list,
        image_citation_ids: list,
    ):
        """Extracts and sends citations from search results."""
        citations = await self.extract_citations(
            grounding_retriever,
            grounding_results,
            text_citation_ids,
            image_citation_ids,
        )

        await self._send_citation_message(
            request_id,
            response,
            request_id,
            citations.get("text_citations", []),
            citations.get("image_citations", []),
        )

    @abstractmethod
    async def extract_citations(
        self,
        grounding_retriever: GroundingRetriever,
        grounding_results: List[GroundingResult],
        text_citation_ids: list,
        image_citation_ids: list,
    ) -> dict:
        pass

    async def _create_stream_response(self, request):
        """Creates and prepares the SSE stream response."""
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Connection": "keep-alive",
                "Cache-Control": "no-cache, no-transform",
            },
        )
        await response.prepare(request)
        return response

    async def _send_error_message(
        self, request_id: str, response: web.StreamResponse, message: str
    ):
        """Sends an error message through the stream."""
        await self._send_message(
            response,
            MessageType.ERROR.value,
            {
                "request_id": request_id,
                "message_id": str(uuid.uuid4()),
                "message": message,
            },
        )

    async def _send_info_message(
        self,
        request_id: str,
        response: web.StreamResponse,
        message: str,
        details: str = None,
    ):
        """Sends an info message through the stream."""
        await self._send_message(
            response,
            MessageType.INFO.value,
            {
                "request_id": request_id,
                "message_id": str(uuid.uuid4()),
                "message": message,
                "details": details,
            },
        )

    async def _send_processing_step_message(
        self,
        request_id: str,
        response: web.StreamResponse,
        processing_step: ProcessingStep,
    ):
        logger.info(
            f"Sending processing step message for step: {processing_step.title}"
        )

        # Convert the processing step to a dictionary
        step_dict = processing_step.to_dict()

        # Log the full content for debugging
        try:
            content_str = json.dumps(step_dict["content"], indent=2)
            logger.info(f"Processing step content (truncated): {content_str[:1000]}")
            logger.info(f"Content type: {type(step_dict['content'])}")

            if step_dict["title"] == "Prompt Flow Response":
                logger.info("Found Prompt Flow Response step")
                logger.info(f"Full content length: {len(content_str)}")
        except (TypeError, KeyError) as e:
            logger.error(f"Error serializing processing step content: {e}")

        # Create the message
        message = {
            "request_id": request_id,
            "message_id": str(uuid.uuid4()),
            "processingStep": step_dict,
        }

        # Send the message
        await self._send_message(response, MessageType.ProcessingStep.value, message)

    async def _send_answer_message(
        self,
        request_id: str,
        response: web.StreamResponse,
        message_id: str,
        content: str,
        text_citations: list = [],
        image_citations: list = [],
    ):
        logger.info(
            f"Sending answer message (rag_base.py _send_answer_message) - content: '{content}'"
        )
        await self._send_message(
            response,
            MessageType.ANSWER.value,
            {
                "request_id": request_id,
                "message_id": message_id,
                "role": "assistant",
                "answerPartial": {"answer": content},
                "textCitations": text_citations,
                "imageCitations": image_citations,
            },
        )

    async def _send_citation_message(
        self,
        request_id: str,
        response: web.StreamResponse,
        message_id: str,
        text_citations: list,
        image_citations: list,
    ):

        await self._send_message(
            response,
            MessageType.CITATION.value,
            {
                "request_id": request_id,
                "message_id": message_id,
                "textCitations": text_citations,
                "imageCitations": image_citations,
            },
        )

    async def _send_message(self, response, event, data):
        try:
            message_json = json.dumps(data)
            message_size = len(message_json)

            # Log message details
            logger.info(
                f"Sending SSE message type: {event}, size: {message_size} bytes"
            )

            # Log truncated message for debugging
            if message_size > 1000:
                logger.info(
                    f"Message content (truncated): {message_json[:500]}...{message_json[-500:]}"
                )
            else:
                logger.info(f"Message content: {message_json}")

            message_to_send = f"event:{event}\ndata: {message_json}\n\n"

            # Log the actual message being sent
            logger.info(
                f"Attempting to send SSE message (rag_base.py _send_message), size: {len(message_to_send)} bytes"
            )

            await response.write(message_to_send.encode("utf-8"))
            logger.info(f"Successfully sent message of type {event}")
        except ConnectionResetError:
            # TODO: Something is wrong here, the messages attempted and failed here is not what the UI sees, thats another set of stream...
            # logger.warning("Connection reset by client.")
            pass
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)

    async def _send_end(self, response):
        await self._send_message(response, MessageType.END.value, {})

    def attach_to_app(self, app, path):
        """Attaches the handler to the web app."""
        app.router.add_post(path, self._handle_request)

    async def _process_message(
        self,
        request_id: str,
        response: web.StreamResponse,
        messages: list,
        grounding_retriever: GroundingRetriever,
        grounding_results: GroundingResults,
        search_config: SearchConfig,
    ):
        """Handles streaming chat completion and sends citations."""

        logger.info("Formulating LLM response")
        await self._send_processing_step_message(
            request_id,
            response,
            ProcessingStep(title="LLM Payload", type="code", content=messages),
        )

        msg_id = str(uuid.uuid4())

        if search_config.get("use_prompt_flow", False):
            logger.info("Calling Prompt Flow endpoint")
            if not self.prompt_flow_client:
                raise RuntimeError("Prompt Flow client is not configured")

            # Extract the user's query from the messages
            user_query = ""
            for message in reversed(messages):
                if message.get("role") == "user" and isinstance(
                    message.get("content"), list
                ):
                    for content_item in message.get("content", []):
                        if content_item.get("type") == "text":
                            user_query = content_item.get("text", "")
                            break
                    if user_query:
                        break

            # Create a simplified chat history for the PromptFlow endpoint
            chat_history = []
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    user_msg = messages[i]
                    assistant_msg = messages[i + 1]

                    user_content = ""
                    assistant_content = ""

                    # Extract user content
                    if user_msg.get("role") == "user" and isinstance(
                        user_msg.get("content"), list
                    ):
                        for content_item in user_msg.get("content", []):
                            if content_item.get("type") == "text":
                                user_content = content_item.get("text", "")
                                break

                    # Extract assistant content
                    if assistant_msg.get("role") == "assistant" and isinstance(
                        assistant_msg.get("content"), list
                    ):
                        for content_item in assistant_msg.get("content", []):
                            if content_item.get("type") == "text":
                                assistant_content = content_item.get("text", "")
                                break

                    if user_content and assistant_content:
                        chat_history.append({"role": "user", "content": user_content})
                        chat_history.append(
                            {"role": "assistant", "content": assistant_content}
                        )

            # Call the PromptFlow endpoint with the extracted query and history
            raw_pf_response = await self.prompt_flow_client.call_endpoint(
                user_query, chat_history
            )

            # Ensure raw_pf_response is a dictionary
            parsed_pf_response = None
            if isinstance(raw_pf_response, str):
                try:
                    parsed_pf_response = json.loads(raw_pf_response)
                    logger.info(
                        f"Successfully parsed raw_pf_response string into dict."
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse raw_pf_response string as JSON: {e}. Data: {raw_pf_response}"
                    )
                    parsed_pf_response = {
                        "error": f"Error: Could not parse Prompt Flow response JSON: {e}"
                    }
            elif isinstance(raw_pf_response, dict):
                parsed_pf_response = raw_pf_response
            else:
                logger.error(
                    f"Unexpected type for raw_pf_response: {type(raw_pf_response)}. Data: {raw_pf_response}"
                )
                parsed_pf_response = {
                    "error": "Error: Unexpected Prompt Flow response type."
                }

            # Log the response for debugging
            try:
                logger.info(
                    f"PromptFlow Response:\n{json.dumps(parsed_pf_response, indent=2)}"
                )
            except TypeError as e:
                logger.warning(
                    f"Could not serialize parsed_pf_response for logging: {e}"
                )

            # Send processing step message with the parsed response
            await self._send_processing_step_message(
                request_id,
                response,
                ProcessingStep(
                    title="Prompt Flow Response",
                    type="code",
                    content=parsed_pf_response,
                ),
            )

            # Process the response and send it to the client
            await self._formulate_response(
                request_id,
                response,
                messages,
                grounding_retriever,
                grounding_results,
                search_config,
                msg_id,
                parsed_pf_response,
            )
            return

        # If not using PromptFlow, continue with the existing code...
