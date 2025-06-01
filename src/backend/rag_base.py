import logging
import json
import os
import time
from typing import List
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

logger = logging.getLogger("rag")


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
            chunk_count=config_dict.get("chunk_count", 10),
            openai_api_mode=config_dict.get("openai_api_mode", "chat_completions"),
            use_semantic_ranker=config_dict.get("use_semantic_ranker", False),
            use_streaming=config_dict.get("use_streaming", False),
            use_knowledge_agent=config_dict.get("use_knowledge_agent", False),
            use_prompt_flow=config_dict.get("use_prompt_flow", False),
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
    ):
        """Handles streaming chat completion and sends citations."""

        logger.info("Formulating LLM response")
        await self._send_processing_step_message(
            request_id,
            response,
            ProcessingStep(title="LLM Payload", type="code", content=messages),
        )

        complete_response: dict = {}

        if search_config.get("use_prompt_flow", False):
            logger.info("Calling Prompt Flow endpoint")
            if not self.prompt_flow_client:
                raise RuntimeError("Prompt Flow client is not configured")

            raw_pf_response_data = await self.prompt_flow_client.run_flow(messages)

            # Ensure raw_pf_response_data is a dictionary
            parsed_pf_response = None
            if isinstance(raw_pf_response_data, str):
                try:
                    parsed_pf_response = json.loads(raw_pf_response_data)
                    logger.info(
                        f"Successfully parsed raw_pf_response_data string into dict."
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse raw_pf_response_data string as JSON: {e}. Data: {raw_pf_response_data}"
                    )
                    parsed_pf_response = {
                        "output": {
                            "output": f"Error: Could not parse Prompt Flow response JSON: {e}"
                        }
                    }  # Set error
            elif isinstance(raw_pf_response_data, dict):
                parsed_pf_response = raw_pf_response_data
            else:
                logger.error(
                    f"Unexpected type for raw_pf_response_data: {type(raw_pf_response_data)}. Data: {raw_pf_response_data}"
                )
                parsed_pf_response = {
                    "output": {"output": "Error: Unexpected Prompt Flow response type."}
                }  # Set error

            # Add direct logging of the raw response to the console
            try:
                logger.info(
                    f"Raw Prompt Flow Response received (rag_base.py, after potential parse):\n{json.dumps(parsed_pf_response, indent=2)}"
                )
                logger.info(f"Type of parsed_pf_response: {type(parsed_pf_response)}")

                # Debug: Log all top-level keys in the response
                if isinstance(parsed_pf_response, dict):
                    logger.info(
                        f"Top-level keys in parsed_pf_response: {list(parsed_pf_response.keys())}"
                    )

                    # Check different possible response structures
                    if "output" in parsed_pf_response:
                        output_dict = parsed_pf_response.get("output")
                        logger.info(
                            f'Type of parsed_pf_response.get("output"): {type(output_dict)}'
                        )
                        logger.info(
                            f'Value of parsed_pf_response.get("output"): {output_dict}'
                        )

                    # Check if the response might be directly at the top level
                    if (
                        "id" in parsed_pf_response
                        and "output" not in parsed_pf_response
                    ):
                        logger.info(
                            "Response appears to have id at top level - checking for direct structure"
                        )
                        if "citations" in parsed_pf_response:
                            logger.info("Found citations at top level")

                    # Check for error responses
                    if "error" in parsed_pf_response:
                        logger.error(
                            f"Prompt Flow returned an error: {parsed_pf_response.get('error')}"
                        )

            except TypeError as e:
                logger.warning(
                    f"Could not serialize parsed_pf_response for logging: {e}. Response: {parsed_pf_response}"
                )

            await self._send_processing_step_message(
                request_id,
                response,
                ProcessingStep(
                    title="Parsed Prompt Flow Response",  # Changed title
                    type="code",
                    content=parsed_pf_response,  # Log parsed version
                ),
            )

            msg_id = str(uuid.uuid4())

            # Adapt to the new Prompt Flow output structure, using parsed_pf_response
            final_answer = ""
            pf_citations = []

            if isinstance(parsed_pf_response, dict):
                # Check for nested output structure (expected format)
                if "output" in parsed_pf_response and isinstance(
                    parsed_pf_response["output"], dict
                ):
                    output_level1 = parsed_pf_response["output"]
                    final_answer = output_level1.get("output", "")
                    pf_citations = output_level1.get("citations", [])
                    logger.info("Found answer in nested output structure")

                # Check for chat_output structure (another common format)
                elif "chat_output" in parsed_pf_response:
                    chat_output = parsed_pf_response["chat_output"]
                    if isinstance(chat_output, str):
                        final_answer = chat_output
                        logger.info("Found answer in chat_output as string")
                    elif isinstance(chat_output, dict):
                        # Check if chat_output has nested structure
                        final_answer = chat_output.get(
                            "output", chat_output.get("answer", str(chat_output))
                        )
                        pf_citations = chat_output.get("citations", [])
                        logger.info("Found answer in chat_output as dict")
                    pf_citations = parsed_pf_response.get("citations", [])

                # Check for direct structure (output at top level)
                elif "output" in parsed_pf_response and isinstance(
                    parsed_pf_response["output"], str
                ):
                    final_answer = parsed_pf_response["output"]
                    pf_citations = parsed_pf_response.get("citations", [])
                    logger.info("Found answer in direct output structure")

                # Check for response with id at top level (another possible format)
                elif "id" in parsed_pf_response and not "output" in parsed_pf_response:
                    # The actual output might be in a different key
                    for key in ["answer", "response", "result", "text"]:
                        if key in parsed_pf_response:
                            final_answer = parsed_pf_response[key]
                            logger.info(f"Found answer in '{key}' field")
                            break
                    pf_citations = parsed_pf_response.get("citations", [])

                # Check for error response
                elif "error" in parsed_pf_response:
                    final_answer = (
                        f"Error from Prompt Flow: {parsed_pf_response['error']}"
                    )
                    logger.error(f"Prompt Flow error response: {parsed_pf_response}")

                # Fallback for unexpected structure
                else:
                    logger.error(
                        f"Unexpected Prompt Flow response structure. Keys: {list(parsed_pf_response.keys())}"
                    )
                    final_answer = (
                        "Error: Unable to parse Prompt Flow response structure"
                    )

            else:
                final_answer = "Error: parsed_pf_response is not a dict"
                logger.error(
                    f"parsed_pf_response is not a dict: {type(parsed_pf_response)}"
                )

            logger.info(
                f"Extracted final_answer from Prompt Flow (rag_base.py): '{final_answer}'"
            )

            # Convert Prompt Flow citations to the format expected by the frontend
            text_citations = []
            image_citations = []

            for citation_obj in pf_citations:
                # Extract relevant fields from the citation
                citation = {
                    "id": citation_obj.get("id", ""),
                    "title": citation_obj.get("title", ""),
                    "filepath": citation_obj.get("filepath", ""),
                    "url": citation_obj.get("url", ""),
                    "content": citation_obj.get("content", ""),
                    "chunk_id": citation_obj.get("chunk_id"),
                    "reindex_id": citation_obj.get("reindex_id"),
                }

                # Determine if it's an image or text citation based on filepath
                filepath = citation_obj.get("filepath", "").lower()
                if any(
                    ext in filepath for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
                ):
                    image_citations.append(citation)
                else:
                    text_citations.append(citation)

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

        elif search_config.get("use_streaming", False):
            logger.info("Streaming chat completion")
            chat_stream_response = instructor.from_openai(
                self.openai_client,
            ).chat.completions.create_partial(
                stream=True,
                model=self.chatcompletions_model_name,
                response_model=AnswerFormat,
                messages=messages,
            )
            msg_id = str(uuid.uuid4())

            async for stream_response in chat_stream_response:
                if stream_response.answer is not None:
                    await self._send_answer_message(
                        request_id, response, msg_id, stream_response.answer
                    )
                    complete_response = stream_response.model_dump()
            if len(complete_response.keys()) == 0:
                raise ValueError("No response received from chat completion stream.")

        else:
            logger.info("Waiting for chat completion")
            chat_completion = await instructor.from_openai(
                self.openai_client,
            ).chat.completions.create(
                stream=False,
                model=self.chatcompletions_model_name,
                response_model=AnswerFormat,
                messages=messages,
            )
            msg_id = str(uuid.uuid4())

            if chat_completion is not None:
                await self._send_answer_message(
                    request_id, response, msg_id, chat_completion.answer
                )
                complete_response = chat_completion.model_dump()
            else:
                raise ValueError("No response received from chat completion stream.")

        await self._extract_and_send_citations(
            request_id,
            response,
            grounding_retriever,
            grounding_results["references"],
            complete_response["text_citations"] or [],
            complete_response["image_citations"] or [],
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
        await self._send_message(
            response,
            MessageType.ProcessingStep.value,
            {
                "request_id": request_id,
                "message_id": str(uuid.uuid4()),
                "processingStep": processing_step.to_dict(),
            },
        )

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
            message_to_send = f"event:{event}\ndata: {json.dumps(data)}\n\n"
            logger.info(
                f"Attempting to send SSE message (rag_base.py _send_message):\n{message_to_send.strip()}"
            )
            await response.write(message_to_send.encode("utf-8"))
        except ConnectionResetError:
            # TODO: Something is wrong here, the messages attempted and failed here is not what the UI sees, thats another set of stream...
            # logger.warning("Connection reset by client.")
            pass
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def _send_end(self, response):
        await self._send_message(response, MessageType.END.value, {})

    def attach_to_app(self, app, path):
        """Attaches the handler to the web app."""
        app.router.add_post(path, self._handle_request)
