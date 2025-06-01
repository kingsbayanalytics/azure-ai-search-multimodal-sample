import os
import logging
from typing import List, Dict, Any, Optional
import json

import aiohttp
from azure.identity.aio import DefaultAzureCredential

logger = logging.getLogger("promptflow")


class PromptFlowClient:
    """Asynchronous client for invoking a Prompt Flow endpoint."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        flow_name: Optional[str] = None,
        use_prompt_flow: Optional[bool] = None,
        credential: Optional[DefaultAzureCredential] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint or os.getenv("PROMPT_FLOW_ENDPOINT")
        self.flow_name = flow_name or os.getenv("PROMPT_FLOW_FLOW_NAME")
        self.api_key = api_key or os.getenv("PROMPT_FLOW_API_KEY")

        if use_prompt_flow is None:
            use_prompt_flow_env = os.getenv("USE_PROMPT_FLOW", "false")
            self.use_prompt_flow = str(use_prompt_flow_env).lower() in (
                "1",
                "true",
                "yes",
            )
        else:
            self.use_prompt_flow = use_prompt_flow

        if not self.api_key:
            self.credential = credential or DefaultAzureCredential()
            self.scope = "https://ml.azure.com/.default"
        else:
            self.credential = None

    async def run_flow(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the configured Prompt Flow with the provided messages."""
        if not self.use_prompt_flow:
            raise RuntimeError(
                "Prompt Flow usage is not enabled via USE_PROMPT_FLOW environment variable."
            )
        if not self.endpoint or not self.flow_name:
            raise ValueError(
                "PROMPT_FLOW_ENDPOINT and PROMPT_FLOW_FLOW_NAME must be set in environment variables."
            )

        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if self.api_key:
            logger.info("Using API key for Prompt Flow authentication.")
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.credential:
            logger.info("Using Azure AD token for Prompt Flow authentication.")
            try:
                token = await self.credential.get_token(self.scope)
                headers["Authorization"] = f"Bearer {token.token}"
            except Exception as e:
                logger.error(f"Failed to get Azure AD token: {e}")
                raise RuntimeError(f"Failed to get Azure AD token for Prompt Flow: {e}")
        else:
            raise RuntimeError(
                "Prompt Flow authentication not configured: API key (PROMPT_FLOW_API_KEY) or Azure AD credentials must be available."
            )

        url = f"{self.endpoint}/flows/{self.flow_name}:run"

        if self.api_key:
            if not self.endpoint.endswith("/score"):
                logger.warning(
                    f"PROMPT_FLOW_ENDPOINT ('{self.endpoint}') does not end with '/score'. For key-based auth, it usually should be the full scoring URL."
                )
            url = self.endpoint
        else:
            url = f"{self.endpoint}/flows/{self.flow_name}:run"

        logger.info("Calling Prompt Flow at %s", url)

        # Reconstruct payload for Prompt Flow
        # Assuming the 'messages' list from MultimodalRAG is structured like:
        # [system_message, user_q1, assistant_a1, ..., current_user_question, document_context_message]

        if len(messages) < 1:  # Should at least have the current user question
            logger.error(
                "Messages list is too short to extract question for Prompt Flow."
            )
            raise ValueError("Cannot extract question from messages for Prompt Flow.")

        # The actual user question is the second to last if docs are appended as the last message.
        # If no docs, it's the last one.
        # For simplicity, let's assume the prompt flow doesn't want the big doc context directly.
        # It should use the question to do its own RAG.

        # Try to find the last actual user question (not the document dump)
        actual_question_content = ""
        chat_history_for_pf = []

        # The 'messages' list from MultimodalRAG has:
        # System prompt
        # Chat history (alternating user/assistant)
        # Current user question (text)
        # Document context (also as a user message, but structured)

        # Let's find the last message that is a simple text query from the user.
        # The message containing documents is also role:user but its content is a list of dicts.
        # The actual user question's content is a list with one dict: [{"text": "...", "type": "text"}]

        # Default to the last message if it's a simple user text query
        if (
            messages
            and messages[-1].get("role") == "user"
            and isinstance(messages[-1].get("content"), list)
            and len(messages[-1].get("content")) == 1
            and messages[-1]["content"][0].get("type") == "text"
        ):
            actual_question_content = messages[-1]["content"][0]["text"]
            chat_history_for_pf = messages[:-1]
        elif (
            len(messages) > 1
            and messages[-2].get("role") == "user"
            and isinstance(messages[-2].get("content"), list)
            and len(messages[-2].get("content")) == 1
            and messages[-2]["content"][0].get("type") == "text"
        ):
            # This assumes the last message is the document context block
            actual_question_content = messages[-2]["content"][0]["text"]
            chat_history_for_pf = messages[:-2]  # Exclude question and doc context
        else:
            logger.warning(
                "Could not reliably determine the user's question from the messages structure for Prompt Flow. Sending all as history and an empty question."
            )
            actual_question_content = ""  # Fallback
            chat_history_for_pf = messages  # Send everything as history

        # Convert chat_history_for_pf to the {"role": ..., "content": ...} format if needed,
        # or the {"inputs": ..., "outputs": ...} format.
        # The check_promptflow.py script used an empty chat_history.
        # For now, let's pass an empty history, similar to the test script, and just the question.
        # This simplifies things and relies on the Prompt Flow to manage its own history/context if designed for it.

        simple_chat_history = []
        for msg in chat_history_for_pf:
            role = msg.get("role")
            content_list = msg.get("content")
            if (
                role
                and isinstance(content_list, list)
                and content_list
                and isinstance(content_list[0], dict)
                and "text" in content_list[0]
            ):
                simple_chat_history.append(
                    {"role": role, "content": content_list[0]["text"]}
                )
            # else, skip complex messages like image urls or the doc block for simple history

        payload = {
            "inputs": {
                "question": actual_question_content,
                "chat_history": simple_chat_history,  # Or [] if the flow doesn't use it or expects a different format
            }
        }
        # Override config can be added here if needed:
        # "config_override": { ... }

        logger.info(f"Prompt Flow Request URL: {url}")
        try:
            logger.info(
                f"Prompt Flow Request Payload:\n{json.dumps(payload, indent=2)}"
            )
        except TypeError as e:
            logger.warning(
                f"Could not serialize payload for logging: {e}. Payload: {payload}"
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_content = await resp.text()
                    logger.error(
                        f"Prompt Flow request failed with status {resp.status}: {error_content}"
                    )
                    resp.raise_for_status()
                return await resp.json()
