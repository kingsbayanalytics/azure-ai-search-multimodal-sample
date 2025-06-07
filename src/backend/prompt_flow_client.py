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
        self.use_prompt_flow = (
            use_prompt_flow
            if use_prompt_flow is not None
            else os.getenv("USE_PROMPT_FLOW", "false").lower() == "true"
        )
        self.credential = credential
        self.api_key = api_key or os.getenv("PROMPT_FLOW_API_KEY")

        # Field name customization through environment variables
        self.request_field_name = os.getenv(
            "PROMPTFLOW_REQUEST_FIELD_NAME", "chat_input"
        )
        self.response_field_name = os.getenv("PROMPTFLOW_RESPONSE_FIELD_NAME", "output")
        self.citations_field_name = os.getenv(
            "PROMPTFLOW_CITATIONS_FIELD_NAME", "citations"
        )

        logger.info(f"PromptFlow Client initialized with endpoint: {self.endpoint}")
        logger.info(f"PromptFlow Client initialized with flow name: {self.flow_name}")
        logger.info(
            f"PromptFlow Client initialized with use_prompt_flow: {self.use_prompt_flow}"
        )
        logger.info(f"Using request field name: {self.request_field_name}")
        logger.info(f"Using response field name: {self.response_field_name}")
        logger.info(f"Using citations field name: {self.citations_field_name}")

    def is_enabled(self) -> bool:
        """Check if the Prompt Flow client is enabled."""
        return self.use_prompt_flow and bool(self.endpoint)

    async def get_token(self) -> str:
        """Get an authorization token using the DefaultAzureCredential."""
        if not self.credential:
            self.credential = DefaultAzureCredential()

        token = await self.credential.get_token("https://ml.azure.com/.default")
        return token.token

    def create_payload(
        self, query: str, history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Create the payload for the Prompt Flow endpoint."""
        request_field = self.request_field_name

        logger.info(f"Creating payload with request field name: {request_field}")
        logger.info(f"Query: {query[:100]}...")
        logger.info(f"History: {json.dumps(history[:2], indent=2)}")

        # Create a payload using the custom field name
        payload = {
            "inputs": {
                request_field: query,
                "chat_history": history,
            }
        }

        logger.info(f"Final payload: {json.dumps(payload, indent=2)}")
        return payload

    async def call_endpoint(
        self, query: str, history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Call the Prompt Flow endpoint with the given query."""
        if not self.endpoint or not self.api_key:
            logger.error("Prompt Flow endpoint or API key not set")
            return {"error": "Prompt Flow endpoint or API key not set"}

        if history is None:
            history = []

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Use the create_payload method to build the payload
        payload = self.create_payload(query, history)

        logger.info(f"Calling Prompt Flow endpoint: {self.endpoint}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        # For key-based auth, the endpoint might already be the full scoring URL
        # or it might need /flows/{flow_name}:run appended.
        if (
            self.endpoint and "/score" in self.endpoint.lower()
        ):  # Check if it's a managed online endpoint scoring URI
            url = self.endpoint
            logger.info(f"Using direct scoring URL: {url}")
        elif (
            self.endpoint and self.flow_name
        ):  # Assume it's a base AML endpoint, append flow name
            url = f"{self.endpoint}/flows/{self.flow_name}:run"
            logger.info(f"Constructed run URL: {url}")
        else:
            logger.error("Invalid Prompt Flow endpoint configuration")
            return {"error": "Invalid Prompt Flow endpoint configuration"}

        try:
            async with aiohttp.ClientSession() as session:
                logger.info(
                    f"Sending request to {url} with payload: {json.dumps(payload)}"
                )
                async with session.post(url, json=payload, headers=headers) as resp:
                    logger.info(f"Response status: {resp.status}")
                    resp.raise_for_status()

                    # Get response text first to log it
                    response_text = await resp.text()
                    logger.info(f"Raw response text: {response_text}")

                    # Parse to JSON and return
                    try:
                        response_json = json.loads(response_text)
                        logger.info(
                            f"Parsed response JSON: {json.dumps(response_json, indent=2)}"
                        )
                        return response_json
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse response as JSON: {e}")
                        return {
                            "error": f"Failed to parse response as JSON: {e}",
                            "raw_response": response_text,
                        }
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection Error: Failed to connect to {url}. Details: {e}")
            return {
                "error": f"Connection Error: Failed to connect to {url}. Details: {e}"
            }
        except aiohttp.ClientResponseError as e:
            logger.error(f"Response Error: {e.status} {e.message}")
            return {"error": f"Response Error: {e.status} {e.message}"}
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}
