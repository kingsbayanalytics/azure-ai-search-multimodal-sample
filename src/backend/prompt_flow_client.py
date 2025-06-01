import os
import logging
from typing import List, Dict, Any, Optional

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
        payload = {"messages": messages}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_content = await resp.text()
                    logger.error(
                        f"Prompt Flow request failed with status {resp.status}: {error_content}"
                    )
                    resp.raise_for_status()
                return await resp.json()
