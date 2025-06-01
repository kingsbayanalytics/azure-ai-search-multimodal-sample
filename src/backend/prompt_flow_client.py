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
    ) -> None:
        self.endpoint = endpoint or os.getenv("PROMPT_FLOW_ENDPOINT")
        self.flow_name = flow_name or os.getenv("PROMPT_FLOW_FLOW_NAME")
        if use_prompt_flow is None:
            use_prompt_flow = os.getenv("USE_PROMPT_FLOW", "false")
        self.use_prompt_flow = str(use_prompt_flow).lower() in ("1", "true", "yes")
        self.credential = credential or DefaultAzureCredential()
        self.scope = "https://ml.azure.com/.default"

    async def run_flow(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the configured Prompt Flow with the provided messages."""
        if not self.use_prompt_flow:
            raise RuntimeError("Prompt Flow usage is not enabled")
        if not self.endpoint or not self.flow_name:
            raise ValueError(
                "PROMPT_FLOW_ENDPOINT and PROMPT_FLOW_FLOW_NAME must be set"
            )

        token = await self.credential.get_token(self.scope)
        headers = {
            "Authorization": f"Bearer {token.token}",
            "Content-Type": "application/json",
        }
        url = f"{self.endpoint}/flows/{self.flow_name}:run"
        logger.info("Calling Prompt Flow at %s", url)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"messages": messages}, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json()

