#!/usr/bin/env python3
"""
Simple script to test the Prompt Flow endpoint, API key, and flow name.
"""

import os
import asyncio
import aiohttp
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%X]",
)
logger = logging.getLogger(__name__)


async def check_prompt_flow():
    """Connects to the Prompt Flow endpoint and sends a test message."""
    try:
        from dotenv import load_dotenv

        # Try to load from several potential locations
        env_loaded = False
        project_root = Path(__file__).resolve().parent.parent

        potential_env_paths = [
            project_root / ".env",
            project_root
            / ".azure"
            / "my-multimodal-env"
            / ".env",  # Common azd env location
            project_root / "src" / "backend" / ".env",
        ]

        for env_path in potential_env_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                logger.info(f"Loaded environment variables from: {env_path}")
                env_loaded = True
                break

        if not env_loaded:
            logger.info(
                "No .env file found in common locations. Relying on existing environment variables."
            )

    except ImportError:
        logger.warning(
            "python-dotenv not installed. Relying on existing environment variables."
        )

    endpoint = os.getenv("PROMPT_FLOW_ENDPOINT")
    api_key = os.getenv("PROMPT_FLOW_API_KEY")
    flow_name = os.getenv("PROMPT_FLOW_FLOW_NAME")

    if not all([endpoint, api_key, flow_name]):
        logger.error(
            "Missing one or more required environment variables: "
            "PROMPT_FLOW_ENDPOINT, PROMPT_FLOW_API_KEY, PROMPT_FLOW_FLOW_NAME"
        )
        logger.error("Please ensure these are set in your .env file or environment.")
        return

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # For key-based auth, the endpoint might already be the full scoring URL
    # or it might need /flows/{flow_name}:run appended.
    # The PromptFlowClient has logic for this, here we make a common assumption.
    if (
        "/score" in endpoint.lower()
    ):  # Check if it's a managed online endpoint scoring URI
        url = endpoint
        logger.info(f"Using direct scoring URL: {url}")
    else:  # Assume it's a base AML endpoint, append flow name
        url = f"{endpoint}/flows/{flow_name}:run"
        logger.info(f"Constructed run URL: {url}")

    # Check for environment variables that specify field names
    request_field_name = os.getenv("PROMPTFLOW_REQUEST_FIELD_NAME", "question")

    payload = {
        "inputs": {
            request_field_name: "Hello Prompt Flow! This is a test message.",
            "chat_history": [],
        }
    }

    logger.info(f"Attempting to call Prompt Flow...")
    logger.info(f"Endpoint: {url}")
    logger.info(
        f"API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if api_key else 'Not Set'}"
    )  # Mask API key
    logger.info(f"Flow Name: {flow_name}")
    logger.info(f"Using request field name: {request_field_name}")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url, json=payload, headers=headers, timeout=60
            ) as resp:
                logger.info(f"Response Status: {resp.status}")
                response_text = await resp.text()
                try:
                    response_json = json.loads(response_text)
                    logger.info(
                        f"Response JSON:\n{json.dumps(response_json, indent=2)}"
                    )
                except json.JSONDecodeError:
                    logger.info(f"Response Text (not JSON):\n{response_text}")

                if resp.status == 200:
                    logger.info("Prompt Flow test successful!")
                else:
                    logger.error(f"Prompt Flow test failed with status {resp.status}.")

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection Error: Failed to connect to {url}. Details: {e}")
        except aiohttp.ClientTimeout as e:
            logger.error(f"Timeout Error: Request to {url} timed out. Details: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(check_prompt_flow())
