#!/usr/bin/env python3
"""
Minimal command-line interface to chat with the Prompt Flow endpoint.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

CHAT_LOG_FILE = project_root / "scripts" / "cli_chat_log.json"

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
    print(
        "python-dotenv not found, will rely on environment variables already set.",
        file=sys.stderr,
    )

try:
    from src.backend.prompt_flow_client import PromptFlowClient
except ImportError as e:
    print(f"Error importing PromptFlowClient: {e}", file=sys.stderr)
    print(
        "Please ensure you run this script from the project root, or that src/backend is in your PYTHONPATH.",
        file=sys.stderr,
    )
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="[%X]",
)
logger = logging.getLogger("cli_chat")


def load_env_variables():
    """Loads environment variables from .env files if python-dotenv is available."""
    if load_dotenv:
        env_loaded = False
        potential_env_paths = [
            project_root / ".env",
            project_root / ".azure" / "my-multimodal-env" / ".env",
            project_root / "src" / "backend" / ".env",
        ]
        for env_path in potential_env_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=True)
                logger.info(f"Loaded environment variables from: {env_path}")
                env_loaded = True
                break
        if not env_loaded:
            logger.info(
                "No .env file found in common locations. Relying on existing environment variables."
            )
    else:
        logger.info(
            "python-dotenv not installed. Relying on existing environment variables."
        )


def extract_answer_and_citations(response_json: dict, client: PromptFlowClient):
    """
    Extracts the main answer and citations from the Prompt Flow response.
    Uses field names configured in the PromptFlowClient.
    """
    answer = None
    citations_text = None

    response_field = client.response_field_name
    citations_field = client.citations_field_name

    if response_json:
        if response_field in response_json:
            answer = response_json[response_field]
        elif (
            "output" in response_json
            and isinstance(response_json["output"], dict)
            and response_field in response_json["output"]
        ):
            answer = response_json["output"][response_field]
        elif "chat_output" in response_json:  # Common alternative structure
            chat_output_data = response_json["chat_output"]
            if isinstance(chat_output_data, str):
                answer = chat_output_data
            elif (
                isinstance(chat_output_data, dict)
                and response_field in chat_output_data
            ):
                answer = chat_output_data[response_field]

        output_obj = response_json.get("output", response_json)
        if isinstance(output_obj, dict) and citations_field in output_obj:
            citations_data = output_obj[citations_field]
            citations_text = (
                json.dumps(citations_data, indent=2) if citations_data else None
            )

    if answer is None:
        answer = "Could not extract a definitive answer. See raw JSON above."

    return answer, citations_text


def save_chat_history(history: list):
    """Saves the chat history to a JSON file."""
    try:
        with open(CHAT_LOG_FILE, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Chat history saved to {CHAT_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")


async def chat_cli():
    """Main asynchronous chat loop."""
    load_env_variables()

    client = PromptFlowClient()

    if not client.is_enabled():
        logger.error(
            "Prompt Flow is not enabled. Check PROMPT_FLOW_ENDPOINT and USE_PROMPT_FLOW env vars."
        )
        return

    logger.info(
        f"Chatting with Prompt Flow: {client.endpoint} (Flow: {client.flow_name})"
    )
    logger.info(
        f"Request field: '{client.request_field_name}', Response field: '{client.response_field_name}', Citations field: '{client.citations_field_name}'"
    )
    logger.info("Type 'exit' or 'quit' to end the chat.")

    chat_history = []

    try:
        while True:
            try:
                user_query = input("You: ")
                if user_query.lower() in ["exit", "quit"]:
                    logger.info("Exiting chat normally.")
                    break
                if not user_query.strip():
                    continue

                logger.info(f"Sending query: {user_query}")
                response_json = await client.call_endpoint(
                    query=user_query, history=chat_history
                )

                logger.info(
                    f"Raw Response JSON:\n{json.dumps(response_json, indent=2)}"
                )

                answer, citations = extract_answer_and_citations(response_json, client)

                print(f"AI: {answer}")
                if citations:
                    print(f"Citations:\n{citations}")

                if answer and not response_json.get("error"):
                    chat_history.append(
                        {
                            "inputs": {client.request_field_name: user_query},
                            "outputs": {client.response_field_name: answer},
                        }
                    )
                    if len(chat_history) > 10:
                        chat_history = chat_history[-10:]

            except KeyboardInterrupt:
                logger.info("Exiting chat due to interrupt.")
                break
            except Exception as e:
                logger.error(f"An error occurred during chat loop: {e}", exc_info=True)
                # Decide if you want to break or continue on specific errors
    finally:
        if chat_history:
            save_chat_history(chat_history)
        else:
            logger.info("No chat history to save.")


if __name__ == "__main__":
    try:
        asyncio.run(chat_cli())
    except Exception as e:
        logger.critical(f"CLI tool failed to run: {e}", exc_info=True)
