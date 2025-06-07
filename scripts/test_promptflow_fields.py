#!/usr/bin/env python3
"""
Script to test the PromptFlow field name customization and verify it's working correctly.
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


async def test_promptflow_fields(
    request_field="chat_input", response_field="output", citations_field="citations"
):
    """Tests whether the PromptFlow field name environment variables are properly applied."""
    try:
        from dotenv import load_dotenv

        # Try to load from several potential locations
        env_loaded = False
        project_root = Path(__file__).resolve().parent.parent

        potential_env_paths = [
            project_root / ".env",
            project_root / ".azure" / "my-multimodal-env" / ".env",
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

    # Check if required environment variables are set
    endpoint = os.getenv("PROMPT_FLOW_ENDPOINT")
    api_key = os.getenv("PROMPT_FLOW_API_KEY")
    flow_name = os.getenv("PROMPT_FLOW_FLOW_NAME")

    # Override environment variables with passed parameters
    os.environ["PROMPTFLOW_REQUEST_FIELD_NAME"] = request_field
    os.environ["PROMPTFLOW_RESPONSE_FIELD_NAME"] = response_field
    os.environ["PROMPTFLOW_CITATIONS_FIELD_NAME"] = citations_field

    # Get the current field name settings
    request_field = os.getenv("PROMPTFLOW_REQUEST_FIELD_NAME", "question")
    response_field = os.getenv("PROMPTFLOW_RESPONSE_FIELD_NAME", "output")
    citations_field = os.getenv("PROMPTFLOW_CITATIONS_FIELD_NAME", "citations")

    logger.info(f"Using request field name: {request_field}")
    logger.info(f"Using response field name: {response_field}")
    logger.info(f"Using citations field name: {citations_field}")

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
    if (
        "/score" in endpoint.lower()
    ):  # Check if it's a managed online endpoint scoring URI
        url = endpoint
        logger.info(f"Using direct scoring URL: {url}")
    else:  # Assume it's a base AML endpoint, append flow name
        url = f"{endpoint}/flows/{flow_name}:run"
        logger.info(f"Constructed run URL: {url}")

    # Create a payload using the custom field name
    payload = {
        "inputs": {
            request_field: "What is creativity according to the course materials?",
            "chat_history": [],
        }
    }

    logger.info(f"Attempting to call Prompt Flow...")
    logger.info(f"Endpoint: {url}")
    logger.info(
        f"API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if api_key else 'Not Set'}"
    )
    logger.info(f"Flow Name: {flow_name}")
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

                    # Check if response is in the top level
                    if response_field in response_json:
                        logger.info(
                            f"✅ SUCCESS: Found response directly in '{response_field}' field at top level"
                        )
                        answer = response_json[response_field]
                    # Check for id in the response (common in direct API responses)
                    elif "id" in response_json:
                        logger.info(
                            "Response has ID at top level - checking all fields"
                        )
                        # Try common response field names
                        for field in [
                            "output",
                            "answer",
                            "response",
                            "text",
                            "content",
                            "chat_output",
                        ]:
                            if field in response_json:
                                logger.info(
                                    f"✅ Found response in '{field}' field at top level"
                                )
                                answer = response_json[field]
                                # If this is the field that worked, recommend using it
                                if field != response_field:
                                    logger.info(
                                        f"⚠️ RECOMMENDATION: Update PROMPTFLOW_RESPONSE_FIELD_NAME to '{field}'"
                                    )
                                break
                    # Check for nested output structure (most common format)
                    elif "output" in response_json and isinstance(
                        response_json["output"], dict
                    ):
                        output_obj = response_json["output"]
                        logger.info(
                            f"Found nested output structure with keys: {list(output_obj.keys())}"
                        )

                        if response_field in output_obj:
                            logger.info(
                                f"✅ SUCCESS: Found response in 'output.{response_field}' field"
                            )
                            answer = output_obj[response_field]
                        else:
                            # Try common field names within the output object
                            for field in [
                                "output",
                                "answer",
                                "response",
                                "text",
                                "content",
                            ]:
                                if field in output_obj:
                                    logger.info(
                                        f"✅ Found response in 'output.{field}' field"
                                    )
                                    answer = output_obj[field]
                                    # If this is the field that worked, recommend using it
                                    if field != response_field:
                                        logger.info(
                                            f"⚠️ RECOMMENDATION: Update PROMPTFLOW_RESPONSE_FIELD_NAME to '{field}'"
                                        )
                                    break
                            else:
                                logger.error(
                                    f"❌ ERROR: Could not find response in 'output.{response_field}' field or common alternatives"
                                )
                                logger.info(
                                    f"Available fields in output object: {list(output_obj.keys())}"
                                )

                        # Check for citations
                        if citations_field in output_obj:
                            logger.info(
                                f"✅ SUCCESS: Found citations in 'output.{citations_field}' field"
                            )
                        else:
                            # Try common citation field names
                            for field in [
                                "citations",
                                "context",
                                "sources",
                                "references",
                            ]:
                                if field in output_obj:
                                    logger.info(
                                        f"✅ Found citations in 'output.{field}' field"
                                    )
                                    # If this is the field that worked, recommend using it
                                    if field != citations_field:
                                        logger.info(
                                            f"⚠️ RECOMMENDATION: Update PROMPTFLOW_CITATIONS_FIELD_NAME to '{field}'"
                                        )
                                    break
                            else:
                                logger.warning(
                                    f"⚠️ WARNING: Could not find citations in 'output.{citations_field}' field or common alternatives"
                                )
                    # Check for chat_output structure (another common format)
                    elif "chat_output" in response_json:
                        chat_output = response_json["chat_output"]
                        logger.info("Found chat_output structure")

                        if isinstance(chat_output, str):
                            logger.info(
                                "✅ SUCCESS: Found response in chat_output as string"
                            )
                            answer = chat_output
                        elif isinstance(chat_output, dict):
                            logger.info(
                                f"chat_output is a dict with keys: {list(chat_output.keys())}"
                            )
                            if response_field in chat_output:
                                logger.info(
                                    f"✅ SUCCESS: Found response in 'chat_output.{response_field}' field"
                                )
                                answer = chat_output[response_field]
                            else:
                                # Try common field names
                                for field in [
                                    "output",
                                    "answer",
                                    "response",
                                    "text",
                                    "content",
                                ]:
                                    if field in chat_output:
                                        logger.info(
                                            f"✅ Found response in 'chat_output.{field}' field"
                                        )
                                        # If this is the field that worked, recommend using it
                                        if field != response_field:
                                            logger.info(
                                                f"⚠️ RECOMMENDATION: Update PROMPTFLOW_RESPONSE_FIELD_NAME to '{field}'"
                                            )
                                        break
                    else:
                        logger.warning(
                            "❌ ERROR: Response structure does not match expected format"
                        )
                        logger.info(
                            f"Top-level keys in response: {list(response_json.keys())}"
                        )

                except json.JSONDecodeError:
                    logger.info(f"Response Text (not JSON):\n{response_text}")

                if resp.status == 200:
                    logger.info("Prompt Flow test completed!")
                else:
                    logger.error(f"Prompt Flow test failed with status {resp.status}.")

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection Error: Failed to connect to {url}. Details: {e}")
        except aiohttp.ClientTimeout as e:
            logger.error(f"Timeout Error: Request to {url} timed out. Details: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)


async def test_multiple_configurations():
    """Test multiple field name configurations to find the correct one."""
    configurations = [
        # Test the most common configurations
        {
            "request_field": "chat_input",
            "response_field": "output",
            "citations_field": "citations",
        },
        {
            "request_field": "question",
            "response_field": "answer",
            "citations_field": "citations",
        },
        {
            "request_field": "question",
            "response_field": "output",
            "citations_field": "citations",
        },
        {
            "request_field": "chat_input",
            "response_field": "answer",
            "citations_field": "citations",
        },
        # Add more configurations if needed
    ]

    for i, config in enumerate(configurations):
        logger.info(f"\n\n{'='*80}\nTEST CONFIGURATION #{i+1}\n{'='*80}")
        await test_promptflow_fields(**config)
        # Wait a bit between tests
        await asyncio.sleep(1)


if __name__ == "__main__":
    # If command-line arguments are provided, use them to override the defaults
    import sys

    if len(sys.argv) > 1:
        # Run single test with specified fields
        if len(sys.argv) >= 4:
            asyncio.run(
                test_promptflow_fields(
                    request_field=sys.argv[1],
                    response_field=sys.argv[2],
                    citations_field=sys.argv[3],
                )
            )
        else:
            print(
                "Usage: python test_promptflow_fields.py [request_field response_field citations_field]"
            )
            print("Or run without args to test multiple configurations")
    else:
        # Run multiple test configurations
        asyncio.run(test_multiple_configurations())
