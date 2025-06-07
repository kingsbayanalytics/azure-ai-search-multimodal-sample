import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from aiohttp import web
from rich.logging import RichHandler
from openai import AsyncAzureOpenAI
from azure.identity.aio import (
    DefaultAzureCredential,
    AzureCliCredential,
    get_bearer_token_provider,
)
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient
from azure.core.pipeline.policies import UserAgentPolicy

from azure.storage.blob.aio import BlobServiceClient

from search_grounding import SearchGroundingRetriever
from knowledge_agent import KnowledgeAgentGrounding
from constants import USER_AGENT
from multimodalrag import MultimodalRag
from data_model import DocumentPerChunkDataModel
from citation_file_handler import CitationFilesHandler
from prompt_flow_client import PromptFlowClient

# Load environment variables from .env file
load_dotenv()
# Try to load from .azure directory if exists
if os.path.exists(".azure/my-multimodal-env/.env"):
    load_dotenv(".azure/my-multimodal-env/.env")

# Set PromptFlow field names directly
os.environ["PROMPTFLOW_REQUEST_FIELD_NAME"] = "question"
os.environ["PROMPTFLOW_RESPONSE_FIELD_NAME"] = "output"
os.environ["PROMPTFLOW_CITATIONS_FIELD_NAME"] = "citations"

logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more verbose logging
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Make sure all loggers are set to debug level
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.DEBUG)

# Get logger for app
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)


async def list_indexes(index_client: SearchIndexClient):
    indexes = []
    async for index in index_client.list_indexes():
        indexes.append({"name": index.name})
    return web.json_response([index["name"] for index in indexes])


async def create_app():
    # Credential for Azure AD authenticated services (OpenAI, Storage, etc.)
    azure_ad_credential = AzureCliCredential()

    # Configure Azure OpenAI
    openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    # Look for model and deployment names with fallbacks
    openai_deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    chatcompletions_model_name = os.environ.get(
        "AZURE_OPENAI_MODEL_NAME", openai_deployment_name
    )

    # Log OpenAI configuration
    logging.info(f"Using OpenAI endpoint: {openai_endpoint}")
    logging.info(f"Using OpenAI deployment: {openai_deployment_name}")
    logging.info(f"Using OpenAI model: {chatcompletions_model_name}")

    # Support token-based auth for OpenAI
    openai_token_provider = get_bearer_token_provider(
        azure_ad_credential, "https://cognitiveservices.azure.com/.default"
    )

    # Determine authentication method for AI Search
    use_key_auth = (
        "SEARCH_SERVICE_KEY" in os.environ and os.environ["SEARCH_SERVICE_KEY"]
    )

    # Configure Azure AI Search credentials
    if use_key_auth:
        # Use API key authentication
        search_credential = AzureKeyCredential(os.environ["SEARCH_SERVICE_KEY"])
    else:
        # Use Azure AD authentication
        search_credential = azure_ad_credential

    # Get Search service endpoint and index name
    search_endpoint = os.environ.get("SEARCH_SERVICE_ENDPOINT")
    search_index_name = os.environ.get(
        "SEARCH_INDEX_NAME", "adaption-innovation-course-mats-index"
    )

    # Log the actual search index being used
    logging.info(f"Using search endpoint: {search_endpoint}")
    logging.info(f"Using search index: {search_index_name}")

    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=search_index_name,
        credential=search_credential,
        user_agent=USER_AGENT,
    )

    index_client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=search_credential,
        user_agent=USER_AGENT,
    )

    data_model = DocumentPerChunkDataModel()

    # Azure AI Search Knowledge Agent configuration
    knowledge_agent_name = os.environ.get("KNOWLEDGE_AGENT_NAME", "default-agent")
    search_service_name = search_endpoint.split("https://")[1].split(".")[0]

    # Setup Knowledge Agent Client
    ka_retrieval_client = KnowledgeAgentRetrievalClient(
        endpoint=search_endpoint,
        credential=search_credential,
        agent_name=knowledge_agent_name,
    )

    knowledge_agent = KnowledgeAgentGrounding(
        ka_retrieval_client,
        search_client,
        index_client,
        data_model,
        search_index_name,
        knowledge_agent_name,
        openai_endpoint,
        openai_deployment_name,
        chatcompletions_model_name,
    )

    openai_client = AsyncAzureOpenAI(
        azure_ad_token_provider=openai_token_provider,
        api_version="2024-08-01-preview",
        azure_endpoint=openai_endpoint,
        timeout=30,
    )

    search_grounding = SearchGroundingRetriever(
        search_client,
        openai_client,
        data_model,
        chatcompletions_model_name,
    )

    # Configure storage - using defaults if not found
    blob_service_client = BlobServiceClient(
        account_url=os.environ.get(
            "ARTIFACTS_STORAGE_ACCOUNT_URL",
            "https://storageaccount.blob.core.windows.net",
        ),
        credential=azure_ad_credential,
    )
    artifacts_container_client = blob_service_client.get_container_client(
        os.environ.get("ARTIFACTS_STORAGE_CONTAINER", "documents")
    )

    prompt_flow_client = PromptFlowClient()

    app = web.Application(middlewares=[])

    mmrag = MultimodalRag(
        knowledge_agent,
        search_grounding,
        openai_client,
        chatcompletions_model_name,
        artifacts_container_client,
        prompt_flow_client,
    )
    mmrag.attach_to_app(app, "/chat")

    citation_files_handler = CitationFilesHandler(
        blob_service_client, artifacts_container_client
    )

    current_directory = Path(__file__).parent
    app.add_routes(
        [
            web.get(
                "/", lambda _: web.FileResponse(current_directory / "static/index.html")
            ),
            web.get("/list_indexes", lambda req: list_indexes(index_client)),
            web.post("/get_citation_doc", citation_files_handler.handle),
        ]
    )
    app.router.add_static(
        "/assets", path=current_directory / "static/assets", name="assets"
    )
    app.router.add_static("/", path=current_directory / "static", name="static")

    return app


# Test token acquisition (can be removed or adapted if not needed for key-based auth)
# async def test_token_acquisition():
#     print(
#         "\n\n********************************************************************************"
#     )
#     print("--- STARTING DIRECT TOKEN ACQUISITION TEST FOR AZURE SEARCH ---")
#     print(
#         "********************************************************************************\n"
#     )
#     sys.stdout.flush()
#     logging.info(
#         "--- Attempting direct token acquisition for Azure Search (logging) ---"
#     )
#     try:
#         credential = AzureCliCredential()
#         token = await credential.get_token("https://search.azure.com/.default")
#         print(
#             f"--- Successfully retrieved Azure Search token. Token: {token.token[:20]}... (truncated), Expires on: {token.expires_on} ---"
#         )
#         logging.info(
#             f"--- Successfully retrieved Azure Search token. Expires on: {token.expires_on} (logging) ---"
#         )
#     except Exception as e:
#         print(f"--- FAILED to retrieve Azure Search token directly: {e} ---")
#         logging.error(
#             f"--- Failed to retrieve Azure Search token directly: {e} (logging) ---",
#             exc_info=True,
#         )
#         import traceback
#
#         traceback.print_exc(file=sys.stdout)
#     finally:
#         print(
#             "\n********************************************************************************"
#         )
#         print("--- DIRECT TOKEN ACQUISITION TEST FINISHED ---")
#         print(
#             "********************************************************************************\n"
#         )
#         sys.stdout.flush()
#         sys.exit(1)


if __name__ == "__main__":
    # import asyncio
    # asyncio.run(test_token_acquisition()) # Ensure this is commented out

    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"======== Running on http://{host}:{port} ========")
    web.run_app(create_app(), host=host, port=port)
