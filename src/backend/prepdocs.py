from email.policy import default
import os
import glob
from typing import Optional
import instructor
import aiofiles
import asyncio
import sys
from dotenv import load_dotenv

from azure.core.pipeline.policies import UserAgentPolicy
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.inference.aio import EmbeddingsClient, ImageEmbeddingsClient
from openai import AsyncAzureOpenAI, api_version
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient, SearchIndexerClient
from azure.storage.blob.aio import BlobServiceClient
from data_injestion.models import ProcessRequest
from data_injestion.indexer_img_verbalize_strategy import (
    IndexerImgVerbalizationStrategy,
)
from data_injestion.strategy import Strategy
from constants import USER_AGENT
from processfile import ProcessFile
from azure.identity.aio import DefaultAzureCredential
import argparse

# Load environment variables from .env file
load_dotenv()


def load_environment_variables():
    """Loads environment variables from the .env file."""
    required_vars = [
        "DOCUMENTINTELLIGENCE_ENDPOINT",
        "AZURE_INFERENCE_EMBED_ENDPOINT",
        "SEARCH_SERVICE_ENDPOINT",
        "SEARCH_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "ARTIFACTS_STORAGE_ACCOUNT_URL",
    ]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing environment variable: {var}")


def setup_directories():
    """Sets up necessary directories for document and image processing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    documents_to_process_folder = os.path.abspath(
        os.path.join(script_dir, "../../data")
    )
    documents_output_folder = os.path.abspath(os.path.join(script_dir, "./static"))
    os.makedirs(documents_output_folder, exist_ok=True)

    return documents_to_process_folder, documents_output_folder


def get_blob_storage_credentials():
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    blob_container_name = os.getenv("BLOB_CONTAINER_NAME")
    sas_token = os.getenv("BLOB_SAS_TOKEN")
    if not blob_container_name or not sas_token:
        raise ValueError(
            "Blob container name and SAS token must be provided for blob storage source."
        )
    return storage_account_name, blob_container_name, sas_token


async def main(
    source: str, indexer_Strategy: Optional[str] = None, data_path: Optional[str] = None
):
    load_environment_variables()
    documents_to_process_folder, documents_output_folder = setup_directories()

    # If data_path is specified, use it instead of the default documents folder
    if data_path:
        # Handle both absolute and relative paths
        if os.path.isabs(data_path):
            target_path = data_path
        else:
            # Relative to the data directory
            target_path = os.path.join(documents_to_process_folder, data_path)

        # Validate the path exists
        if not os.path.exists(target_path):
            raise ValueError(f"Specified data path does not exist: {target_path}")

        documents_to_process_folder = target_path
        print(f"Processing documents from: {documents_to_process_folder}")

    tokenCredential = DefaultAzureCredential()

    document_client = DocumentIntelligenceClient(
        endpoint=os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"],
        credential=tokenCredential,
    )

    text_embedding_client = EmbeddingsClient(
        endpoint=os.environ["AZURE_INFERENCE_EMBED_ENDPOINT"],
        credential=tokenCredential,
        model=os.environ["AZURE_INFERENCE_EMBED_MODEL_NAME"],
    )

    image_embedding_client = ImageEmbeddingsClient(
        endpoint=os.environ["AZURE_INFERENCE_EMBED_ENDPOINT"],
        credential=tokenCredential,
        model=os.environ["AZURE_INFERENCE_EMBED_MODEL_NAME"],
    )

    search_client = SearchClient(
        index_name=os.environ["SEARCH_INDEX_NAME"],
        endpoint=os.environ["SEARCH_SERVICE_ENDPOINT"],
        credential=tokenCredential,
        base_user_agent=USER_AGENT,
    )

    index_client = SearchIndexClient(
        index_name=os.environ["SEARCH_INDEX_NAME"],
        endpoint=os.environ["SEARCH_SERVICE_ENDPOINT"],
        credential=tokenCredential,
        user_agent_policy=UserAgentPolicy(base_user_agent=USER_AGENT),
    )

    indexer_Client = SearchIndexerClient(
        endpoint=os.environ["SEARCH_SERVICE_ENDPOINT"],
        credential=tokenCredential,
        user_agent_policy=UserAgentPolicy(base_user_agent=USER_AGENT),
    )

    instructor_openai_client = instructor.from_openai(
        AsyncAzureOpenAI(
            azure_ad_token=(
                await tokenCredential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )
            ).token,
            api_version="2024-08-01-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
    )

    blob_service_client = BlobServiceClient(
        account_url=os.environ["ARTIFACTS_STORAGE_ACCOUNT_URL"],
        credential=tokenCredential,
    )

    strategy: Strategy | None = None
    request: Optional[ProcessRequest] = None
    if indexer_Strategy == "indexer-image-verbal":
        strategy = IndexerImgVerbalizationStrategy()
        request = ProcessRequest(
            blobServiceClient=blob_service_client,
            blobSource=os.environ["SAMPLES_STORAGE_CONTAINER"],
            indexClient=index_client,
            indexName=os.environ["SEARCH_INDEX_NAME"],
            knowledgeStoreContainer=os.environ["ARTIFACTS_STORAGE_CONTAINER"],
            localDataSource=documents_to_process_folder,
            indexerClient=indexer_Client,
            chatCompletionEndpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            chatCompletionModel=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            chatCompletionDeployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            aoaiEmbeddingEndpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            aoaiEmbeddingDeployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            aoaiEmbeddingModel=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            cognitiveServicesEndpoint=os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"],
            subscriptionId=os.environ["AZURE_SUBSCRIPTION_ID"],
            resourceGroup=os.environ["AZURE_RESOURCE_GROUP"],
        )
        await strategy.run(request) if request is not None else None
    elif indexer_Strategy == "self-multimodal-embedding":
        process_file = ProcessFile(
            document_client,
            text_embedding_client,
            image_embedding_client,
            search_client,
            index_client,
            instructor_openai_client,
            blob_service_client,
            os.environ["AZURE_OPENAI_DEPLOYMENT"],
        )

        if source == "files":
            await process_files(
                process_file, documents_to_process_folder, documents_output_folder
            )
        elif source == "blobs":
            await process_blobs(process_file, *get_blob_storage_credentials())
        else:
            raise ValueError("Invalid source. Must be 'files' or 'blobs'.")
    else:
        raise ValueError("Invalid indexer strategy. Check readme for available.")

    print("Done")
    await document_client.close()
    await text_embedding_client.close()
    await image_embedding_client.close()
    await blob_service_client.close()
    await search_client.close()
    await index_client.close()
    await indexer_Client.close()
    await instructor_openai_client.close()
    await tokenCredential.close()


async def process_files(
    process_file, documents_to_process_folder, documents_output_folder
):
    """
    Process files from the specified folder or file path.
    Supports both individual files and directories (including subdirectories).
    """
    document_paths = []

    if os.path.isfile(documents_to_process_folder):
        # Single file specified
        if documents_to_process_folder.lower().endswith(
            (".pdf", ".docx", ".doc", ".txt")
        ):
            document_paths = [documents_to_process_folder]
            print(f"Processing single file: {documents_to_process_folder}")
        else:
            print(f"Skipping unsupported file type: {documents_to_process_folder}")
            return
    elif os.path.isdir(documents_to_process_folder):
        # Directory specified - search for all supported files recursively
        supported_extensions = ["*.pdf", "*.docx", "*.doc", "*.txt"]
        for extension in supported_extensions:
            # Use recursive glob to find files in subdirectories
            pattern = os.path.join(documents_to_process_folder, "**", extension)
            document_paths.extend(glob.glob(pattern, recursive=True))

        # Also check for files directly in the specified directory
        for extension in supported_extensions:
            pattern = os.path.join(documents_to_process_folder, extension)
            document_paths.extend(glob.glob(pattern))

        # Remove duplicates and sort
        document_paths = sorted(list(set(document_paths)))
        print(
            f"Found {len(document_paths)} documents to process in: {documents_to_process_folder}"
        )
    else:
        raise ValueError(f"Invalid path: {documents_to_process_folder}")

    if not document_paths:
        print("No supported documents found to process.")
        return

    # Display files that will be processed
    print("Documents to be processed:")
    for i, doc_path in enumerate(document_paths, 1):
        relative_path = os.path.relpath(doc_path, documents_to_process_folder)
        print(f"  {i}. {relative_path}")

    # Process each document
    for doc_path in document_paths:
        print(f"\nProcessing file: {doc_path}")
        try:
            async with aiofiles.open(doc_path, "rb") as f:
                file_bytes = await f.read()
                await process_file.process_file(
                    file_bytes,
                    os.path.basename(doc_path),
                    os.environ["SEARCH_INDEX_NAME"],
                )

                # Copy the document to documents_output_folder
                destination_path = os.path.join(
                    documents_output_folder, os.path.basename(doc_path)
                )
                with open(doc_path, "rb") as src_file:
                    with open(destination_path, "wb") as dest_file:
                        dest_file.write(src_file.read())

                print(f"✅ Successfully processed: {os.path.basename(doc_path)}")
        except Exception as e:
            print(f"❌ Error processing {doc_path}: {str(e)}")
            continue


async def process_blobs(
    process_file, storage_account_name: str, blob_container_name: str, sas_token: str
):
    print(f"storage_account_name: {storage_account_name}")
    print(f"blob_container_name: {blob_container_name}")
    print(f"sas_token: {sas_token}")

    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=sas_token,
    )
    container_client = blob_service_client.get_container_client(blob_container_name)

    blobs = container_client.list_blobs(include=["metadata"])

    count = 0
    for blob in blobs:
        print(f"Processing blob: {blob.name}")
        blob_client = container_client.get_blob_client(blob)

        stream = blob_client.download_blob()
        data = stream.readall()
        count += 1
        await process_file.process_file(
            data,
            blob.name,
            os.environ["SEARCH_INDEX_NAME"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process documents or blobs using indexers."
    )
    parser.add_argument(
        "--source",
        choices=["files", "blobs"],
        help="Specify the source of documents: 'files' for local files or 'blobs' for blobs.",
    )
    parser.add_argument(
        "--indexer_strategy",
        choices=["indexer-image-verbal", "self-multimodal-embedding"],
    )
    parser.add_argument(
        "--data_path",
        help="Specify a path to a specific folder or file within the data directory.",
    )
    args = parser.parse_args()

    asyncio.run(main(args.source, args.indexer_strategy, args.data_path))
