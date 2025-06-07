from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from models import (
    SearchRequestParameters,
    SearchConfig,
    GroundingResult,
    GroundingResults,
)

logger = logging.getLogger("data_model")


class DataModel(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def create_search_payload(
        self, query: str, search_config: SearchConfig
    ) -> SearchRequestParameters:
        """Creates the search request payload."""
        pass

    @abstractmethod
    def extract_citation(
        self,
        document: dict,
    ) -> dict:
        """Extracts citations from search results."""
        pass

    @abstractmethod
    async def collect_grounding_results(self, search_results: List[dict]) -> list:
        """Collects and formats documents from search results."""
        pass


class DocumentPerChunkDataModel(DataModel):
    def create_search_payload(
        self, query: str, search_config: SearchConfig
    ) -> SearchRequestParameters:
        """Creates the search request payload with vector/semantic/hybrid configurations using a configured vectorizer."""

        # Determine fields to select based on index schema
        # The standard fields expected by the application
        standard_fields = [
            "content_id",
            "content_text",
            "document_title",
            "text_document_id",
            "image_document_id",
            "locationMetadata",
            "content_path",
        ]

        # For backward compatibility, also include common field names that might exist in other indexes
        alternative_fields = {
            "content_id": ["id", "chunk_id", "document_id"],
            "content_text": ["content", "text", "chunk_content"],
            "document_title": ["title", "name", "file_name"],
            "text_document_id": ["source_id", "parent_id"],
            "locationMetadata": ["metadata", "location", "page_info"],
        }

        # Construct the select string, starting with standard fields
        select_fields = standard_fields.copy()

        # Create the payload
        payload = {
            "search": query,
            "top": 5,
            "vector_queries": [
                {
                    "text": query,
                    "fields": "content_embedding",
                    "kind": "text",
                    "k": 5,
                }
            ],
            "select": ",".join(select_fields),
        }

        logger.info(f"Created search payload with select: {payload['select']}")
        return payload

    def extract_citation(self, document):
        """
        Extract citation data from a document, handling different schema structures.
        """
        # Get content_id (required field, use alternatives if needed)
        content_id = self._get_field_value(
            document, "content_id", ["id", "chunk_id", "document_id"]
        )

        # Get text field (required)
        text = self._get_field_value(
            document, "content_text", ["content", "text", "chunk_content"]
        )

        # Get title field
        title = self._get_field_value(
            document, "document_title", ["title", "name", "file_name"]
        )

        # Get location metadata (if available)
        location_metadata = self._get_field_value(
            document, "locationMetadata", ["metadata", "location", "page_info"]
        )

        # Get document ID (text or image)
        text_doc_id = self._get_field_value(
            document, "text_document_id", ["source_id", "parent_id"]
        )
        image_doc_id = self._get_field_value(document, "image_document_id", [])

        doc_id = text_doc_id if text_doc_id is not None else image_doc_id

        return {
            "locationMetadata": location_metadata,
            "text": text,
            "title": title,
            "content_id": content_id,
            "docId": doc_id,
        }

    async def collect_grounding_results(
        self, search_results: List[dict]
    ) -> List[GroundingResult]:
        collected_documents = []
        for result in search_results:
            # Extract content_id (required field)
            content_id = self._get_field_value(
                result, "content_id", ["id", "chunk_id", "document_id"]
            )

            if not content_id:
                logger.warning(
                    f"Missing content_id in search result, generating placeholder: {result}"
                )
                content_id = f"result_{len(collected_documents)}"

            # Determine if text or image
            is_image = result.get("image_document_id") is not None
            is_text = result.get("text_document_id") is not None

            # If neither is explicitly set, look at content type
            if not is_image and not is_text:
                content_path = self._get_field_value(
                    result, "content_path", ["path", "file_path"]
                )
                if content_path and any(
                    ext in content_path.lower()
                    for ext in [".jpg", ".jpeg", ".png", ".gif"]
                ):
                    is_image = True
                else:
                    is_text = True  # Default to text if can't determine

            # Get content text
            content_text = self._get_field_value(
                result, "content_text", ["content", "text", "chunk_content"]
            )

            # Get content path for images
            content_path = self._get_field_value(
                result, "content_path", ["path", "file_path", "url"]
            )

            if is_text and content_text:
                collected_documents.append(
                    {
                        "ref_id": content_id,
                        "content": {
                            "ref_id": content_id,
                            "text": content_text,
                        },
                        "content_type": "text",
                        **result,
                    }
                )
            elif is_image and content_path:
                collected_documents.append(
                    {
                        "ref_id": content_id,
                        "content": content_path,
                        "content_type": "image",
                        **result,
                    }
                )
            else:
                logger.warning(f"Skipping result with missing content: {result}")

        return collected_documents

    def _get_field_value(
        self,
        document: Dict[str, Any],
        primary_field: str,
        alternative_fields: List[str] = None,
    ):
        """
        Helper method to get a field value from a document, trying alternative field names if needed.

        Args:
            document: The document to extract the field from
            primary_field: The primary field name to try
            alternative_fields: List of alternative field names to try if primary field is not found

        Returns:
            The field value or None if not found
        """
        # Try primary field first
        if primary_field in document:
            return document[primary_field]

        # Try alternative fields
        if alternative_fields:
            for field in alternative_fields:
                if field in document:
                    return document[field]

        # Not found
        return None
