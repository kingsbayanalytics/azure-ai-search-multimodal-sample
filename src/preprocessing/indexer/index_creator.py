"""
Azure Search index creator for cost-effective document preprocessing.

This module creates Azure Search indexes that are fully compatible with the existing
multimodal RAG application while maintaining exact visual citation metadata format.
The indexes created by this module can be consumed by the app without any changes.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import structlog
from dataclasses import dataclass
from datetime import datetime
import json
import uuid

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    ComplexField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchVectorizer,
    AzureOpenAIVectorizer,
    AzureOpenAIParameters,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

from config.settings import get_settings


@dataclass
class IndexCreationResult:
    """Result of index creation operation."""

    index_name: str
    success: bool
    documents_indexed: int
    total_documents: int
    creation_time_ms: int
    index_size_mb: float
    error_message: Optional[str] = None
    warnings: List[str] = None


class IndexCreator:
    """
    Azure Search index creator for preprocessed documents.

    Creates indexes that are fully compatible with the existing multimodal RAG
    application, preserving visual citation metadata and maintaining the exact
    schema structure expected by the app.
    """

    def __init__(self):
        """Initialize the index creator with Azure Search configuration."""
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)

        # Initialize Azure Search clients
        self.credential = AzureKeyCredential(self.settings.search_service_key)
        self.index_client = SearchIndexClient(
            endpoint=self.settings.search_service_endpoint, credential=self.credential
        )

        # Schema configuration for app compatibility
        self.schema_config = self._initialize_schema_config()

    def _initialize_schema_config(self) -> Dict[str, Any]:
        """
        Initialize the exact schema configuration expected by the app.

        This schema must match exactly what the existing app expects for
        visual citations and search functionality to work properly.
        """
        return {
            # Core document fields
            "id_field": "id",
            "content_field": "content",
            "title_field": "title",
            "filename_field": "file_name",
            "url_field": "url",
            # Vector search fields
            "content_vector_field": "contentVector",
            "vector_dimension": 384,  # Default for SentenceTransformers all-MiniLM-L6-v2
            # Visual citation fields (critical for app compatibility)
            "chunks_field": "chunks",
            "location_metadata_field": "locationMetadata",
            "images_field": "images",
            # Metadata fields
            "processing_metadata_field": "processingMetadata",
            "timestamp_field": "timestamp",
        }

    async def create_index(
        self,
        processed_documents: List[Dict[str, Any]],
        index_name: str,
        vector_dimension: Optional[int] = None,
        overwrite_existing: bool = False,
    ) -> IndexCreationResult:
        """
        Create Azure Search index with processed documents.

        Args:
            processed_documents: List of processed document data
            index_name: Name for the Azure Search index
            vector_dimension: Embedding vector dimension (auto-detected if None)
            overwrite_existing: Whether to overwrite existing index

        Returns:
            IndexCreationResult with creation status and metrics
        """
        start_time = asyncio.get_event_loop().time()
        warnings = []

        self.logger.info(
            "Starting index creation",
            index_name=index_name,
            document_count=len(processed_documents),
            overwrite_existing=overwrite_existing,
        )

        try:
            # Validate documents and detect vector dimension
            if processed_documents:
                detected_dimension = self._detect_vector_dimension(processed_documents)
                if vector_dimension is None:
                    vector_dimension = detected_dimension
                elif vector_dimension != detected_dimension:
                    warnings.append(
                        f"Vector dimension mismatch: specified {vector_dimension}, "
                        f"detected {detected_dimension}"
                    )
            else:
                vector_dimension = (
                    vector_dimension or self.schema_config["vector_dimension"]
                )

            # Create or update index schema
            index_created = await self._create_index_schema(
                index_name, vector_dimension, overwrite_existing
            )

            if not index_created:
                return IndexCreationResult(
                    index_name=index_name,
                    success=False,
                    documents_indexed=0,
                    total_documents=len(processed_documents),
                    creation_time_ms=0,
                    index_size_mb=0.0,
                    error_message="Failed to create index schema",
                    warnings=warnings,
                )

            # Upload documents to index
            documents_indexed = 0
            if processed_documents:
                documents_indexed = await self._upload_documents(
                    index_name, processed_documents
                )

            # Calculate processing time and index size
            end_time = asyncio.get_event_loop().time()
            creation_time_ms = int((end_time - start_time) * 1000)
            index_size_mb = self._estimate_index_size(processed_documents)

            self.logger.info(
                "Index creation completed",
                index_name=index_name,
                documents_indexed=documents_indexed,
                total_documents=len(processed_documents),
                creation_time_ms=creation_time_ms,
                index_size_mb=index_size_mb,
            )

            return IndexCreationResult(
                index_name=index_name,
                success=True,
                documents_indexed=documents_indexed,
                total_documents=len(processed_documents),
                creation_time_ms=creation_time_ms,
                index_size_mb=index_size_mb,
                error_message=None,
                warnings=warnings,
            )

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            creation_time_ms = int((end_time - start_time) * 1000)

            self.logger.error(
                "Index creation failed",
                index_name=index_name,
                error=str(e),
                creation_time_ms=creation_time_ms,
                exc_info=True,
            )

            return IndexCreationResult(
                index_name=index_name,
                success=False,
                documents_indexed=0,
                total_documents=len(processed_documents),
                creation_time_ms=creation_time_ms,
                index_size_mb=0.0,
                error_message=str(e),
                warnings=warnings,
            )

    def _detect_vector_dimension(
        self, processed_documents: List[Dict[str, Any]]
    ) -> int:
        """Detect vector dimension from processed documents."""
        for doc in processed_documents:
            embeddings = doc.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return len(embeddings[0])

        # Fallback to default dimension
        return self.schema_config["vector_dimension"]

    async def _create_index_schema(
        self, index_name: str, vector_dimension: int, overwrite_existing: bool
    ) -> bool:
        """
        Create Azure Search index with schema compatible with the existing app.

        The schema must exactly match what the app expects for visual citations.
        """
        try:
            # Check if index already exists
            try:
                existing_index = self.index_client.get_index(index_name)
                if not overwrite_existing:
                    self.logger.info(
                        "Index already exists, skipping creation", index_name=index_name
                    )
                    return True
                else:
                    self.logger.info(
                        "Deleting existing index for recreation", index_name=index_name
                    )
                    self.index_client.delete_index(index_name)
            except ResourceNotFoundError:
                # Index doesn't exist, which is fine
                pass

            # Define fields that exactly match the app's expectations
            fields = [
                # Core document fields
                SimpleField(
                    name=self.schema_config["id_field"],
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                    sortable=True,
                ),
                SearchableField(
                    name=self.schema_config["content_field"],
                    type=SearchFieldDataType.String,
                    searchable=True,
                    retrievable=True,
                ),
                SearchableField(
                    name=self.schema_config["title_field"],
                    type=SearchFieldDataType.String,
                    searchable=True,
                    retrievable=True,
                    filterable=True,
                    sortable=True,
                ),
                SimpleField(
                    name=self.schema_config["filename_field"],
                    type=SearchFieldDataType.String,
                    retrievable=True,
                    filterable=True,
                    sortable=True,
                ),
                SimpleField(
                    name=self.schema_config["url_field"],
                    type=SearchFieldDataType.String,
                    retrievable=True,
                    filterable=True,
                ),
                # Vector search field for semantic search
                SearchField(
                    name=self.schema_config["content_vector_field"],
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=vector_dimension,
                    vector_search_profile_name="default-vector-profile",
                ),
                # Visual citation fields (critical for app compatibility)
                ComplexField(
                    name=self.schema_config["chunks_field"],
                    fields=[
                        SimpleField(
                            name="text",
                            type=SearchFieldDataType.String,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="page_number",
                            type=SearchFieldDataType.Int32,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="chunk_id",
                            type=SearchFieldDataType.String,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="bounding_box",
                            type=SearchFieldDataType.Collection(
                                SearchFieldDataType.Double
                            ),
                            retrievable=True,
                        ),
                        SimpleField(
                            name="char_start",
                            type=SearchFieldDataType.Int32,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="char_end",
                            type=SearchFieldDataType.Int32,
                            retrievable=True,
                        ),
                    ],
                    collection=True,
                ),
                # Location metadata for visual citations (exact app format)
                ComplexField(
                    name=self.schema_config["location_metadata_field"],
                    fields=[
                        SimpleField(
                            name="page_number",
                            type=SearchFieldDataType.Int32,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="bounding_polygons",
                            type=SearchFieldDataType.Collection(
                                SearchFieldDataType.Collection(
                                    SearchFieldDataType.Double
                                )
                            ),
                            retrievable=True,
                        ),
                        SimpleField(
                            name="ref_id",
                            type=SearchFieldDataType.String,
                            retrievable=True,
                        ),
                    ],
                    collection=True,
                ),
                # Image metadata
                ComplexField(
                    name=self.schema_config["images_field"],
                    fields=[
                        SimpleField(
                            name="image_id",
                            type=SearchFieldDataType.String,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="page_number",
                            type=SearchFieldDataType.Int32,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="bounding_box",
                            type=SearchFieldDataType.Collection(
                                SearchFieldDataType.Double
                            ),
                            retrievable=True,
                        ),
                        SimpleField(
                            name="ocr_text",
                            type=SearchFieldDataType.String,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="ocr_confidence",
                            type=SearchFieldDataType.Double,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="width",
                            type=SearchFieldDataType.Int32,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="height",
                            type=SearchFieldDataType.Int32,
                            retrievable=True,
                        ),
                    ],
                    collection=True,
                ),
                # Processing metadata
                ComplexField(
                    name=self.schema_config["processing_metadata_field"],
                    fields=[
                        SimpleField(
                            name="processing_strategy",
                            type=SearchFieldDataType.String,
                            retrievable=True,
                            filterable=True,
                        ),
                        SimpleField(
                            name="page_count",
                            type=SearchFieldDataType.Int32,
                            retrievable=True,
                            filterable=True,
                        ),
                        SimpleField(
                            name="processing_cost_usd",
                            type=SearchFieldDataType.Double,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="quality_score",
                            type=SearchFieldDataType.Double,
                            retrievable=True,
                        ),
                        SimpleField(
                            name="processed_at",
                            type=SearchFieldDataType.DateTimeOffset,
                            retrievable=True,
                            filterable=True,
                            sortable=True,
                        ),
                    ],
                ),
                # Timestamp for sorting and filtering
                SimpleField(
                    name=self.schema_config["timestamp_field"],
                    type=SearchFieldDataType.DateTimeOffset,
                    retrievable=True,
                    filterable=True,
                    sortable=True,
                ),
            ]

            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default-hnsw-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters=HnswParameters(
                            m=4, ef_construction=400, ef_search=500, metric="cosine"
                        ),
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-hnsw-config",
                    )
                ],
            )

            # Create the index
            index = SearchIndex(
                name=index_name, fields=fields, vector_search=vector_search
            )

            created_index = self.index_client.create_index(index)

            self.logger.info(
                "Index schema created successfully",
                index_name=index_name,
                vector_dimension=vector_dimension,
                field_count=len(fields),
            )

            return True

        except Exception as e:
            self.logger.error(
                "Failed to create index schema",
                index_name=index_name,
                error=str(e),
                exc_info=True,
            )
            return False

    async def _upload_documents(
        self, index_name: str, processed_documents: List[Dict[str, Any]]
    ) -> int:
        """
        Upload processed documents to the Azure Search index.

        Converts document format to match exact app expectations.
        """
        search_client = SearchClient(
            endpoint=self.settings.search_service_endpoint,
            index_name=index_name,
            credential=self.credential,
        )

        # Convert documents to Azure Search format
        azure_documents = []
        for doc in processed_documents:
            azure_doc = self._convert_to_azure_search_format(doc)
            azure_documents.append(azure_doc)

        # Upload in batches to handle large document sets
        batch_size = 1000
        documents_uploaded = 0

        for i in range(0, len(azure_documents), batch_size):
            batch = azure_documents[i : i + batch_size]

            try:
                # Upload batch
                result = search_client.upload_documents(documents=batch)

                # Count successful uploads
                successful_uploads = sum(1 for r in result if r.succeeded)
                documents_uploaded += successful_uploads

                # Log any failures
                failed_uploads = [r for r in result if not r.succeeded]
                if failed_uploads:
                    self.logger.warning(
                        "Some documents failed to upload",
                        batch_start=i,
                        failed_count=len(failed_uploads),
                        total_batch_size=len(batch),
                    )

                self.logger.debug(
                    "Batch uploaded",
                    batch_start=i,
                    batch_size=len(batch),
                    successful=successful_uploads,
                    failed=len(failed_uploads),
                )

            except Exception as e:
                self.logger.error(
                    "Failed to upload document batch",
                    batch_start=i,
                    batch_size=len(batch),
                    error=str(e),
                )
                # Continue with next batch rather than failing entirely
                continue

        return documents_uploaded

    def _convert_to_azure_search_format(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert processed document to Azure Search format.

        This conversion ensures exact compatibility with the existing app's
        expectations for field names and data structures.
        """
        # Extract embeddings - use first embedding if multiple chunks
        content_vector = []
        if doc.get("embeddings") and len(doc["embeddings"]) > 0:
            content_vector = doc["embeddings"][
                0
            ]  # Use first embedding for main content

        # Convert chunks to Azure Search format
        chunks = []
        for chunk in doc.get("chunks", []):
            chunks.append(
                {
                    "text": chunk.get("text", ""),
                    "page_number": chunk.get("page_number", 1),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "bounding_box": chunk.get("bounding_box", []),
                    "char_start": chunk.get("char_start", 0),
                    "char_end": chunk.get("char_end", 0),
                }
            )

        # Convert location metadata to exact app format
        location_metadata = []
        for loc_id, loc_data in doc.get("locationMetadata", {}).items():
            location_metadata.append(
                {
                    "page_number": loc_data.get("page_number", 1),
                    "bounding_polygons": [loc_data.get("bounding_polygons", [])],
                    "ref_id": loc_data.get("ref_id", loc_id),
                }
            )

        # Convert images to Azure Search format
        images = []
        for img in doc.get("images", []):
            images.append(
                {
                    "image_id": img.get("image_id", ""),
                    "page_number": img.get("page_number", 1),
                    "bounding_box": img.get("bounding_box", []),
                    "ocr_text": img.get("ocr_text", ""),
                    "ocr_confidence": img.get("ocr_confidence", 0.0),
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                }
            )

        # Create Azure Search document
        azure_doc = {
            # Core fields
            self.schema_config["id_field"]: doc.get("id", str(uuid.uuid4())),
            self.schema_config["content_field"]: doc.get("content", ""),
            self.schema_config["title_field"]: doc.get("file_name", ""),
            self.schema_config["filename_field"]: doc.get("file_name", ""),
            self.schema_config["url_field"]: doc.get("file_path", ""),
            # Vector field
            self.schema_config["content_vector_field"]: content_vector,
            # Visual citation fields (critical for app compatibility)
            self.schema_config["chunks_field"]: chunks,
            self.schema_config["location_metadata_field"]: location_metadata,
            self.schema_config["images_field"]: images,
            # Processing metadata
            self.schema_config["processing_metadata_field"]: {
                "processing_strategy": doc.get("metadata", {}).get(
                    "processing_strategy", "unknown"
                ),
                "page_count": doc.get("metadata", {}).get("page_count", 1),
                "processing_cost_usd": doc.get("metadata", {}).get(
                    "processing_cost_usd", 0.0
                ),
                "quality_score": doc.get("metadata", {}).get("quality_score", 0.0),
                "processed_at": doc.get("metadata", {}).get(
                    "processed_at", datetime.utcnow().isoformat()
                ),
            },
            # Timestamp
            self.schema_config["timestamp_field"]: datetime.utcnow().isoformat(),
        }

        return azure_doc

    def _estimate_index_size(self, processed_documents: List[Dict[str, Any]]) -> float:
        """Estimate index size in MB based on document content."""
        total_size = 0

        for doc in processed_documents:
            # Estimate size based on content and metadata
            content_size = len(doc.get("content", "").encode("utf-8"))

            # Add vector data size
            embeddings = doc.get("embeddings", [])
            vector_size = sum(len(emb) * 4 for emb in embeddings)  # 4 bytes per float

            # Add metadata size
            metadata_size = len(str(doc.get("metadata", {})).encode("utf-8"))

            # Add chunk and location data size
            chunks_size = len(str(doc.get("chunks", [])).encode("utf-8"))
            location_size = len(str(doc.get("locationMetadata", {})).encode("utf-8"))

            doc_size = (
                content_size + vector_size + metadata_size + chunks_size + location_size
            )
            total_size += doc_size

        # Convert to MB and add overhead estimate
        size_mb = (total_size / (1024 * 1024)) * 1.5  # 50% overhead estimate
        return round(size_mb, 2)

    async def delete_index(self, index_name: str) -> bool:
        """Delete an Azure Search index."""
        try:
            self.index_client.delete_index(index_name)
            self.logger.info("Index deleted successfully", index_name=index_name)
            return True
        except ResourceNotFoundError:
            self.logger.warning("Index not found for deletion", index_name=index_name)
            return False
        except Exception as e:
            self.logger.error(
                "Failed to delete index", index_name=index_name, error=str(e)
            )
            return False

    async def list_indexes(self) -> List[str]:
        """List all indexes in the Azure Search service."""
        try:
            indexes = self.index_client.list_indexes()
            index_names = [index.name for index in indexes]
            self.logger.debug("Listed indexes", count=len(index_names))
            return index_names
        except Exception as e:
            self.logger.error("Failed to list indexes", error=str(e))
            return []

    async def get_index_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific index."""
        try:
            search_client = SearchClient(
                endpoint=self.settings.search_service_endpoint,
                index_name=index_name,
                credential=self.credential,
            )

            # Get document count (approximate)
            count_result = search_client.search(
                search_text="*", include_total_count=True, top=0
            )

            document_count = count_result.get_count()

            # Get index definition for additional stats
            index_def = self.index_client.get_index(index_name)

            return {
                "index_name": index_name,
                "document_count": document_count,
                "field_count": len(index_def.fields),
                "vector_search_enabled": index_def.vector_search is not None,
                "created_date": index_def.e_tag,  # Approximate creation info
            }

        except Exception as e:
            self.logger.error(
                "Failed to get index stats", index_name=index_name, error=str(e)
            )
            return None

    def validate_app_compatibility(self, index_name: str) -> Dict[str, Any]:
        """
        Validate that an index is compatible with the existing app.

        Checks for required fields and schema structure.
        """
        try:
            index_def = self.index_client.get_index(index_name)

            # Check for required fields
            field_names = {field.name for field in index_def.fields}
            required_fields = {
                self.schema_config["id_field"],
                self.schema_config["content_field"],
                self.schema_config["chunks_field"],
                self.schema_config["location_metadata_field"],
                self.schema_config["content_vector_field"],
            }

            missing_fields = required_fields - field_names
            has_vector_search = index_def.vector_search is not None

            is_compatible = len(missing_fields) == 0 and has_vector_search

            return {
                "compatible": is_compatible,
                "missing_fields": list(missing_fields),
                "has_vector_search": has_vector_search,
                "total_fields": len(field_names),
                "validation_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "compatible": False,
                "error": str(e),
                "validation_timestamp": datetime.utcnow().isoformat(),
            }
