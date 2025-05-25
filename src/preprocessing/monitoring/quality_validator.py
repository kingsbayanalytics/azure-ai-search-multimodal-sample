"""
Quality validation module for the preprocessing pipeline.

This module validates the quality of processed documents to ensure that
cost optimization doesn't compromise search accuracy or visual citation
functionality. Provides quality scores and validation metrics.
"""

import asyncio
from typing import Dict, Any, List, Optional
import structlog
from dataclasses import dataclass
import re
import statistics

from config.settings import get_settings


@dataclass
class QualityMetrics:
    """Quality metrics for a processed document."""

    overall_score: float  # 0.0 to 1.0
    content_quality: float
    citation_quality: float
    embedding_quality: float
    metadata_completeness: float
    processing_consistency: float
    validation_details: Dict[str, Any]


class QualityValidator:
    """
    Quality validator for preprocessed documents.

    Validates that cost-effective processing maintains quality standards
    for search accuracy and visual citation functionality.
    """

    def __init__(self):
        """Initialize the quality validator with configuration."""
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)

        # Quality thresholds
        self.quality_thresholds = {
            "minimum_overall_score": 0.7,
            "minimum_content_quality": 0.8,
            "minimum_citation_quality": 0.95,  # High requirement for citations
            "minimum_embedding_quality": 0.7,
            "minimum_metadata_completeness": 0.8,
        }

    async def validate_document(self, processed_document: Dict[str, Any]) -> float:
        """
        Validate quality of a single processed document.

        Args:
            processed_document: Processed document data

        Returns:
            Overall quality score from 0.0 to 1.0
        """
        try:
            metrics = await self._calculate_quality_metrics(processed_document)

            self.logger.debug(
                "Document quality validated",
                document_id=processed_document.get("id", "unknown"),
                overall_score=metrics.overall_score,
                content_quality=metrics.content_quality,
                citation_quality=metrics.citation_quality,
            )

            return metrics.overall_score

        except Exception as e:
            self.logger.error(
                "Quality validation failed",
                document_id=processed_document.get("id", "unknown"),
                error=str(e),
            )
            return 0.0

    async def validate_document_batch(
        self, processed_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate quality of a batch of processed documents.

        Args:
            processed_documents: List of processed documents

        Returns:
            Batch quality validation results
        """
        if not processed_documents:
            return {
                "batch_size": 0,
                "average_quality": 0.0,
                "quality_distribution": {},
                "validation_summary": {},
            }

        document_scores = []
        quality_details = []

        for doc in processed_documents:
            try:
                metrics = await self._calculate_quality_metrics(doc)
                document_scores.append(metrics.overall_score)
                quality_details.append(
                    {"document_id": doc.get("id", "unknown"), "metrics": metrics}
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to validate document quality",
                    document_id=doc.get("id", "unknown"),
                    error=str(e),
                )
                document_scores.append(0.0)

        # Calculate batch statistics
        average_quality = statistics.mean(document_scores) if document_scores else 0.0
        median_quality = statistics.median(document_scores) if document_scores else 0.0

        # Quality distribution
        quality_ranges = {
            "excellent": sum(1 for score in document_scores if score >= 0.9),
            "good": sum(1 for score in document_scores if 0.8 <= score < 0.9),
            "acceptable": sum(1 for score in document_scores if 0.7 <= score < 0.8),
            "poor": sum(1 for score in document_scores if score < 0.7),
        }

        # Validation summary
        validation_summary = {
            "total_documents": len(processed_documents),
            "validated_documents": len(document_scores),
            "passed_minimum_threshold": sum(
                1
                for score in document_scores
                if score >= self.quality_thresholds["minimum_overall_score"]
            ),
            "average_quality": average_quality,
            "median_quality": median_quality,
            "min_quality": min(document_scores) if document_scores else 0.0,
            "max_quality": max(document_scores) if document_scores else 0.0,
        }

        return {
            "batch_size": len(processed_documents),
            "average_quality": average_quality,
            "quality_distribution": quality_ranges,
            "validation_summary": validation_summary,
            "document_details": quality_details,
        }

    async def _calculate_quality_metrics(
        self, processed_document: Dict[str, Any]
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics for a document."""

        # Content quality assessment
        content_quality = await self._assess_content_quality(processed_document)

        # Citation quality assessment (critical for app compatibility)
        citation_quality = await self._assess_citation_quality(processed_document)

        # Embedding quality assessment
        embedding_quality = await self._assess_embedding_quality(processed_document)

        # Metadata completeness assessment
        metadata_completeness = await self._assess_metadata_completeness(
            processed_document
        )

        # Processing consistency assessment
        processing_consistency = await self._assess_processing_consistency(
            processed_document
        )

        # Calculate overall score with weighted components
        weights = {
            "content": 0.25,
            "citation": 0.30,  # High weight for citation quality
            "embedding": 0.20,
            "metadata": 0.15,
            "consistency": 0.10,
        }

        overall_score = (
            content_quality * weights["content"]
            + citation_quality * weights["citation"]
            + embedding_quality * weights["embedding"]
            + metadata_completeness * weights["metadata"]
            + processing_consistency * weights["consistency"]
        )

        validation_details = {
            "content_length": len(processed_document.get("content", "")),
            "chunk_count": len(processed_document.get("chunks", [])),
            "has_location_metadata": bool(processed_document.get("locationMetadata")),
            "has_embeddings": bool(processed_document.get("embeddings")),
            "processing_strategy": processed_document.get("metadata", {}).get(
                "processing_strategy"
            ),
            "validation_timestamp": asyncio.get_event_loop().time(),
        }

        return QualityMetrics(
            overall_score=min(overall_score, 1.0),
            content_quality=content_quality,
            citation_quality=citation_quality,
            embedding_quality=embedding_quality,
            metadata_completeness=metadata_completeness,
            processing_consistency=processing_consistency,
            validation_details=validation_details,
        )

    async def _assess_content_quality(self, document: Dict[str, Any]) -> float:
        """Assess the quality of extracted text content."""
        content = document.get("content", "")

        if not content:
            return 0.0

        quality_score = 0.0

        # Text length appropriateness (not too short, not empty)
        if len(content) > 100:
            quality_score += 0.3
        elif len(content) > 50:
            quality_score += 0.15

        # Character variety (indicates good extraction)
        unique_chars = len(set(content.lower()))
        if unique_chars > 50:
            quality_score += 0.2
        elif unique_chars > 20:
            quality_score += 0.1

        # Word formation (indicates proper text extraction)
        words = content.split()
        valid_words = sum(1 for word in words if len(word) > 1 and word.isalnum())
        word_ratio = valid_words / len(words) if words else 0
        quality_score += word_ratio * 0.3

        # Sentence structure (indicates coherent extraction)
        sentences = re.split(r"[.!?]+", content)
        avg_sentence_length = (
            sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        )
        if 5 <= avg_sentence_length <= 50:  # Reasonable sentence length
            quality_score += 0.2

        return min(quality_score, 1.0)

    async def _assess_citation_quality(self, document: Dict[str, Any]) -> float:
        """Assess quality of visual citation data (critical for app compatibility)."""
        quality_score = 0.0

        chunks = document.get("chunks", [])
        location_metadata = document.get("locationMetadata", {})

        # Check if chunks exist and have required fields
        if chunks:
            quality_score += 0.3

            # Validate chunk structure
            valid_chunks = 0
            for chunk in chunks:
                required_fields = ["text", "page_number", "chunk_id", "bounding_box"]
                if all(field in chunk for field in required_fields):
                    # Validate bounding box format
                    bbox = chunk.get("bounding_box", [])
                    if len(bbox) == 4 and all(
                        isinstance(x, (int, float)) for x in bbox
                    ):
                        valid_chunks += 1

            chunk_validity_ratio = valid_chunks / len(chunks)
            quality_score += chunk_validity_ratio * 0.3

        # Check location metadata structure
        if location_metadata:
            quality_score += 0.2

            # Validate location metadata structure
            valid_locations = 0
            for loc_id, loc_data in location_metadata.items():
                required_fields = ["page_number", "bounding_polygons", "ref_id"]
                if all(field in loc_data for field in required_fields):
                    # Validate bounding polygons format
                    polygons = loc_data.get("bounding_polygons", [])
                    if polygons and all(
                        len(poly) >= 6 for poly in polygons
                    ):  # At least 3 points
                        valid_locations += 1

            if location_metadata:
                location_validity_ratio = valid_locations / len(location_metadata)
                quality_score += location_validity_ratio * 0.2

        return min(quality_score, 1.0)

    async def _assess_embedding_quality(self, document: Dict[str, Any]) -> float:
        """Assess quality of generated embeddings."""
        embeddings = document.get("embeddings", [])

        if not embeddings:
            return 0.0

        quality_score = 0.0

        # Check if embeddings exist
        quality_score += 0.4

        # Validate embedding dimensions and values
        valid_embeddings = 0
        for embedding in embeddings:
            if isinstance(embedding, list) and len(embedding) > 0:
                # Check for valid numeric values
                if all(isinstance(x, (int, float)) for x in embedding):
                    # Check for reasonable magnitude (should be normalized)
                    import math

                    magnitude = math.sqrt(sum(x * x for x in embedding))
                    if 0.1 <= magnitude <= 10:  # Reasonable range
                        valid_embeddings += 1

        if embeddings:
            embedding_validity_ratio = valid_embeddings / len(embeddings)
            quality_score += embedding_validity_ratio * 0.6

        return min(quality_score, 1.0)

    async def _assess_metadata_completeness(self, document: Dict[str, Any]) -> float:
        """Assess completeness of document metadata."""
        metadata = document.get("metadata", {})

        if not metadata:
            return 0.0

        # Required metadata fields
        required_fields = ["processing_strategy", "page_count", "processed_at"]

        # Optional but valuable fields
        optional_fields = ["processing_cost_usd", "quality_score", "file_size_bytes"]

        # Calculate completeness score
        required_present = sum(1 for field in required_fields if field in metadata)
        optional_present = sum(1 for field in optional_fields if field in metadata)

        required_score = required_present / len(required_fields)
        optional_score = optional_present / len(optional_fields)

        # Weight required fields more heavily
        completeness_score = (required_score * 0.8) + (optional_score * 0.2)

        return min(completeness_score, 1.0)

    async def _assess_processing_consistency(self, document: Dict[str, Any]) -> float:
        """Assess consistency of processing across document components."""
        quality_score = 0.0

        # Check consistency between content and chunks
        content = document.get("content", "")
        chunks = document.get("chunks", [])

        if content and chunks:
            # Check if chunk text appears in main content
            chunk_texts = [chunk.get("text", "") for chunk in chunks]
            content_coverage = sum(
                1 for chunk_text in chunk_texts if chunk_text and chunk_text in content
            )

            if chunk_texts:
                coverage_ratio = content_coverage / len(chunk_texts)
                quality_score += coverage_ratio * 0.4

        # Check consistency between chunks and location metadata
        location_metadata = document.get("locationMetadata", {})

        if chunks and location_metadata:
            chunk_ids = {chunk.get("chunk_id") for chunk in chunks}
            location_refs = {
                loc_data.get("ref_id") for loc_data in location_metadata.values()
            }

            # Check if location metadata references match chunk IDs
            matching_refs = chunk_ids.intersection(location_refs)
            if chunk_ids:
                ref_consistency = len(matching_refs) / len(chunk_ids)
                quality_score += ref_consistency * 0.3

        # Check page number consistency
        page_numbers_chunks = {
            chunk.get("page_number") for chunk in chunks if chunk.get("page_number")
        }
        page_numbers_locations = {
            loc_data.get("page_number")
            for loc_data in location_metadata.values()
            if loc_data.get("page_number")
        }

        if page_numbers_chunks and page_numbers_locations:
            page_overlap = page_numbers_chunks.intersection(page_numbers_locations)
            page_consistency = len(page_overlap) / max(
                len(page_numbers_chunks), len(page_numbers_locations)
            )
            quality_score += page_consistency * 0.3

        return min(quality_score, 1.0)

    def get_quality_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable quality report."""
        if not validation_results:
            return "No validation results available."

        report = []
        report.append("=== Document Quality Validation Report ===")

        summary = validation_results.get("validation_summary", {})
        report.append(f"Total Documents: {summary.get('total_documents', 0)}")
        report.append(f"Average Quality: {summary.get('average_quality', 0.0):.3f}")
        report.append(f"Passed Threshold: {summary.get('passed_minimum_threshold', 0)}")

        distribution = validation_results.get("quality_distribution", {})
        report.append("\nQuality Distribution:")
        report.append(f"  Excellent (≥0.9): {distribution.get('excellent', 0)}")
        report.append(f"  Good (0.8-0.9): {distribution.get('good', 0)}")
        report.append(f"  Acceptable (0.7-0.8): {distribution.get('acceptable', 0)}")
        report.append(f"  Poor (<0.7): {distribution.get('poor', 0)}")

        # Quality recommendations
        avg_quality = summary.get("average_quality", 0.0)
        report.append("\nRecommendations:")

        if avg_quality >= 0.9:
            report.append(
                "  ✓ Excellent quality! Processing pipeline is working optimally."
            )
        elif avg_quality >= 0.8:
            report.append(
                "  ✓ Good quality. Consider minor optimizations for consistency."
            )
        elif avg_quality >= 0.7:
            report.append(
                "  ⚠ Acceptable quality. Review processing strategies for improvements."
            )
        else:
            report.append(
                "  ⚠ Quality below threshold. Review and adjust processing pipeline."
            )

        return "\n".join(report)

    def validate_citation_compatibility(self, document: Dict[str, Any]) -> bool:
        """
        Validate that document is compatible with app's visual citation requirements.

        This is a critical validation for ensuring the app can display citations.
        """
        try:
            # Check required citation fields
            chunks = document.get("chunks", [])
            location_metadata = document.get("locationMetadata", {})

            if not chunks:
                return False

            # Validate chunk structure
            for chunk in chunks:
                required_fields = ["text", "page_number", "chunk_id", "bounding_box"]
                if not all(field in chunk for field in required_fields):
                    return False

                # Validate bounding box format
                bbox = chunk.get("bounding_box", [])
                if not (
                    len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox)
                ):
                    return False

            # Validate location metadata if present
            if location_metadata:
                for loc_data in location_metadata.values():
                    required_fields = ["page_number", "bounding_polygons", "ref_id"]
                    if not all(field in loc_data for field in required_fields):
                        return False

            return True

        except Exception as e:
            self.logger.error("Citation compatibility validation failed", error=str(e))
            return False
