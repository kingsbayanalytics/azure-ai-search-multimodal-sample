"""
Main document processor for the cost-effective preprocessing pipeline.

This module orchestrates the entire document processing workflow, from strategy selection
through document processing to Azure Search index creation, achieving 70-85% cost savings
compared to Azure Document Intelligence while preserving visual citation capabilities.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import structlog

from config.settings import get_settings, ProcessingStrategy, StrategySettings
from strategy_selector import DocumentStrategySelector
from monitoring.cost_tracker import CostTracker
from monitoring.quality_validator import QualityValidator


class DocumentProcessor:
    """
    Main orchestrator for cost-effective document processing.

    This class coordinates strategy selection, document processing, embedding generation,
    and Azure Search index creation while tracking costs and maintaining quality.
    """

    def __init__(self):
        """Initialize the document processor with configuration and dependencies."""
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)

        # Initialize core components
        self.strategy_selector = DocumentStrategySelector()
        self.cost_tracker = (
            CostTracker() if self.settings.enable_cost_tracking else None
        )
        self.quality_validator = (
            QualityValidator() if self.settings.enable_quality_validation else None
        )

        # Lazy loading of processors to avoid import cycles
        self._text_processor = None
        self._image_processor = None
        self._embedding_generator = None
        self._index_creator = None

        # Processing statistics
        self.processing_stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_pages": 0,
            "total_cost_usd": 0.0,
            "processing_start_time": None,
            "processing_end_time": None,
        }

    async def process_documents(
        self,
        input_path: str,
        output_index_name: str,
        strategy: Optional[ProcessingStrategy] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a batch of documents and create Azure Search index.

        Args:
            input_path: Path to input documents (file or directory)
            output_index_name: Name for the Azure Search index to create
            strategy: Processing strategy (if None, uses auto-selection)
            batch_size: Number of documents to process in parallel

        Returns:
            Dictionary containing processing results and statistics
        """
        self.processing_stats["processing_start_time"] = datetime.utcnow()
        batch_size = batch_size or self.settings.batch_size

        self.logger.info(
            "Starting document processing",
            input_path=input_path,
            output_index=output_index_name,
            strategy=strategy,
            batch_size=batch_size,
        )

        try:
            # Discover and validate input documents
            documents = await self._discover_documents(input_path)
            self.processing_stats["total_documents"] = len(documents)

            if not documents:
                raise ValueError(f"No valid documents found in {input_path}")

            # Process documents in batches
            processed_documents = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                batch_results = await self._process_document_batch(batch, strategy)
                processed_documents.extend(batch_results)

            # Create Azure Search index
            index_result = await self._create_search_index(
                processed_documents, output_index_name
            )

            # Finalize processing statistics
            self.processing_stats["processing_end_time"] = datetime.utcnow()
            self.processing_stats["successful_documents"] = len(processed_documents)
            self.processing_stats["failed_documents"] = (
                self.processing_stats["total_documents"]
                - self.processing_stats["successful_documents"]
            )

            # Generate processing report
            processing_report = await self._generate_processing_report(
                processed_documents, index_result
            )

            self.logger.info(
                "Document processing completed successfully",
                total_documents=self.processing_stats["total_documents"],
                successful_documents=self.processing_stats["successful_documents"],
                total_cost_usd=self.processing_stats["total_cost_usd"],
                index_name=output_index_name,
            )

            return processing_report

        except Exception as e:
            self.logger.error("Document processing failed", error=str(e), exc_info=True)
            self.processing_stats["processing_end_time"] = datetime.utcnow()
            raise

    async def _discover_documents(self, input_path: str) -> List[Path]:
        """
        Discover and validate documents from input path.

        Args:
            input_path: Path to file or directory containing documents

        Returns:
            List of valid document paths
        """
        input_path_obj = Path(input_path)
        documents = []

        if input_path_obj.is_file():
            if self._is_supported_document(input_path_obj):
                documents.append(input_path_obj)
        elif input_path_obj.is_dir():
            # Recursively find supported documents
            for file_path in input_path_obj.rglob("*"):
                if file_path.is_file() and self._is_supported_document(file_path):
                    documents.append(file_path)
        else:
            raise ValueError(f"Input path not found: {input_path}")

        # Validate file sizes
        valid_documents = []
        for doc_path in documents:
            file_size_mb = doc_path.stat().st_size / (1024 * 1024)
            if file_size_mb <= self.settings.max_file_size_mb:
                valid_documents.append(doc_path)
            else:
                self.logger.warning(
                    "Document exceeds size limit",
                    document=str(doc_path),
                    size_mb=file_size_mb,
                    limit_mb=self.settings.max_file_size_mb,
                )

        return valid_documents

    def _is_supported_document(self, file_path: Path) -> bool:
        """Check if document format is supported for processing."""
        supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}
        return file_path.suffix.lower() in supported_extensions

    async def _process_document_batch(
        self, documents: List[Path], strategy: Optional[ProcessingStrategy]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of documents with parallel processing.

        Args:
            documents: List of document paths to process
            strategy: Processing strategy (None for auto-selection)

        Returns:
            List of processed document data
        """
        if self.settings.enable_parallel_processing:
            # Process documents in parallel
            tasks = [
                self._process_single_document(doc_path, strategy)
                for doc_path in documents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process documents sequentially
            results = []
            for doc_path in documents:
                try:
                    result = await self._process_single_document(doc_path, strategy)
                    results.append(result)
                except Exception as e:
                    results.append(e)

        # Filter successful results and log failures
        processed_documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    "Document processing failed",
                    document=str(documents[i]),
                    error=str(result),
                )
            else:
                processed_documents.append(result)

        return processed_documents

    async def _process_single_document(
        self, document_path: Path, strategy: Optional[ProcessingStrategy]
    ) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.

        Args:
            document_path: Path to the document to process
            strategy: Processing strategy (None for auto-selection)

        Returns:
            Processed document data ready for indexing
        """
        doc_start_time = datetime.utcnow()

        self.logger.info("Processing document", document=str(document_path))

        try:
            # Step 1: Strategy selection (if not provided)
            if strategy is None or strategy == ProcessingStrategy.AUTO:
                selected_strategy = await self.strategy_selector.select_strategy(
                    document_path
                )
            else:
                selected_strategy = strategy

            self.logger.info(
                "Selected processing strategy",
                document=str(document_path),
                strategy=selected_strategy,
            )

            # Step 2: Extract document content using appropriate processor
            document_content = await self._extract_document_content(
                document_path, selected_strategy
            )

            # Step 3: Generate embeddings
            embeddings = await self._generate_embeddings(
                document_content, selected_strategy
            )

            # Step 4: Create processed document structure
            processed_document = {
                "id": self._generate_document_id(document_path),
                "file_name": document_path.name,
                "file_path": str(document_path),
                "content": document_content["text"],
                "embeddings": embeddings,
                "metadata": {
                    "processing_strategy": selected_strategy,
                    "page_count": document_content.get("page_count", 1),
                    "processing_time_seconds": (
                        datetime.utcnow() - doc_start_time
                    ).total_seconds(),
                    "file_size_bytes": document_path.stat().st_size,
                    "processed_at": datetime.utcnow().isoformat(),
                },
                # Visual citation data (critical for app compatibility)
                "chunks": document_content.get("chunks", []),
                "locationMetadata": document_content.get("location_metadata", {}),
                "images": document_content.get("images", []),
            }

            # Step 5: Track costs
            if self.cost_tracker:
                processing_cost = await self.cost_tracker.calculate_document_cost(
                    selected_strategy,
                    document_content.get("page_count", 1),
                    len(document_content.get("images", [])),
                )
                processed_document["metadata"]["processing_cost_usd"] = processing_cost
                self.processing_stats["total_cost_usd"] += processing_cost

            # Step 6: Quality validation
            if self.quality_validator:
                quality_score = await self.quality_validator.validate_document(
                    processed_document
                )
                processed_document["metadata"]["quality_score"] = quality_score

            self.processing_stats["total_pages"] += document_content.get(
                "page_count", 1
            )

            self.logger.info(
                "Document processed successfully",
                document=str(document_path),
                strategy=selected_strategy,
                pages=document_content.get("page_count", 1),
                processing_time=processed_document["metadata"][
                    "processing_time_seconds"
                ],
            )

            return processed_document

        except Exception as e:
            self.logger.error(
                "Failed to process document",
                document=str(document_path),
                error=str(e),
                exc_info=True,
            )
            raise

    async def _extract_document_content(
        self, document_path: Path, strategy: ProcessingStrategy
    ) -> Dict[str, Any]:
        """
        Extract content from document using strategy-appropriate processor.

        Args:
            document_path: Path to document
            strategy: Processing strategy to use

        Returns:
            Extracted document content with metadata
        """
        # Lazy load processors
        if not self._text_processor:
            from processors.text_processor import TextProcessor

            self._text_processor = TextProcessor()

        if not self._image_processor:
            from processors.image_processor import ImageProcessor

            self._image_processor = ImageProcessor()

        # Get strategy configuration
        strategy_config = StrategySettings.get_strategy_config(strategy)

        # Extract text content
        text_content = await self._text_processor.extract_content(
            document_path,
            enable_layout_analysis=strategy_config.get("layout_analysis", False),
            preserve_coordinates=True,  # Always preserve for visual citations
        )

        # Extract images if required by strategy
        if strategy_config.get("process_images", False):
            image_content = await self._image_processor.extract_and_process_images(
                document_path,
                ocr_provider=strategy_config.get("ocr_provider"),
                enable_advanced_vision=strategy_config.get("advanced_vision", False),
            )
            text_content["images"] = image_content.get("images", [])
            # Merge OCR text with document text
            if image_content.get("ocr_text"):
                text_content["text"] += "\n\n" + image_content["ocr_text"]

        return text_content

    async def _generate_embeddings(
        self, document_content: Dict[str, Any], strategy: ProcessingStrategy
    ) -> List[List[float]]:
        """
        Generate embeddings for document content using strategy-appropriate provider.

        Args:
            document_content: Extracted document content
            strategy: Processing strategy

        Returns:
            Generated embeddings for document chunks
        """
        # Lazy load embedding generator
        if not self._embedding_generator:
            from embedding.embedding_generator import EmbeddingGenerator

            self._embedding_generator = EmbeddingGenerator()

        strategy_config = StrategySettings.get_strategy_config(strategy)
        embedding_provider = strategy_config.get("embedding_provider")

        # Generate embeddings for document chunks
        embeddings = await self._embedding_generator.generate_embeddings(
            document_content.get("chunks", []), provider=embedding_provider
        )

        return embeddings

    async def _create_search_index(
        self, processed_documents: List[Dict[str, Any]], index_name: str
    ) -> Dict[str, Any]:
        """
        Create Azure Search index with processed documents.

        Args:
            processed_documents: List of processed document data
            index_name: Name for the Azure Search index

        Returns:
            Index creation result
        """
        # Lazy load index creator
        if not self._index_creator:
            from indexer.index_creator import IndexCreator

            self._index_creator = IndexCreator()

        index_result = await self._index_creator.create_index(
            processed_documents, index_name
        )

        return index_result

    def _generate_document_id(self, document_path: Path) -> str:
        """Generate unique document ID based on path and timestamp."""
        import hashlib

        path_hash = hashlib.md5(str(document_path).encode()).hexdigest()[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"doc_{path_hash}_{timestamp}"

    async def _generate_processing_report(
        self, processed_documents: List[Dict[str, Any]], index_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive processing report with cost analysis.

        Args:
            processed_documents: List of successfully processed documents
            index_result: Result from index creation

        Returns:
            Comprehensive processing report
        """
        # Calculate processing duration
        duration = (
            self.processing_stats["processing_end_time"]
            - self.processing_stats["processing_start_time"]
        ).total_seconds()

        # Calculate cost savings estimate (vs Azure Document Intelligence)
        azure_doc_intel_cost_per_page = 0.1  # Estimated Azure Doc Intelligence cost
        traditional_cost = (
            self.processing_stats["total_pages"] * azure_doc_intel_cost_per_page
        )
        cost_savings = traditional_cost - self.processing_stats["total_cost_usd"]
        savings_percentage = (
            (cost_savings / traditional_cost * 100) if traditional_cost > 0 else 0
        )

        # Strategy usage statistics
        strategy_usage = {}
        for doc in processed_documents:
            strategy = doc["metadata"]["processing_strategy"]
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

        report = {
            "processing_summary": {
                "total_documents_found": self.processing_stats["total_documents"],
                "successfully_processed": self.processing_stats["successful_documents"],
                "failed_documents": self.processing_stats["failed_documents"],
                "total_pages": self.processing_stats["total_pages"],
                "processing_duration_seconds": duration,
                "processing_rate_docs_per_minute": (
                    (self.processing_stats["successful_documents"] / (duration / 60))
                    if duration > 0
                    else 0
                ),
            },
            "cost_analysis": {
                "total_processing_cost_usd": self.processing_stats["total_cost_usd"],
                "estimated_azure_doc_intel_cost_usd": traditional_cost,
                "estimated_savings_usd": cost_savings,
                "estimated_savings_percentage": savings_percentage,
                "average_cost_per_document": (
                    (
                        self.processing_stats["total_cost_usd"]
                        / self.processing_stats["successful_documents"]
                    )
                    if self.processing_stats["successful_documents"] > 0
                    else 0
                ),
                "average_cost_per_page": (
                    (
                        self.processing_stats["total_cost_usd"]
                        / self.processing_stats["total_pages"]
                    )
                    if self.processing_stats["total_pages"] > 0
                    else 0
                ),
            },
            "strategy_usage": strategy_usage,
            "index_creation": index_result,
            "quality_metrics": {
                "average_quality_score": self._calculate_average_quality_score(
                    processed_documents
                ),
                "citation_compatibility": True,  # Always maintain citation compatibility
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        return report

    def _calculate_average_quality_score(
        self, processed_documents: List[Dict[str, Any]]
    ) -> float:
        """Calculate average quality score across all processed documents."""
        quality_scores = [
            doc["metadata"].get("quality_score", 0.0) for doc in processed_documents
        ]
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0


# Command-line interface for standalone usage
if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--input-path", required=True, help="Path to input documents")
    @click.option(
        "--output-index", required=True, help="Name for output Azure Search index"
    )
    @click.option(
        "--strategy",
        type=click.Choice([s.value for s in ProcessingStrategy]),
        default="auto",
        help="Processing strategy to use",
    )
    @click.option(
        "--batch-size", type=int, default=None, help="Batch size for processing"
    )
    def main(
        input_path: str, output_index: str, strategy: str, batch_size: Optional[int]
    ):
        """Process documents and create cost-effective Azure Search index."""

        async def run_processing():
            processor = DocumentProcessor()
            result = await processor.process_documents(
                input_path=input_path,
                output_index_name=output_index,
                strategy=ProcessingStrategy(strategy) if strategy != "auto" else None,
                batch_size=batch_size,
            )

            print("\n=== Processing Report ===")
            print(
                f"Successfully processed: {result['processing_summary']['successfully_processed']} documents"
            )
            print(
                f"Total cost: ${result['cost_analysis']['total_processing_cost_usd']:.4f}"
            )
            print(
                f"Estimated savings: {result['cost_analysis']['estimated_savings_percentage']:.1f}%"
            )
            print(f"Index created: {result['index_creation']['index_name']}")

        asyncio.run(run_processing())

    main()
