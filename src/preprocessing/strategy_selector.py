"""
Document Strategy Selector for cost-effective preprocessing pipeline.

This module analyzes document characteristics to automatically select the most
cost-effective processing strategy while maintaining quality requirements.
The strategy selection is critical for achieving 70-85% cost savings.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import structlog
from dataclasses import dataclass
import fitz  # PyMuPDF for quick document analysis
from PIL import Image
import io

from config.settings import get_settings, ProcessingStrategy, StrategySettings


@dataclass
class DocumentAnalysis:
    """Results of document content analysis for strategy selection."""

    page_count: int
    total_text_length: int
    image_count: int
    table_count: int
    text_density: float  # Ratio of text coverage to page area
    image_ratio: float  # Ratio of image area to total page area
    layout_complexity: float  # Score from 0-1 indicating layout complexity
    has_complex_formatting: bool
    estimated_processing_difficulty: float  # Score from 0-1
    file_size_mb: float


class DocumentStrategySelector:
    """
    Intelligent document strategy selector for cost optimization.

    Analyzes document characteristics to select the most cost-effective processing
    strategy while maintaining quality requirements for visual citations and search.
    """

    def __init__(self):
        """Initialize the strategy selector with configuration."""
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)

        # Strategy selection thresholds from configuration
        self.text_density_threshold = self.settings.text_density_threshold
        self.image_ratio_threshold = self.settings.image_ratio_threshold
        self.complex_layout_threshold = self.settings.complex_layout_threshold

    async def select_strategy(
        self, document_path: Path, force_strategy: Optional[ProcessingStrategy] = None
    ) -> ProcessingStrategy:
        """
        Select the optimal processing strategy for a document.

        Args:
            document_path: Path to the document to analyze
            force_strategy: Optional strategy to force (overrides analysis)

        Returns:
            Selected processing strategy
        """
        if force_strategy and force_strategy != ProcessingStrategy.AUTO:
            self.logger.info(
                "Using forced processing strategy",
                document=str(document_path),
                strategy=force_strategy,
            )
            return force_strategy

        # Analyze document characteristics
        analysis = await self._analyze_document(document_path)

        # Apply strategy selection logic
        selected_strategy = self._apply_strategy_selection_logic(analysis)

        # Log strategy selection reasoning
        self.logger.info(
            "Strategy selected based on document analysis",
            document=str(document_path),
            strategy=selected_strategy,
            text_density=analysis.text_density,
            image_ratio=analysis.image_ratio,
            layout_complexity=analysis.layout_complexity,
            page_count=analysis.page_count,
            file_size_mb=analysis.file_size_mb,
        )

        return selected_strategy

    async def _analyze_document(self, document_path: Path) -> DocumentAnalysis:
        """
        Perform comprehensive document analysis for strategy selection.

        Args:
            document_path: Path to the document to analyze

        Returns:
            DocumentAnalysis with content characteristics
        """
        self.logger.debug("Analyzing document", document=str(document_path))

        try:
            if document_path.suffix.lower() == ".pdf":
                return await self._analyze_pdf_document(document_path)
            else:
                # For non-PDF documents, use simpler analysis
                return await self._analyze_non_pdf_document(document_path)

        except Exception as e:
            self.logger.warning(
                "Document analysis failed, using fallback strategy",
                document=str(document_path),
                error=str(e),
            )
            # Return conservative analysis for fallback
            file_size_mb = document_path.stat().st_size / (1024 * 1024)
            return DocumentAnalysis(
                page_count=1,
                total_text_length=0,
                image_count=0,
                table_count=0,
                text_density=0.5,  # Neutral
                image_ratio=0.5,  # Neutral
                layout_complexity=0.5,  # Neutral
                has_complex_formatting=True,  # Conservative
                estimated_processing_difficulty=0.7,  # Conservative
                file_size_mb=file_size_mb,
            )

    async def _analyze_pdf_document(self, document_path: Path) -> DocumentAnalysis:
        """
        Analyze PDF document characteristics using PyMuPDF.

        Args:
            document_path: Path to PDF document

        Returns:
            DocumentAnalysis with PDF-specific analysis
        """
        doc = fitz.open(str(document_path))

        try:
            page_count = len(doc)
            total_text_length = 0
            image_count = 0
            table_count = 0
            total_text_area = 0.0
            total_image_area = 0.0
            total_page_area = 0.0
            complex_formatting_indicators = 0

            # Analyze each page
            for page_num in range(
                min(page_count, 10)
            ):  # Sample first 10 pages for efficiency
                page = doc[page_num]
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height
                total_page_area += page_area

                # Extract text and calculate coverage
                text = page.get_text()
                total_text_length += len(text)

                # Analyze text blocks for density calculation
                text_blocks = page.get_text("dict")["blocks"]
                page_text_area = 0.0

                for block in text_blocks:
                    if "lines" in block:  # Text block
                        block_rect = fitz.Rect(block["bbox"])
                        page_text_area += block_rect.width * block_rect.height

                        # Check for complex formatting
                        for line in block["lines"]:
                            for span in line["spans"]:
                                # Check for multiple fonts, sizes, colors
                                if (
                                    len(set(s.get("font", "") for s in line["spans"]))
                                    > 2
                                ):
                                    complex_formatting_indicators += 1

                total_text_area += page_text_area

                # Count images
                images = page.get_images()
                page_image_count = len(images)
                image_count += page_image_count

                # Calculate image area
                page_image_area = 0.0
                for img_index, img in enumerate(images):
                    try:
                        # Get image bbox if available
                        img_rects = page.get_image_rects(img[0])
                        for rect in img_rects:
                            page_image_area += rect.width * rect.height
                    except:
                        # Fallback: estimate image area
                        page_image_area += page_area * 0.1  # Assume 10% per image

                total_image_area += page_image_area

                # Detect tables (simple heuristic)
                lines = page.get_drawings()
                if len(lines) > 10:  # Many drawing objects might indicate tables
                    table_count += 1

            doc.close()

            # Calculate metrics
            text_density = (
                (total_text_area / total_page_area) if total_page_area > 0 else 0.0
            )
            image_ratio = (
                (total_image_area / total_page_area) if total_page_area > 0 else 0.0
            )

            # Calculate layout complexity score
            layout_complexity = self._calculate_layout_complexity(
                text_density,
                image_ratio,
                table_count,
                complex_formatting_indicators,
                page_count,
            )

            # Estimate processing difficulty
            processing_difficulty = self._estimate_processing_difficulty(
                text_density, image_ratio, layout_complexity, page_count, image_count
            )

            file_size_mb = document_path.stat().st_size / (1024 * 1024)

            return DocumentAnalysis(
                page_count=page_count,
                total_text_length=total_text_length,
                image_count=image_count,
                table_count=table_count,
                text_density=min(text_density, 1.0),  # Cap at 1.0
                image_ratio=min(image_ratio, 1.0),  # Cap at 1.0
                layout_complexity=layout_complexity,
                has_complex_formatting=(complex_formatting_indicators > page_count),
                estimated_processing_difficulty=processing_difficulty,
                file_size_mb=file_size_mb,
            )

        except Exception as e:
            doc.close()
            raise

    async def _analyze_non_pdf_document(self, document_path: Path) -> DocumentAnalysis:
        """
        Analyze non-PDF document characteristics.

        Args:
            document_path: Path to non-PDF document

        Returns:
            DocumentAnalysis with basic analysis
        """
        file_size_mb = document_path.stat().st_size / (1024 * 1024)

        # Simple analysis for non-PDF documents
        if document_path.suffix.lower() in [".txt", ".md"]:
            # Text-only documents
            return DocumentAnalysis(
                page_count=1,
                total_text_length=file_size_mb * 1024,  # Rough estimate
                image_count=0,
                table_count=0,
                text_density=0.9,
                image_ratio=0.0,
                layout_complexity=0.1,
                has_complex_formatting=False,
                estimated_processing_difficulty=0.2,
                file_size_mb=file_size_mb,
            )
        else:
            # Word documents and others - assume moderate complexity
            return DocumentAnalysis(
                page_count=max(1, int(file_size_mb * 2)),  # Rough page estimate
                total_text_length=file_size_mb * 500,
                image_count=max(0, int(file_size_mb * 0.5)),  # Rough image estimate
                table_count=max(0, int(file_size_mb * 0.2)),
                text_density=0.7,
                image_ratio=0.2,
                layout_complexity=0.5,
                has_complex_formatting=True,
                estimated_processing_difficulty=0.6,
                file_size_mb=file_size_mb,
            )

    def _calculate_layout_complexity(
        self,
        text_density: float,
        image_ratio: float,
        table_count: int,
        formatting_indicators: int,
        page_count: int,
    ) -> float:
        """
        Calculate layout complexity score (0-1).

        Args:
            text_density: Ratio of text coverage
            image_ratio: Ratio of image coverage
            table_count: Number of detected tables
            formatting_indicators: Count of complex formatting elements
            page_count: Total number of pages

        Returns:
            Layout complexity score from 0 (simple) to 1 (very complex)
        """
        # Base complexity from text/image mix
        base_complexity = 0.0

        # Pure text is simple
        if text_density > 0.8 and image_ratio < 0.1:
            base_complexity = 0.1
        # Balanced text/image is moderate
        elif 0.3 <= text_density <= 0.7 and 0.1 <= image_ratio <= 0.3:
            base_complexity = 0.5
        # Image-heavy is complex
        elif image_ratio > 0.3:
            base_complexity = 0.8
        # Low text density (might be scanned) is complex
        elif text_density < 0.3:
            base_complexity = 0.7
        else:
            base_complexity = 0.4

        # Adjust for tables and formatting
        table_complexity = min(table_count / page_count * 0.3, 0.3)
        formatting_complexity = min(formatting_indicators / page_count * 0.2, 0.2)

        total_complexity = base_complexity + table_complexity + formatting_complexity
        return min(total_complexity, 1.0)

    def _estimate_processing_difficulty(
        self,
        text_density: float,
        image_ratio: float,
        layout_complexity: float,
        page_count: int,
        image_count: int,
    ) -> float:
        """
        Estimate overall processing difficulty (0-1).

        Higher scores indicate more expensive processing needed.
        """
        # Base difficulty from content type
        content_difficulty = (
            (layout_complexity * 0.5) + (image_ratio * 0.3) + ((1 - text_density) * 0.2)
        )

        # Scale difficulty
        scale_difficulty = 0.0
        if page_count > 100:
            scale_difficulty += 0.2
        elif page_count > 20:
            scale_difficulty += 0.1

        if image_count > page_count * 3:  # Many images per page
            scale_difficulty += 0.2
        elif image_count > page_count:
            scale_difficulty += 0.1

        total_difficulty = content_difficulty + scale_difficulty
        return min(total_difficulty, 1.0)

    def _apply_strategy_selection_logic(
        self, analysis: DocumentAnalysis
    ) -> ProcessingStrategy:
        """
        Apply strategy selection logic based on document analysis.

        Args:
            analysis: DocumentAnalysis results

        Returns:
            Selected ProcessingStrategy
        """
        # Strategy selection decision tree

        # 1. TEXT_ONLY strategy for pure text documents
        if (
            analysis.text_density >= self.text_density_threshold
            and analysis.image_ratio <= 0.1
            and analysis.layout_complexity <= 0.3
            and not analysis.has_complex_formatting
        ):

            self.logger.debug(
                "Selected TEXT_ONLY strategy",
                text_density=analysis.text_density,
                image_ratio=analysis.image_ratio,
                layout_complexity=analysis.layout_complexity,
            )
            return ProcessingStrategy.TEXT_ONLY

        # 2. IMAGE_HEAVY strategy for complex visual documents
        elif (
            analysis.image_ratio >= self.image_ratio_threshold
            or analysis.layout_complexity >= 0.7
            or analysis.estimated_processing_difficulty >= 0.8
        ):

            self.logger.debug(
                "Selected IMAGE_HEAVY strategy",
                image_ratio=analysis.image_ratio,
                layout_complexity=analysis.layout_complexity,
                processing_difficulty=analysis.estimated_processing_difficulty,
            )
            return ProcessingStrategy.IMAGE_HEAVY

        # 3. TEXT_OPTIMIZED for text-heavy with some images
        elif (
            analysis.text_density >= 0.5
            and analysis.image_ratio <= self.image_ratio_threshold
            and analysis.layout_complexity <= 0.6
        ):

            self.logger.debug(
                "Selected TEXT_OPTIMIZED strategy",
                text_density=analysis.text_density,
                image_ratio=analysis.image_ratio,
                layout_complexity=analysis.layout_complexity,
            )
            return ProcessingStrategy.TEXT_OPTIMIZED

        # 4. BALANCED for everything else
        else:
            self.logger.debug(
                "Selected BALANCED strategy (default)",
                text_density=analysis.text_density,
                image_ratio=analysis.image_ratio,
                layout_complexity=analysis.layout_complexity,
            )
            return ProcessingStrategy.BALANCED

    async def analyze_batch_strategy_distribution(
        self, document_paths: list[Path]
    ) -> Dict[ProcessingStrategy, int]:
        """
        Analyze strategy distribution for a batch of documents.

        Useful for cost estimation and batch optimization.

        Args:
            document_paths: List of document paths to analyze

        Returns:
            Dictionary mapping strategies to document counts
        """
        strategy_counts = {strategy: 0 for strategy in ProcessingStrategy}

        for doc_path in document_paths:
            try:
                strategy = await self.select_strategy(doc_path)
                strategy_counts[strategy] += 1
            except Exception as e:
                self.logger.warning(
                    "Failed to analyze document for batch distribution",
                    document=str(doc_path),
                    error=str(e),
                )
                # Default to balanced for failed analysis
                strategy_counts[ProcessingStrategy.BALANCED] += 1

        return strategy_counts

    async def estimate_batch_cost(self, document_paths: list[Path]) -> Dict[str, float]:
        """
        Estimate total processing cost for a batch of documents.

        Args:
            document_paths: List of document paths

        Returns:
            Dictionary with cost breakdown by strategy and totals
        """
        strategy_distribution = await self.analyze_batch_strategy_distribution(
            document_paths
        )

        cost_breakdown = {}
        total_cost = 0.0
        total_pages = 0

        for strategy, doc_count in strategy_distribution.items():
            if doc_count == 0:
                continue

            # Estimate average pages per document (rough heuristic)
            avg_pages = 5  # Default estimate
            if strategy == ProcessingStrategy.TEXT_ONLY:
                avg_pages = 3
            elif strategy == ProcessingStrategy.IMAGE_HEAVY:
                avg_pages = 10

            strategy_pages = doc_count * avg_pages
            strategy_cost = StrategySettings.estimate_processing_cost(
                strategy, strategy_pages
            )

            cost_breakdown[strategy.value] = {
                "document_count": doc_count,
                "estimated_pages": strategy_pages,
                "estimated_cost_usd": strategy_cost,
            }

            total_cost += strategy_cost
            total_pages += strategy_pages

        # Compare with Azure Document Intelligence costs
        azure_doc_intel_cost = total_pages * 0.1  # Estimated $0.10 per page
        estimated_savings = azure_doc_intel_cost - total_cost
        savings_percentage = (
            (estimated_savings / azure_doc_intel_cost * 100)
            if azure_doc_intel_cost > 0
            else 0
        )

        return {
            "strategy_breakdown": cost_breakdown,
            "total_estimated_cost_usd": total_cost,
            "total_estimated_pages": total_pages,
            "azure_doc_intel_estimated_cost_usd": azure_doc_intel_cost,
            "estimated_savings_usd": estimated_savings,
            "estimated_savings_percentage": savings_percentage,
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_strategy_selector():
        selector = DocumentStrategySelector()

        # Test with a sample document path
        test_path = Path("sample.pdf")  # Replace with actual path for testing

        if test_path.exists():
            strategy = await selector.select_strategy(test_path)
            print(f"Selected strategy for {test_path}: {strategy}")
        else:
            print("Test document not found")

    # asyncio.run(test_strategy_selector())
