"""
Text processing module for cost-effective document preprocessing.

This module implements multiple text extraction strategies using PyMuPDF, PDFPlumber,
and pymupdf4llm libraries while preserving visual citation coordinates required for
app compatibility. Each strategy is optimized for different document types and cost requirements.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import structlog
from dataclasses import dataclass
import fitz  # PyMuPDF
import pdfplumber
import pymupdf4llm
from docx import Document  # python-docx for Word documents
import zipfile
import tempfile
import json

from config.settings import get_settings, ProcessingStrategy


@dataclass
class TextChunk:
    """Text chunk with coordinates for visual citations."""

    text: str
    page_number: int
    bounding_box: List[float]  # [x1, y1, x2, y2]
    chunk_id: str
    char_start: int
    char_end: int


@dataclass
class LocationMetadata:
    """Location metadata for visual citations compatibility."""

    page_number: int
    bounding_polygons: List[List[float]]  # List of polygon coordinates
    ref_id: str


class TextProcessor:
    """
    Comprehensive text processor supporting multiple extraction strategies.

    Implements PyMuPDF (fast), PDFPlumber (precise), and pymupdf4llm (LLM-optimized)
    strategies while preserving visual citation coordinates for app compatibility.
    """

    def __init__(self):
        """Initialize the text processor with configuration."""
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)

        # Chunk size configuration for visual citations
        self.max_chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 100  # Character overlap between chunks

    async def extract_content(
        self,
        document_path: Path,
        strategy: Optional[ProcessingStrategy] = None,
        enable_layout_analysis: bool = True,
        preserve_coordinates: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract text content from document using specified strategy.

        Args:
            document_path: Path to the document
            strategy: Processing strategy to use
            enable_layout_analysis: Whether to analyze document layout
            preserve_coordinates: Whether to preserve coordinates for visual citations

        Returns:
            Dictionary containing extracted text, chunks, and metadata
        """
        self.logger.info(
            "Starting text extraction",
            document=str(document_path),
            strategy=strategy,
            layout_analysis=enable_layout_analysis,
            preserve_coordinates=preserve_coordinates,
        )

        try:
            file_extension = document_path.suffix.lower()

            if file_extension == ".pdf":
                return await self._extract_pdf_content(
                    document_path,
                    strategy,
                    enable_layout_analysis,
                    preserve_coordinates,
                )
            elif file_extension in [".docx", ".doc"]:
                return await self._extract_word_content(
                    document_path, preserve_coordinates
                )
            elif file_extension in [".txt", ".md"]:
                return await self._extract_text_file_content(
                    document_path, preserve_coordinates
                )
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

        except Exception as e:
            self.logger.error(
                "Text extraction failed",
                document=str(document_path),
                error=str(e),
                exc_info=True,
            )
            raise

    async def _extract_pdf_content(
        self,
        document_path: Path,
        strategy: Optional[ProcessingStrategy],
        enable_layout_analysis: bool,
        preserve_coordinates: bool,
    ) -> Dict[str, Any]:
        """Extract content from PDF using strategy-specific approach."""

        if strategy == ProcessingStrategy.TEXT_ONLY:
            return await self._extract_pymupdf_fast(document_path, preserve_coordinates)
        elif strategy == ProcessingStrategy.TEXT_OPTIMIZED:
            return await self._extract_pdfplumber_precise(
                document_path, preserve_coordinates
            )
        elif strategy == ProcessingStrategy.IMAGE_HEAVY:
            return await self._extract_pymupdf4llm_optimized(
                document_path, preserve_coordinates
            )
        elif strategy == ProcessingStrategy.BALANCED:
            return await self._extract_hybrid_approach(
                document_path, preserve_coordinates
            )
        else:
            # Default to PyMuPDF for unknown strategies
            return await self._extract_pymupdf_fast(document_path, preserve_coordinates)

    async def _extract_pymupdf_fast(
        self, document_path: Path, preserve_coordinates: bool
    ) -> Dict[str, Any]:
        """
        Fast text extraction using PyMuPDF with minimal processing.

        Optimized for text-only documents where speed and cost are priorities.
        """
        self.logger.debug("Using PyMuPDF fast extraction", document=str(document_path))

        doc = fitz.open(str(document_path))
        extracted_text = ""
        chunks = []
        location_metadata = {}

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]

                if preserve_coordinates:
                    # Extract text with coordinates
                    blocks = page.get_text("dict")["blocks"]
                    page_text = ""

                    for block_idx, block in enumerate(blocks):
                        if "lines" in block:  # Text block
                            block_text = ""
                            block_bbox = block["bbox"]

                            for line in block["lines"]:
                                for span in line["spans"]:
                                    block_text += span["text"]

                            if block_text.strip():
                                page_text += block_text + " "

                                # Create chunk with coordinates
                                chunk_id = f"page_{page_num + 1}_block_{block_idx}"
                                chunk = TextChunk(
                                    text=block_text.strip(),
                                    page_number=page_num + 1,
                                    bounding_box=list(block_bbox),
                                    chunk_id=chunk_id,
                                    char_start=len(extracted_text),
                                    char_end=len(extracted_text)
                                    + len(block_text.strip()),
                                )
                                chunks.append(chunk)

                                # Create location metadata for visual citations
                                location_metadata[chunk_id] = LocationMetadata(
                                    page_number=page_num + 1,
                                    bounding_polygons=[
                                        [
                                            block_bbox[0],
                                            block_bbox[1],  # top-left
                                            block_bbox[2],
                                            block_bbox[1],  # top-right
                                            block_bbox[2],
                                            block_bbox[3],  # bottom-right
                                            block_bbox[0],
                                            block_bbox[3],  # bottom-left
                                        ]
                                    ],
                                    ref_id=chunk_id,
                                )

                    extracted_text += page_text + "\n"
                else:
                    # Simple text extraction without coordinates
                    page_text = page.get_text()
                    extracted_text += page_text + "\n"

            doc.close()

            # If no coordinates preserved, create simple chunks
            if not preserve_coordinates:
                chunks = self._create_simple_chunks(extracted_text)

            return {
                "text": extracted_text.strip(),
                "chunks": [self._chunk_to_dict(chunk) for chunk in chunks],
                "location_metadata": {
                    k: self._location_to_dict(v) for k, v in location_metadata.items()
                },
                "page_count": len(doc),
                "processing_method": "pymupdf_fast",
                "coordinates_preserved": preserve_coordinates,
            }

        except Exception as e:
            doc.close()
            raise

    async def _extract_pdfplumber_precise(
        self, document_path: Path, preserve_coordinates: bool
    ) -> Dict[str, Any]:
        """
        Precise text extraction using PDFPlumber with enhanced layout analysis.

        Optimized for documents with tables and complex layouts.
        """
        self.logger.debug(
            "Using PDFPlumber precise extraction", document=str(document_path)
        )

        extracted_text = ""
        chunks = []
        location_metadata = {}

        with pdfplumber.open(str(document_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):

                if preserve_coordinates:
                    # Extract text with precise coordinates
                    chars = page.chars
                    words = page.extract_words()

                    # Group words into logical text blocks
                    text_blocks = self._group_words_into_blocks(words)

                    page_text = ""
                    for block_idx, block in enumerate(text_blocks):
                        block_text = " ".join([word["text"] for word in block])
                        page_text += block_text + " "

                        if block_text.strip():
                            # Calculate bounding box for block
                            x0 = min(word["x0"] for word in block)
                            y0 = min(word["top"] for word in block)
                            x1 = max(word["x1"] for word in block)
                            y1 = max(word["bottom"] for word in block)

                            chunk_id = f"page_{page_num + 1}_block_{block_idx}"
                            chunk = TextChunk(
                                text=block_text.strip(),
                                page_number=page_num + 1,
                                bounding_box=[x0, y0, x1, y1],
                                chunk_id=chunk_id,
                                char_start=len(extracted_text),
                                char_end=len(extracted_text) + len(block_text.strip()),
                            )
                            chunks.append(chunk)

                            # Create location metadata
                            location_metadata[chunk_id] = LocationMetadata(
                                page_number=page_num + 1,
                                bounding_polygons=[[x0, y0, x1, y0, x1, y1, x0, y1]],
                                ref_id=chunk_id,
                            )

                    extracted_text += page_text + "\n"

                    # Extract tables separately
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_text = self._format_table_as_text(table)
                            if table_text.strip():
                                chunk_id = f"page_{page_num + 1}_table_{table_idx}"
                                # Tables typically span the page width
                                table_bbox = page.bbox

                                chunk = TextChunk(
                                    text=table_text,
                                    page_number=page_num + 1,
                                    bounding_box=list(table_bbox),
                                    chunk_id=chunk_id,
                                    char_start=len(extracted_text),
                                    char_end=len(extracted_text) + len(table_text),
                                )
                                chunks.append(chunk)

                                extracted_text += "\n" + table_text + "\n"
                else:
                    # Simple text extraction
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"

            # If no coordinates preserved, create simple chunks
            if not preserve_coordinates:
                chunks = self._create_simple_chunks(extracted_text)

            return {
                "text": extracted_text.strip(),
                "chunks": [self._chunk_to_dict(chunk) for chunk in chunks],
                "location_metadata": {
                    k: self._location_to_dict(v) for k, v in location_metadata.items()
                },
                "page_count": len(pdf.pages),
                "processing_method": "pdfplumber_precise",
                "coordinates_preserved": preserve_coordinates,
            }

    async def _extract_pymupdf4llm_optimized(
        self, document_path: Path, preserve_coordinates: bool
    ) -> Dict[str, Any]:
        """
        LLM-optimized text extraction using pymupdf4llm.

        Optimized for documents that will be processed by LLMs, with structured
        markdown output and enhanced content organization.
        """
        self.logger.debug(
            "Using pymupdf4llm optimized extraction", document=str(document_path)
        )

        # Extract markdown-formatted text optimized for LLMs
        md_text = pymupdf4llm.to_markdown(str(document_path))

        # Also get coordinate information using standard PyMuPDF
        doc = fitz.open(str(document_path))
        chunks = []
        location_metadata = {}

        try:
            if preserve_coordinates:
                # Process with coordinates using PyMuPDF
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    blocks = page.get_text("dict")["blocks"]

                    for block_idx, block in enumerate(blocks):
                        if "lines" in block:  # Text block
                            block_text = ""
                            block_bbox = block["bbox"]

                            for line in block["lines"]:
                                for span in line["spans"]:
                                    block_text += span["text"]

                            if block_text.strip():
                                chunk_id = f"page_{page_num + 1}_block_{block_idx}"
                                chunk = TextChunk(
                                    text=block_text.strip(),
                                    page_number=page_num + 1,
                                    bounding_box=list(block_bbox),
                                    chunk_id=chunk_id,
                                    char_start=0,  # Will be recalculated for markdown
                                    char_end=len(block_text.strip()),
                                )
                                chunks.append(chunk)

                                location_metadata[chunk_id] = LocationMetadata(
                                    page_number=page_num + 1,
                                    bounding_polygons=[
                                        [
                                            block_bbox[0],
                                            block_bbox[1],
                                            block_bbox[2],
                                            block_bbox[1],
                                            block_bbox[2],
                                            block_bbox[3],
                                            block_bbox[0],
                                            block_bbox[3],
                                        ]
                                    ],
                                    ref_id=chunk_id,
                                )
            else:
                # Create simple chunks from markdown
                chunks = self._create_simple_chunks(md_text)

            doc.close()

            return {
                "text": md_text,
                "chunks": [self._chunk_to_dict(chunk) for chunk in chunks],
                "location_metadata": {
                    k: self._location_to_dict(v) for k, v in location_metadata.items()
                },
                "page_count": len(doc),
                "processing_method": "pymupdf4llm_optimized",
                "coordinates_preserved": preserve_coordinates,
                "content_format": "markdown",
            }

        except Exception as e:
            doc.close()
            raise

    async def _extract_hybrid_approach(
        self, document_path: Path, preserve_coordinates: bool
    ) -> Dict[str, Any]:
        """
        Hybrid extraction combining PyMuPDF speed with PDFPlumber precision.

        Uses PyMuPDF for basic text and PDFPlumber for tables and complex elements.
        """
        self.logger.debug(
            "Using hybrid extraction approach", document=str(document_path)
        )

        # First pass: Fast text extraction with PyMuPDF
        pymupdf_result = await self._extract_pymupdf_fast(
            document_path, preserve_coordinates
        )

        # Second pass: Extract tables with PDFPlumber
        additional_content = ""
        additional_chunks = []

        with pdfplumber.open(str(document_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract only tables for hybrid approach
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table:
                        table_text = self._format_table_as_text(table)
                        if table_text.strip():
                            additional_content += (
                                f"\n[Table from page {page_num + 1}]\n{table_text}\n"
                            )

                            if preserve_coordinates:
                                chunk_id = (
                                    f"page_{page_num + 1}_table_{table_idx}_hybrid"
                                )
                                table_bbox = page.bbox

                                chunk = TextChunk(
                                    text=table_text,
                                    page_number=page_num + 1,
                                    bounding_box=list(table_bbox),
                                    chunk_id=chunk_id,
                                    char_start=len(pymupdf_result["text"]),
                                    char_end=len(pymupdf_result["text"])
                                    + len(table_text),
                                )
                                additional_chunks.append(chunk)

        # Combine results
        combined_text = pymupdf_result["text"] + additional_content
        combined_chunks = pymupdf_result["chunks"] + [
            self._chunk_to_dict(chunk) for chunk in additional_chunks
        ]

        return {
            "text": combined_text,
            "chunks": combined_chunks,
            "location_metadata": pymupdf_result["location_metadata"],
            "page_count": pymupdf_result["page_count"],
            "processing_method": "hybrid_pymupdf_pdfplumber",
            "coordinates_preserved": preserve_coordinates,
        }

    async def _extract_word_content(
        self, document_path: Path, preserve_coordinates: bool
    ) -> Dict[str, Any]:
        """Extract content from Word documents (.docx, .doc)."""
        self.logger.debug(
            "Extracting Word document content", document=str(document_path)
        )

        try:
            doc = Document(str(document_path))
            extracted_text = ""
            chunks = []

            for para_idx, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    extracted_text += paragraph.text + "\n"

                    if preserve_coordinates:
                        # Word documents don't have precise coordinates, use paragraph index
                        chunk_id = f"paragraph_{para_idx}"
                        chunk = TextChunk(
                            text=paragraph.text,
                            page_number=1,  # Word documents don't have pages in the same way
                            bounding_box=[
                                0,
                                para_idx * 20,
                                600,
                                (para_idx + 1) * 20,
                            ],  # Estimated
                            chunk_id=chunk_id,
                            char_start=len(extracted_text) - len(paragraph.text) - 1,
                            char_end=len(extracted_text) - 1,
                        )
                        chunks.append(chunk)

            if not preserve_coordinates:
                chunks = self._create_simple_chunks(extracted_text)

            return {
                "text": extracted_text.strip(),
                "chunks": [self._chunk_to_dict(chunk) for chunk in chunks],
                "location_metadata": {},
                "page_count": 1,
                "processing_method": "python_docx",
                "coordinates_preserved": preserve_coordinates,
            }

        except Exception as e:
            self.logger.error("Failed to extract Word document", error=str(e))
            raise

    async def _extract_text_file_content(
        self, document_path: Path, preserve_coordinates: bool
    ) -> Dict[str, Any]:
        """Extract content from plain text files (.txt, .md)."""
        self.logger.debug("Extracting text file content", document=str(document_path))

        try:
            with open(document_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self._create_simple_chunks(content)

            return {
                "text": content,
                "chunks": [self._chunk_to_dict(chunk) for chunk in chunks],
                "location_metadata": {},
                "page_count": 1,
                "processing_method": "plain_text",
                "coordinates_preserved": False,
            }

        except Exception as e:
            self.logger.error("Failed to extract text file", error=str(e))
            raise

    def _group_words_into_blocks(self, words: List[Dict]) -> List[List[Dict]]:
        """Group words into logical text blocks based on proximity."""
        if not words:
            return []

        blocks = []
        current_block = [words[0]]

        for word in words[1:]:
            # Check if word should be in the same block
            last_word = current_block[-1]

            # Simple heuristic: same line or close vertical distance
            if (
                abs(word["top"] - last_word["top"]) < 5
                or abs(word["bottom"] - last_word["bottom"]) < 5
            ):
                current_block.append(word)
            else:
                # Start new block
                blocks.append(current_block)
                current_block = [word]

        if current_block:
            blocks.append(current_block)

        return blocks

    def _format_table_as_text(self, table: List[List[str]]) -> str:
        """Format extracted table as readable text."""
        if not table:
            return ""

        # Simple table formatting
        formatted_rows = []
        for row in table:
            if row:  # Skip empty rows
                formatted_row = " | ".join(cell or "" for cell in row)
                formatted_rows.append(formatted_row)

        return "\n".join(formatted_rows)

    def _create_simple_chunks(self, text: str) -> List[TextChunk]:
        """Create simple text chunks without coordinate information."""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))

            # Try to break at word boundary
            if end < len(text):
                while end > start and text[end] not in [" ", "\n", ".", "!", "?"]:
                    end -= 1
                if end == start:  # Couldn't find good break point
                    end = start + self.max_chunk_size

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    page_number=1,
                    bounding_box=[0, 0, 100, 100],  # Placeholder coordinates
                    chunk_id=f"chunk_{chunk_id}",
                    char_start=start,
                    char_end=end,
                )
                chunks.append(chunk)
                chunk_id += 1

            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def _chunk_to_dict(self, chunk: TextChunk) -> Dict[str, Any]:
        """Convert TextChunk to dictionary."""
        return {
            "text": chunk.text,
            "page_number": chunk.page_number,
            "bounding_box": chunk.bounding_box,
            "chunk_id": chunk.chunk_id,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        }

    def _location_to_dict(self, location: LocationMetadata) -> Dict[str, Any]:
        """Convert LocationMetadata to dictionary."""
        return {
            "page_number": location.page_number,
            "bounding_polygons": location.bounding_polygons,
            "ref_id": location.ref_id,
        }
