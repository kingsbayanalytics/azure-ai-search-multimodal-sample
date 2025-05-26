#!/usr/bin/env python3
"""
Cost-Effective Document Processing Script with Local Embeddings

This script implements the Section 2.2 approach from the README - using PyMuPDF,
PDFPlumber, and LOCAL EMBEDDINGS instead of Azure Document Intelligence to achieve
70-85% cost savings while maintaining visual citation compatibility.

Now uses intfloat/e5-mistral-7b-instruct for highest quality local embeddings.

Usage:
    python scripts/prepdocs_cost_effective.py --input-path data/books/ --index-name cost-effective-books
"""

import asyncio
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import time

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

# Core document processing libraries (cost-effective alternatives)
try:
    import fitz  # PyMuPDF - free alternative to Azure Document Intelligence
    import pdfplumber  # Free precise text positioning
    import pymupdf4llm  # Free LLM-optimized markdown output
    from sentence_transformers import SentenceTransformer  # Local embeddings
    import torch
except ImportError as e:
    print(
        f"❌ Missing required libraries. Please install: pip install PyMuPDF pdfplumber pymupdf4llm sentence-transformers torch"
    )
    sys.exit(1)

# Azure services (only what we need - no OpenAI for embeddings!)
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ComplexField,
)
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CostEffectiveDocumentProcessor:
    """
    Cost-effective document processor using local libraries and LOCAL EMBEDDINGS.

    Achieves 70-85% cost savings by:
    - Using PyMuPDF instead of Azure Document Intelligence ($0 vs $1.50/1K pages)
    - Using LOCAL embeddings (e5-mistral-7b-instruct) instead of Azure OpenAI ($0 vs $0.00013/1K tokens)
    - Selective use of GPT-4o Vision only for complex images (future enhancement)
    """

    def __init__(self):
        """Initialize with Azure environment variables and local embedding model."""
        # Get Azure configuration from environment (set by azd)
        self.search_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT")

        if not self.search_endpoint:
            raise ValueError(
                "Missing SEARCH_SERVICE_ENDPOINT environment variable. Run 'azd env get-values' to check configuration."
            )

        # Initialize Azure clients (no OpenAI needed!)
        self.credential = DefaultAzureCredential()
        self.search_index_client = SearchIndexClient(
            endpoint=self.search_endpoint, credential=self.credential
        )

        # Initialize local embedding model
        logger.info("🤖 Loading local embedding model: intfloat/e5-mistral-7b-instruct")
        logger.info("📥 This may take a few minutes for first-time download (~14GB)")

        self.embedding_model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
        self.embedding_dimensions = 4096  # e5-mistral-7b-instruct dimensions

        logger.info(
            f"✅ Local embedding model loaded! Dimensions: {self.embedding_dimensions}"
        )

        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "total_pages": 0,
            "total_cost": 0.0,  # Should stay at $0 with local embeddings!
            "azure_doc_intel_cost_saved": 0.0,
            "azure_openai_embedding_cost_saved": 0.0,
            "processing_time": 0.0,
        }

    async def process_documents(
        self, input_path: str, index_name: str, strategy: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Process documents using cost-effective local processing with LOCAL EMBEDDINGS.

        Args:
            input_path: Path to documents (file or directory)
            index_name: Name for the search index
            strategy: Processing strategy ("text_only", "balanced", "image_heavy")

        Returns:
            Processing results and cost analysis
        """
        start_time = datetime.now()

        logger.info(
            f"🚀 Starting cost-effective document processing with LOCAL EMBEDDINGS"
        )
        logger.info(f"📁 Input: {input_path}")
        logger.info(f"🔍 Index: {index_name}")
        logger.info(f"⚙️  Strategy: {strategy}")
        logger.info(
            f"🤖 Embedding model: e5-mistral-7b-instruct ({self.embedding_dimensions}D)"
        )

        # Discover documents
        documents = self._discover_documents(input_path)
        logger.info(f"📄 Found {len(documents)} documents")

        if not documents:
            raise ValueError(f"No supported documents found in {input_path}")

        # Create search index with correct dimensions
        await self._create_search_index(index_name)

        # Process documents
        all_chunks = []
        for doc_path in documents:
            try:
                chunks = await self._process_single_document(doc_path, strategy)
                all_chunks.extend(chunks)
                self.stats["documents_processed"] += 1
                logger.info(f"✅ Processed: {doc_path.name}")
            except Exception as e:
                logger.error(f"❌ Failed to process {doc_path.name}: {e}")

        # Upload to search index
        if all_chunks:
            await self._upload_to_search_index(index_name, all_chunks)

        # Calculate final statistics
        end_time = datetime.now()
        self.stats["processing_time"] = (end_time - start_time).total_seconds()
        self.stats["azure_doc_intel_cost_saved"] = (
            self.stats["total_pages"] * 0.0015
        )  # $1.50 per 1K pages

        # Calculate embedding cost savings (rough estimate)
        estimated_tokens = self.stats["total_pages"] * 500  # ~500 tokens per page
        self.stats["azure_openai_embedding_cost_saved"] = (
            estimated_tokens / 1000 * 0.00013
        )  # $0.00013 per 1K tokens

        return self._generate_report()

    def _discover_documents(self, input_path: str) -> List[Path]:
        """Discover PDF documents in the input path."""
        input_path_obj = Path(input_path)
        documents = []

        if input_path_obj.is_file() and input_path_obj.suffix.lower() == ".pdf":
            documents.append(input_path_obj)
        elif input_path_obj.is_dir():
            documents.extend(input_path_obj.rglob("*.pdf"))

        return sorted(documents)

    async def _process_single_document(
        self, doc_path: Path, strategy: str
    ) -> List[Dict[str, Any]]:
        """
        Process a single document using cost-effective local processing.

        This replaces Azure Document Intelligence with PyMuPDF/PDFPlumber.
        """
        logger.info(f"📖 Processing: {doc_path.name}")

        # Step 1: Extract text and layout using PyMuPDF (FREE vs $1.50/1K pages)
        doc_content = self._extract_content_with_pymupdf(doc_path)

        # Step 2: Enhanced table extraction with PDFPlumber if needed
        if strategy in ["balanced", "image_heavy"]:
            tables = self._extract_tables_with_pdfplumber(doc_path)
            doc_content["tables"] = tables

        # Step 3: Process images (selective approach for cost optimization)
        if strategy == "image_heavy":
            image_descriptions = await self._process_images_selective(doc_path)
            doc_content["image_descriptions"] = image_descriptions

        # Step 4: Create chunks with LOCAL embeddings
        chunks = await self._create_chunks_with_embeddings(doc_content, doc_path)

        self.stats["total_pages"] += doc_content["page_count"]

        return chunks

    def _extract_content_with_pymupdf(self, doc_path: Path) -> Dict[str, Any]:
        """
        Extract text and layout using PyMuPDF - FREE alternative to Azure Document Intelligence.

        Maintains visual citation coordinates for app compatibility.
        """
        doc = fitz.open(str(doc_path))

        content = {
            "text": "",
            "pages": [],
            "page_count": len(doc),
            "images": [],
            "location_metadata": {},
        }

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text with coordinates (preserves visual citations)
            text_dict = page.get_text("dict")
            page_text = page.get_text()

            # Store page information
            page_info = {
                "page_number": page_num + 1,
                "text": page_text,
                "bbox": list(page.rect),  # Convert to list for JSON serialization
                "text_blocks": [],
            }

            # Extract text blocks with coordinates for visual citations
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]

                    if block_text.strip():
                        page_info["text_blocks"].append(
                            {
                                "text": block_text,
                                "bbox": block["bbox"],
                                "page": page_num + 1,
                            }
                        )

            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    pix = fitz.Pixmap(doc, img[0])
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        content["images"].append(
                            {
                                "page": page_num + 1,
                                "image_index": img_index,
                                "data": img_data,
                                "bbox": page.get_image_bbox(img),
                            }
                        )
                    pix = None
                except:
                    pass  # Skip problematic images

            content["pages"].append(page_info)
            content["text"] += page_text + "\n\n"

        doc.close()
        return content

    def _extract_tables_with_pdfplumber(self, doc_path: Path) -> List[Dict[str, Any]]:
        """Extract tables using PDFPlumber for enhanced accuracy."""
        tables = []

        try:
            with pdfplumber.open(str(doc_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_index, table in enumerate(page_tables):
                        if table:  # Skip empty tables
                            # Convert table to text
                            table_text = "\n".join(
                                [
                                    "\t".join([cell or "" for cell in row])
                                    for row in table
                                ]
                            )
                            tables.append(
                                {
                                    "page": page_num + 1,
                                    "table_index": table_index,
                                    "text": table_text,
                                    "bbox": [0, 0, page.width, page.height],
                                }
                            )
        except Exception as e:
            logger.warning(f"Table extraction failed for {doc_path.name}: {e}")

        return tables

    async def _process_images_selective(self, doc_path: Path) -> List[Dict[str, Any]]:
        """
        Selective image processing - only use GPT-4o Vision for complex images.

        This is much more cost-effective than processing every image.
        """
        # For now, skip image processing to focus on text
        # In production, you'd implement selective image analysis here
        return []

    async def _create_chunks_with_embeddings(
        self, doc_content: Dict[str, Any], doc_path: Path
    ) -> List[Dict[str, Any]]:
        """Create text chunks with LOCAL embeddings for search indexing."""
        chunks = []

        # Collect all valid pages first
        valid_pages = [
            page_info for page_info in doc_content["pages"] if page_info["text"].strip()
        ]

        if not valid_pages:
            return chunks

        logger.info(f"🤖 Generating LOCAL embeddings for {len(valid_pages)} pages...")

        # Generate embeddings locally (no API calls!)
        texts = [page["text"] for page in valid_pages]
        embeddings = self._generate_local_embeddings(texts)

        # Create chunks with embeddings
        for i, page_info in enumerate(valid_pages):
            embedding = (
                embeddings[i]
                if i < len(embeddings)
                else [0.0] * self.embedding_dimensions
            )

            # Create chunk compatible with existing app (sanitize key for Azure Search)
            sanitized_filename = (
                doc_path.stem.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
            )
            chunk = {
                "id": f"{sanitized_filename}_page_{page_info['page_number']}",
                "content": page_info["text"],
                "title": doc_path.stem,
                "page_number": page_info["page_number"],
                "source_file": str(doc_path),
                "text_vector": embedding,
                "locationMetadata": json.dumps(
                    {
                        "page": page_info["page_number"],
                        "source": str(doc_path),
                        "bbox": page_info["bbox"],
                    }
                ),
                "boundingPolygons": json.dumps(
                    [
                        {
                            "page": page_info["page_number"],
                            "coordinates": page_info["bbox"],
                        }
                    ]
                ),
            }
            chunks.append(chunk)

        return chunks

    def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model - NO API CALLS!"""
        try:
            logger.info(f"🔄 Processing {len(texts)} texts with local model...")

            # Generate embeddings locally
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=8,  # Adjust based on your RAM
                show_progress_bar=True,
                convert_to_numpy=True,
            )

            # Convert to list format for JSON serialization
            embeddings_list = [embedding.tolist() for embedding in embeddings]

            logger.info(f"✅ Generated {len(embeddings_list)} local embeddings")
            return embeddings_list

        except Exception as e:
            logger.error(f"❌ Local embedding generation failed: {e}")
            # Return default embeddings if local generation fails
            logger.warning(f"⚠️  Using default embeddings for {len(texts)} texts")
            return [[0.0] * self.embedding_dimensions for _ in texts]

    async def _create_search_index(self, index_name: str):
        """Create Azure Search index compatible with the existing app and LOCAL EMBEDDINGS."""

        # Define fields compatible with the existing app
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(
                name="title", type=SearchFieldDataType.String, filterable=True
            ),
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="source_file", type=SearchFieldDataType.String, filterable=True
            ),
            SearchField(
                name="text_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=self.embedding_dimensions,  # 4096 for e5-mistral-7b-instruct
                vector_search_profile_name="text-vector-profile",
            ),
            SimpleField(name="locationMetadata", type=SearchFieldDataType.String),
            SimpleField(
                name="boundingPolygons",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=False,
                sortable=False,
                facetable=False,
            ),
        ]

        # Vector search configuration
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="text-vector-profile",
                    algorithm_configuration_name="hnsw-config",
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters=HnswParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE
                    ),
                )
            ],
        )

        # Create index
        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

        try:
            result = self.search_index_client.create_index(index)
            logger.info(
                f"✅ Created search index: {index_name} (dimensions: {self.embedding_dimensions})"
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"📋 Using existing index: {index_name}")
            else:
                raise

    async def _upload_to_search_index(
        self, index_name: str, chunks: List[Dict[str, Any]]
    ):
        """Upload processed chunks to Azure Search index."""
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.credential,
        )

        # Upload in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                result = search_client.upload_documents(documents=batch)
                logger.info(
                    f"📤 Uploaded batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}"
                )
            except Exception as e:
                logger.error(f"❌ Upload failed for batch {i // batch_size + 1}: {e}")

    def _generate_report(self) -> Dict[str, Any]:
        """Generate processing report with cost analysis."""
        total_savings = (
            self.stats["azure_doc_intel_cost_saved"]
            + self.stats["azure_openai_embedding_cost_saved"]
        )

        return {
            "processing_summary": {
                "documents_processed": self.stats["documents_processed"],
                "total_pages": self.stats["total_pages"],
                "processing_time_seconds": self.stats["processing_time"],
                "embedding_model": "intfloat/e5-mistral-7b-instruct",
                "embedding_dimensions": self.embedding_dimensions,
            },
            "cost_analysis": {
                "total_cost_usd": self.stats["total_cost"],  # Should be $0!
                "azure_doc_intel_cost_saved_usd": self.stats[
                    "azure_doc_intel_cost_saved"
                ],
                "azure_openai_embedding_cost_saved_usd": self.stats[
                    "azure_openai_embedding_cost_saved"
                ],
                "total_cost_savings_usd": total_savings,
                "cost_savings_percentage": (
                    (total_savings / (total_savings + self.stats["total_cost"]) * 100)
                    if total_savings > 0
                    else 0
                ),
                "cost_per_page": (
                    self.stats["total_cost"] / self.stats["total_pages"]
                    if self.stats["total_pages"] > 0
                    else 0
                ),
            },
            "strategy": "cost_effective_local_processing_with_local_embeddings",
        }


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Cost-Effective Document Processing with LOCAL EMBEDDINGS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepdocs_cost_effective.py --input-path data/books/ --index-name local-embeddings-books
  python scripts/prepdocs_cost_effective.py --input-path data/reports/report.pdf --index-name local-embeddings-reports --strategy image_heavy

Note: This version uses intfloat/e5-mistral-7b-instruct for local embeddings (4096 dimensions).
First run will download ~14GB model. Subsequent runs will be much faster.
        """,
    )

    parser.add_argument(
        "--input-path", required=True, help="Path to documents (file or directory)"
    )
    parser.add_argument("--index-name", required=True, help="Name for the search index")
    parser.add_argument(
        "--strategy",
        choices=["text_only", "balanced", "image_heavy"],
        default="balanced",
        help="Processing strategy",
    )
    args = parser.parse_args()

    try:
        processor = CostEffectiveDocumentProcessor()

        logger.info(
            "🤖 Using LOCAL embeddings (e5-mistral-7b-instruct) - NO API CALLS!"
        )

        result = await processor.process_documents(
            input_path=args.input_path,
            index_name=args.index_name,
            strategy=args.strategy,
        )

        # Display results
        print("\n" + "=" * 60)
        print("🎉 COST-EFFECTIVE PROCESSING WITH LOCAL EMBEDDINGS COMPLETE!")
        print("=" * 60)

        summary = result["processing_summary"]
        cost_analysis = result["cost_analysis"]

        print(f"✅ Documents processed: {summary['documents_processed']}")
        print(f"📄 Total pages: {summary['total_pages']}")
        print(f"⏱️  Processing time: {summary['processing_time_seconds']:.1f} seconds")
        print(f"🤖 Embedding model: {summary['embedding_model']}")
        print(f"📐 Embedding dimensions: {summary['embedding_dimensions']}")

        print(f"\n💰 COST ANALYSIS:")
        print(f"💵 Total cost: ${cost_analysis['total_cost_usd']:.4f} (LOCAL = FREE!)")
        print(
            f"💡 Azure Doc Intelligence cost saved: ${cost_analysis['azure_doc_intel_cost_saved_usd']:.2f}"
        )
        print(
            f"💡 Azure OpenAI embedding cost saved: ${cost_analysis['azure_openai_embedding_cost_saved_usd']:.4f}"
        )
        print(f"💰 Total cost savings: ${cost_analysis['total_cost_savings_usd']:.2f}")
        print(f"📊 Cost savings: {cost_analysis['cost_savings_percentage']:.1f}%")
        print(f"📈 Cost per page: ${cost_analysis['cost_per_page']:.6f}")

        print(f"\n🔍 Index created: {args.index_name}")
        print(f"🌐 You can now use this index in your app!")
        print(
            f"📝 Note: Update your app to use index '{args.index_name}' for local embeddings"
        )

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
