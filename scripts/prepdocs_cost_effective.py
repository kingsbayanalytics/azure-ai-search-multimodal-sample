#!/usr/bin/env python3
"""
Cost-Effective Document Processing Script with Local Embeddings

This script implements the Section 2.2 approach from the README - using PyMuPDF,
PDFPlumber, and LOCAL EMBEDDINGS instead of Azure Document Intelligence to achieve
70-85% cost savings while maintaining visual citation compatibility.

Now uses intfloat/e5-mistral-7b-instruct for highest quality local embeddings.
Added SPEED OPTIMIZATIONS with processing modes.

Usage:
    python scripts/prepdocs_cost_effective.py --input-path data/books/ --index-name cost-effective-books --mode speed
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

# Memory optimization for MPS (Apple Silicon)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable memory limit
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

# Load environment variables from .env files
try:
    from dotenv import load_dotenv

    # Try to load from several potential locations
    env_loaded = False

    # First try project root .env (if it exists)
    project_root = Path(__file__).parent.parent
    root_env = project_root / ".env"
    if root_env.exists():
        load_dotenv(root_env)
        print(f"✅ Loaded environment from {root_env}")
        env_loaded = True

    # Next try Azure environment .env
    azure_env = project_root / ".azure" / "my-multimodal-env" / ".env"
    if azure_env.exists() and not env_loaded:
        load_dotenv(azure_env)
        print(f"✅ Loaded environment from {azure_env}")
        env_loaded = True

    # Finally try backend .env
    backend_env = project_root / "src" / "backend" / ".env"
    if backend_env.exists() and not env_loaded:
        load_dotenv(backend_env)
        print(f"✅ Loaded environment from {backend_env}")
        env_loaded = True

    if not env_loaded:
        print("⚠️ No .env file found. Using existing environment variables.")
except ImportError:
    print("⚠️ python-dotenv not installed. Using existing environment variables.")

# Core document processing libraries (cost-effective alternatives)
try:
    import fitz  # PyMuPDF - free alternative to Azure Document Intelligence
    import pdfplumber  # Free precise text positioning
    import pymupdf4llm  # Free LLM-optimized markdown output
    from sentence_transformers import SentenceTransformer  # Local embeddings
    import torch
    from tqdm import tqdm  # For progress bars
except ImportError as e:
    print(
        f"❌ Missing required libraries. Please install: pip install PyMuPDF pdfplumber pymupdf4llm sentence-transformers torch tqdm python-dotenv"
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

    def __init__(self, mode="quality"):
        """
        Initialize with Azure environment variables and local embedding model.

        Args:
            mode: "quality" for best results, "speed" for faster processing
        """
        self.mode = mode
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

        # Initialize local embedding model based on mode
        if self.mode == "speed":
            model_name = "all-MiniLM-L6-v2"  # Faster but lower quality (384 dimensions)
            logger.info(f"🚀 SPEED MODE: Loading faster model: {model_name}")
            logger.info("📥 This smaller model is much faster (~80MB)")
            self.embedding_dimensions = 384
        else:
            model_name = (
                "intfloat/e5-mistral-7b-instruct"  # Highest quality (4096 dimensions)
            )
            logger.info(f"🧠 QUALITY MODE: Loading high-quality model: {model_name}")
            logger.info(
                "📥 This may take a few minutes for first-time download (~14GB)"
            )
            self.embedding_dimensions = 4096

        # Optimize for Apple Silicon if available
        if torch.backends.mps.is_available():
            logger.info("🍎 Using Apple Silicon (MPS) acceleration")
            device = "mps"
        elif torch.cuda.is_available():
            logger.info("🎮 Using CUDA GPU acceleration")
            device = "cuda"
        else:
            logger.info("💻 Using CPU for processing")
            device = "cpu"

        self.device = device
        self.embedding_model = SentenceTransformer(model_name, device=device)

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
            f"🚀 Starting cost-effective document processing with LOCAL EMBEDDINGS ({self.mode.upper()} MODE)"
        )
        logger.info(f"📁 Input: {input_path}")
        logger.info(f"🔍 Index: {index_name}")
        logger.info(f"⚙️  Strategy: {strategy}")
        logger.info(
            f"🤖 Device: {self.device}, Embedding dimensions: {self.embedding_dimensions}"
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
        print(f"\n⏳ Processing {len(documents)} documents...")
        for i, doc_path in enumerate(tqdm(documents, desc="Documents", unit="doc")):
            try:
                chunks = await self._process_single_document(doc_path, strategy)
                all_chunks.extend(chunks)
                self.stats["documents_processed"] += 1
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
                .replace("&", "and")  # Replace & with 'and'
                .replace("'", "")  # Remove apostrophes
                .replace("+", "plus")  # Replace + with 'plus'
                .replace(":", "_")  # Replace : with underscore
                .replace(";", "_")  # Replace ; with underscore
                .replace("/", "_")  # Replace / with underscore
                .replace("\\", "_")  # Replace \ with underscore
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
        """Generate embeddings using local model with memory optimization - NO API CALLS!"""
        import gc
        import torch

        try:
            # Choose batch size based on mode
            if self.mode == "speed":
                batch_size = 16  # Much larger batch size for speed
            else:
                batch_size = 4  # Still larger than original but maintain quality

            # Process in batches with progress bar
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            # Only show progress bar for larger document sets
            if len(texts) > 10:
                batch_iterator = tqdm(
                    range(0, len(texts), batch_size),
                    desc="Generating embeddings",
                    total=total_batches,
                    unit="batch",
                )
            else:
                batch_iterator = range(0, len(texts), batch_size)

            for i in batch_iterator:
                batch_texts = texts[i : i + batch_size]

                try:
                    # Generate embeddings for this batch with optimized settings
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        device=self.device,
                    )

                    # Convert to list and add to results
                    batch_embeddings_list = [
                        embedding.tolist() for embedding in batch_embeddings
                    ]
                    all_embeddings.extend(batch_embeddings_list)

                    # Memory cleanup - only do garbage collection every few batches in speed mode
                    if self.mode != "speed" or i % (batch_size * 4) == 0:
                        del batch_embeddings
                        del batch_embeddings_list
                        gc.collect()
                        if self.device == "mps":
                            torch.mps.empty_cache()
                        elif self.device == "cuda":
                            torch.cuda.empty_cache()

                except Exception as batch_error:
                    logger.error(f"Batch embedding failed: {batch_error}")
                    # Add default embeddings for failed batch
                    default_batch = [
                        [0.0] * self.embedding_dimensions for _ in batch_texts
                    ]
                    all_embeddings.extend(default_batch)

            return all_embeddings

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
                vector_search_dimensions=self.embedding_dimensions,
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

        # Upload in batches with progress bar
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        print(f"\n📤 Uploading {len(chunks)} chunks to search index...")
        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc="Uploading batches",
            total=total_batches,
            unit="batch",
        ):
            batch = chunks[i : i + batch_size]
            try:
                result = search_client.upload_documents(documents=batch)
                if result[0].succeeded is False:
                    logger.error(f"❌ Upload failed: {result[0].error_message}")
            except Exception as e:
                logger.error(f"❌ Upload failed: {e}")

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
                "mode": self.mode,
                "embedding_model": str(self.embedding_model),
                "embedding_dimensions": self.embedding_dimensions,
                "device": self.device,
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
            "strategy": f"cost_effective_local_processing_{self.mode}_mode",
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
  python scripts/prepdocs_cost_effective.py --input-path data/books/ --index-name fast-embeddings-books --mode speed

Processing Modes:
  - quality: Uses intfloat/e5-mistral-7b-instruct (4096 dimensions) for best results (default)
  - speed: Uses all-MiniLM-L6-v2 (384 dimensions) for much faster processing
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
    parser.add_argument(
        "--mode",
        choices=["quality", "speed"],
        default="quality",
        help="Processing mode (quality=best results, speed=faster processing)",
    )
    args = parser.parse_args()

    try:
        processor = CostEffectiveDocumentProcessor(mode=args.mode)

        result = await processor.process_documents(
            input_path=args.input_path,
            index_name=args.index_name,
            strategy=args.strategy,
        )

        # Display results
        print("\n" + "=" * 60)
        print(
            f"🎉 COST-EFFECTIVE PROCESSING WITH LOCAL EMBEDDINGS COMPLETE! ({args.mode.upper()} MODE)"
        )
        print("=" * 60)

        summary = result["processing_summary"]
        cost_analysis = result["cost_analysis"]

        print(f"✅ Documents processed: {summary['documents_processed']}")
        print(f"📄 Total pages: {summary['total_pages']}")
        print(f"⏱️  Processing time: {summary['processing_time_seconds']:.1f} seconds")
        print(f"💻 Processing device: {summary['device']}")
        print(f"🤖 Embedding dimensions: {summary['embedding_dimensions']}")

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
