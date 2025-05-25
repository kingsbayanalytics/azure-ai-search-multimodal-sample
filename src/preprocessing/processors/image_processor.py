"""
Image processing module for cost-effective document preprocessing.

This module implements cost-optimized image processing using Mistral OCR for
cost-effective text extraction and selective GPT-4o Vision for complex images.
Preserves image metadata and coordinates for visual citations compatibility.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import structlog
from dataclasses import dataclass
import fitz  # PyMuPDF for image extraction
from PIL import Image, ImageEnhance
import io
import base64
import httpx
import openai
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from config.settings import get_settings, OCRProvider


@dataclass
class ImageData:
    """Image data with metadata for processing and citations."""

    image_id: str
    page_number: int
    image_bytes: bytes
    image_format: str  # PNG, JPEG, etc.
    bounding_box: List[float]  # [x1, y1, x2, y2]
    width: int
    height: int
    dpi: Optional[int]
    file_size_bytes: int


@dataclass
class OCRResult:
    """OCR processing result with text and confidence."""

    text: str
    confidence: float
    provider: str
    processing_time_ms: int
    cost_usd: float
    image_id: str
    bounding_boxes: List[Dict[str, Any]]  # Text regions with coordinates


class ImageProcessor:
    """
    Cost-effective image processor with intelligent OCR strategy selection.

    Uses Mistral for cost-effective OCR and GPT-4o Vision for complex images,
    while preserving image metadata for visual citations.
    """

    def __init__(self):
        """Initialize the image processor with configuration and clients."""
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)

        # Initialize OCR clients
        self._mistral_client = None
        self._openai_client = None

        # Image processing configuration
        self.max_image_size = (2048, 2048)  # Max dimensions for OCR
        self.image_quality_threshold = 0.7  # Quality score for GPT-4o selection
        self.cost_tracking = {}

    async def extract_and_process_images(
        self,
        document_path: Path,
        ocr_provider: Optional[OCRProvider] = None,
        enable_advanced_vision: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract and process images from document with OCR.

        Args:
            document_path: Path to the document
            ocr_provider: OCR provider to use (Mistral, GPT-4o, or Hybrid)
            enable_advanced_vision: Whether to use advanced vision capabilities

        Returns:
            Dictionary containing processed images and extracted text
        """
        self.logger.info(
            "Starting image processing",
            document=str(document_path),
            ocr_provider=ocr_provider,
            advanced_vision=enable_advanced_vision,
        )

        try:
            # Extract images from document
            images = await self._extract_images_from_document(document_path)

            if not images:
                return {
                    "images": [],
                    "ocr_text": "",
                    "total_images": 0,
                    "processing_cost": 0.0,
                }

            # Process images with OCR
            processed_images = []
            total_ocr_text = ""
            total_cost = 0.0

            for image_data in images:
                # Analyze image complexity for provider selection
                if ocr_provider == OCRProvider.HYBRID:
                    selected_provider = await self._select_ocr_provider_for_image(
                        image_data
                    )
                else:
                    selected_provider = ocr_provider or OCRProvider.MISTRAL

                # Process image with selected provider
                ocr_result = await self._process_image_with_ocr(
                    image_data, selected_provider, enable_advanced_vision
                )

                # Aggregate results
                if ocr_result and ocr_result.text.strip():
                    total_ocr_text += f"\n[Image {image_data.image_id} from page {image_data.page_number}]\n"
                    total_ocr_text += ocr_result.text + "\n"
                    total_cost += ocr_result.cost_usd

                # Prepare image data for indexing
                processed_image = {
                    "image_id": image_data.image_id,
                    "page_number": image_data.page_number,
                    "bounding_box": image_data.bounding_box,
                    "width": image_data.width,
                    "height": image_data.height,
                    "format": image_data.image_format,
                    "file_size_bytes": image_data.file_size_bytes,
                    "ocr_text": ocr_result.text if ocr_result else "",
                    "ocr_confidence": ocr_result.confidence if ocr_result else 0.0,
                    "ocr_provider": ocr_result.provider if ocr_result else "none",
                    "processing_cost": ocr_result.cost_usd if ocr_result else 0.0,
                }
                processed_images.append(processed_image)

            self.logger.info(
                "Image processing completed",
                document=str(document_path),
                total_images=len(images),
                total_cost=total_cost,
                text_length=len(total_ocr_text),
            )

            return {
                "images": processed_images,
                "ocr_text": total_ocr_text.strip(),
                "total_images": len(images),
                "processing_cost": total_cost,
                "average_confidence": (
                    sum(img.get("ocr_confidence", 0) for img in processed_images)
                    / len(processed_images)
                    if processed_images
                    else 0.0
                ),
            }

        except Exception as e:
            self.logger.error(
                "Image processing failed",
                document=str(document_path),
                error=str(e),
                exc_info=True,
            )
            raise

    async def _extract_images_from_document(
        self, document_path: Path
    ) -> List[ImageData]:
        """Extract images from PDF document with metadata."""
        if document_path.suffix.lower() != ".pdf":
            # For non-PDF documents, no image extraction yet
            return []

        doc = fitz.open(str(document_path))
        images = []

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        # Skip if image is too small or not useful
                        if pix.width < 50 or pix.height < 50:
                            pix = None
                            continue

                        # Convert to PIL Image for processing
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_format = "PNG"
                        else:  # CMYK
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            img_format = "PNG"
                            pix1 = None

                        # Get image position on page
                        img_rects = page.get_image_rects(xref)
                        bounding_box = (
                            list(img_rects[0])
                            if img_rects
                            else [0, 0, pix.width, pix.height]
                        )

                        # Create image metadata
                        image_metadata = ImageData(
                            image_id=f"page_{page_num + 1}_img_{img_index}",
                            page_number=page_num + 1,
                            image_bytes=img_data,
                            image_format=img_format,
                            bounding_box=bounding_box,
                            width=pix.width,
                            height=pix.height,
                            dpi=pix.xres if hasattr(pix, "xres") else None,
                            file_size_bytes=len(img_data),
                        )

                        images.append(image_metadata)
                        pix = None

                    except Exception as e:
                        self.logger.warning(
                            "Failed to extract image",
                            page=page_num + 1,
                            image_index=img_index,
                            error=str(e),
                        )
                        continue

            doc.close()
            return images

        except Exception as e:
            doc.close()
            raise

    async def _select_ocr_provider_for_image(
        self, image_data: ImageData
    ) -> OCRProvider:
        """
        Intelligently select OCR provider based on image characteristics.

        Uses heuristics to determine if an image needs expensive GPT-4o Vision
        or can be processed with cost-effective Mistral OCR.
        """
        # Analyze image complexity
        complexity_score = await self._analyze_image_complexity(image_data)

        # Selection logic based on complexity
        if complexity_score > 0.8:
            # High complexity: charts, diagrams, complex layouts
            self.logger.debug(
                "Selected GPT-4o Vision for complex image",
                image_id=image_data.image_id,
                complexity=complexity_score,
            )
            return OCRProvider.GPT4O_VISION
        elif complexity_score > 0.5:
            # Medium complexity: mixed content
            # Use Mistral but consider upgrading based on results
            return OCRProvider.MISTRAL
        else:
            # Low complexity: simple text, clean documents
            return OCRProvider.MISTRAL

    async def _analyze_image_complexity(self, image_data: ImageData) -> float:
        """
        Analyze image complexity to guide OCR provider selection.

        Returns complexity score from 0.0 (simple) to 1.0 (very complex).
        """
        try:
            # Load image for analysis
            image = Image.open(io.BytesIO(image_data.image_bytes))

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Basic complexity analysis
            complexity_score = 0.0

            # Size factor (larger images might be more complex)
            size_factor = min(image_data.width * image_data.height / (1000 * 1000), 1.0)
            complexity_score += size_factor * 0.2

            # Color variance (more colors = more complex)
            colors = image.getcolors(maxcolors=256 * 256 * 256)
            if colors:
                color_count = len(colors)
                color_factor = min(color_count / 1000, 1.0)
                complexity_score += color_factor * 0.3
            else:
                complexity_score += 0.8  # Too many colors to count

            # Image entropy (measure of randomness/detail)
            import numpy as np

            img_array = np.array(image)

            # Calculate entropy for each channel
            entropy = 0
            for channel in range(img_array.shape[2]):
                hist = np.histogram(img_array[:, :, channel], bins=256)[0]
                hist = hist / hist.sum()  # Normalize
                entropy += -np.sum(hist * np.log2(hist + 1e-10))

            avg_entropy = entropy / img_array.shape[2]
            entropy_factor = min(
                avg_entropy / 8.0, 1.0
            )  # 8 is roughly max entropy for 8-bit
            complexity_score += entropy_factor * 0.5

            return min(complexity_score, 1.0)

        except Exception as e:
            self.logger.warning(
                "Failed to analyze image complexity",
                image_id=image_data.image_id,
                error=str(e),
            )
            # Return moderate complexity as fallback
            return 0.6

    async def _process_image_with_ocr(
        self,
        image_data: ImageData,
        provider: OCRProvider,
        enable_advanced_vision: bool = False,
    ) -> Optional[OCRResult]:
        """Process image with specified OCR provider."""
        start_time = asyncio.get_event_loop().time()

        try:
            if provider == OCRProvider.MISTRAL:
                result = await self._process_with_mistral_ocr(image_data)
            elif provider == OCRProvider.GPT4O_VISION:
                result = await self._process_with_gpt4o_vision(
                    image_data, enable_advanced_vision
                )
            else:
                raise ValueError(f"Unsupported OCR provider: {provider}")

            if result:
                processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
                result.processing_time_ms = int(processing_time)

                self.logger.debug(
                    "OCR processing completed",
                    image_id=image_data.image_id,
                    provider=provider,
                    confidence=result.confidence,
                    cost=result.cost_usd,
                    processing_time=processing_time,
                )

            return result

        except Exception as e:
            self.logger.error(
                "OCR processing failed",
                image_id=image_data.image_id,
                provider=provider,
                error=str(e),
            )
            return None

    async def _process_with_mistral_ocr(
        self, image_data: ImageData
    ) -> Optional[OCRResult]:
        """
        Process image with Mistral OCR for cost-effective text extraction.

        Note: This is a placeholder implementation. In practice, you would
        integrate with Mistral's actual OCR API when available.
        """
        try:
            # Initialize Mistral client if needed
            if not self._mistral_client:
                self._mistral_client = httpx.AsyncClient(
                    base_url=self.settings.mistral_endpoint,
                    headers={
                        "Authorization": f"Bearer {self.settings.mistral_api_key}"
                    },
                )

            # Prepare image for OCR
            processed_image = await self._preprocess_image_for_ocr(image_data)
            image_b64 = base64.b64encode(processed_image).decode("utf-8")

            # Placeholder OCR call (replace with actual Mistral OCR API)
            # For now, we'll simulate OCR using a simple approach
            extracted_text = await self._simulate_mistral_ocr(image_data)

            # Calculate cost (estimated)
            cost_per_image = 0.0002  # $0.0002 per image

            return OCRResult(
                text=extracted_text,
                confidence=0.85,  # Simulated confidence
                provider="mistral",
                processing_time_ms=0,  # Will be set by caller
                cost_usd=cost_per_image,
                image_id=image_data.image_id,
                bounding_boxes=[],  # Simplified for now
            )

        except Exception as e:
            self.logger.error(
                "Mistral OCR failed", image_id=image_data.image_id, error=str(e)
            )
            return None

    async def _process_with_gpt4o_vision(
        self, image_data: ImageData, enable_advanced_vision: bool = False
    ) -> Optional[OCRResult]:
        """Process image with GPT-4o Vision for high-quality analysis."""
        try:
            # Initialize OpenAI client if needed
            if not self._openai_client:
                self._openai_client = openai.AsyncAzureOpenAI(
                    api_key=self.settings.openai_api_key,
                    api_version=self.settings.openai_api_version,
                    azure_endpoint=self.settings.openai_endpoint,
                )

            # Prepare image
            processed_image = await self._preprocess_image_for_ocr(image_data)
            image_b64 = base64.b64encode(processed_image).decode("utf-8")

            # Create prompt based on processing mode
            if enable_advanced_vision:
                prompt = """
                Analyze this image comprehensively and extract all text content. 
                Also describe any visual elements like charts, diagrams, or layouts 
                that provide important context. Focus on:
                1. All readable text (including text in images, charts, diagrams)
                2. Data from tables, charts, and graphs
                3. Important visual context and relationships
                4. Any structured information
                
                Provide the extracted text in a clear, structured format.
                """
            else:
                prompt = """
                Extract all text content from this image accurately. 
                Focus on readable text and maintain the original structure and formatting as much as possible.
                Return only the extracted text content.
                """

            # Make API call
            response = await self._openai_client.chat.completions.create(
                model=self.settings.openai_gpt4o_deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_data.image_format.lower()};base64,{image_b64}",
                                    "detail": (
                                        "high" if enable_advanced_vision else "low"
                                    ),
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000 if enable_advanced_vision else 500,
                temperature=0.1,
            )

            extracted_text = response.choices[0].message.content

            # Calculate cost (estimated based on OpenAI pricing)
            cost_per_image = 0.005 if enable_advanced_vision else 0.003

            return OCRResult(
                text=extracted_text or "",
                confidence=0.95,  # GPT-4o typically has high confidence
                provider="gpt4o_vision",
                processing_time_ms=0,  # Will be set by caller
                cost_usd=cost_per_image,
                image_id=image_data.image_id,
                bounding_boxes=[],  # GPT-4o doesn't provide bounding boxes directly
            )

        except Exception as e:
            self.logger.error(
                "GPT-4o Vision processing failed",
                image_id=image_data.image_id,
                error=str(e),
            )
            return None

    async def _preprocess_image_for_ocr(self, image_data: ImageData) -> bytes:
        """
        Preprocess image to optimize for OCR accuracy.

        Applies enhancement techniques to improve text recognition.
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data.image_bytes))

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too large
            if (
                image.width > self.max_image_size[0]
                or image.height > self.max_image_size[1]
            ):
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)

            # Enhance image for better OCR
            # Increase contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)

            # Increase sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)

            # Save processed image
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue()

        except Exception as e:
            self.logger.warning(
                "Image preprocessing failed, using original",
                image_id=image_data.image_id,
                error=str(e),
            )
            return image_data.image_bytes

    async def _simulate_mistral_ocr(self, image_data: ImageData) -> str:
        """
        Simulate Mistral OCR processing.

        This is a placeholder until Mistral OCR API is available.
        In practice, this would be replaced with actual Mistral OCR calls.
        """
        # For simulation, we'll use a simple approach or return placeholder text
        # In real implementation, this would call Mistral's OCR service

        # Simple simulation based on image characteristics
        if image_data.width > 800 and image_data.height > 600:
            return f"[OCR simulation: Large image {image_data.image_id} likely contains substantial text content]"
        elif image_data.width < 200 or image_data.height < 200:
            return f"[OCR simulation: Small image {image_data.image_id} - limited text content]"
        else:
            return f"[OCR simulation: Medium image {image_data.image_id} - moderate text content]"

    async def cleanup(self):
        """Clean up resources and close connections."""
        if self._mistral_client:
            await self._mistral_client.aclose()

        if self._openai_client:
            await self._openai_client.close()


# Utility functions for image analysis
def calculate_image_quality_score(image_data: ImageData) -> float:
    """
    Calculate quality score for an image to help with processing decisions.

    Returns score from 0.0 (poor quality) to 1.0 (excellent quality).
    """
    try:
        image = Image.open(io.BytesIO(image_data.image_bytes))

        # Basic quality indicators
        quality_score = 0.0

        # Resolution factor
        total_pixels = image_data.width * image_data.height
        resolution_factor = min(total_pixels / (1000 * 1000), 1.0)  # Normalize to 1MP
        quality_score += resolution_factor * 0.4

        # DPI factor (if available)
        if image_data.dpi and image_data.dpi > 150:
            dpi_factor = min(image_data.dpi / 300, 1.0)  # Normalize to 300 DPI
            quality_score += dpi_factor * 0.3
        else:
            quality_score += 0.5  # Assume moderate DPI

        # File size efficiency (good compression indicates quality)
        bytes_per_pixel = image_data.file_size_bytes / total_pixels
        if 1 <= bytes_per_pixel <= 4:  # Reasonable range for compressed images
            quality_score += 0.3
        elif bytes_per_pixel > 4:  # Uncompressed or high quality
            quality_score += 0.2
        else:  # Over-compressed
            quality_score += 0.1

        return min(quality_score, 1.0)

    except Exception:
        return 0.5  # Default moderate quality
