"""
Embedding generation module for cost-effective document preprocessing.

This module implements multiple embedding generation strategies using local
SentenceTransformers for cost-effective embeddings and Azure OpenAI for
high-quality embeddings when needed. Includes caching and batch processing
for optimization.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import structlog
from dataclasses import dataclass
import hashlib
import pickle
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from azure.core.credentials import AzureKeyCredential

from config.settings import get_settings, EmbeddingProvider


@dataclass
class EmbeddingResult:
    """Result of embedding generation with metadata."""

    embeddings: List[List[float]]
    provider: str
    model_name: str
    embedding_dimension: int
    processing_time_ms: int
    cost_usd: float
    total_tokens: int
    cached_count: int  # Number of embeddings served from cache


class EmbeddingCache:
    """
    Simple file-based embedding cache to avoid redundant processing.

    Caches embeddings based on text content hash to reduce costs and
    improve performance for repeated content.
    """

    def __init__(self, cache_dir: Path = Path("cache/embeddings")):
        """Initialize embedding cache."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger(__name__)

        # In-memory cache for session reuse
        self._memory_cache: Dict[str, List[float]] = {}
        self.max_memory_cache_size = 1000

    def _get_cache_key(self, text: str, provider: str, model: str) -> str:
        """Generate cache key for text and model combination."""
        content = f"{provider}:{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pkl"

    async def get(self, text: str, provider: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if available."""
        cache_key = self._get_cache_key(text, provider, model)

        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Check file cache
        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    embedding = pickle.load(f)

                # Add to memory cache
                if len(self._memory_cache) < self.max_memory_cache_size:
                    self._memory_cache[cache_key] = embedding

                return embedding
            except Exception as e:
                self.logger.warning("Failed to load cached embedding", error=str(e))

        return None

    async def set(self, text: str, provider: str, model: str, embedding: List[float]):
        """Cache an embedding."""
        cache_key = self._get_cache_key(text, provider, model)

        # Add to memory cache
        if len(self._memory_cache) < self.max_memory_cache_size:
            self._memory_cache[cache_key] = embedding

        # Save to file cache
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning("Failed to cache embedding", error=str(e))

    def clear_memory_cache(self):
        """Clear in-memory cache."""
        self._memory_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        file_cache_count = len(list(self.cache_dir.glob("*.pkl")))
        memory_cache_count = len(self._memory_cache)

        return {
            "memory_cache_count": memory_cache_count,
            "file_cache_count": file_cache_count,
            "cache_dir": str(self.cache_dir),
        }


class EmbeddingGenerator:
    """
    Cost-effective embedding generator with multiple provider support.

    Supports local SentenceTransformers for cost-effective embeddings and
    Azure OpenAI for high-quality embeddings with intelligent caching and
    batch processing optimization.
    """

    def __init__(self):
        """Initialize the embedding generator with configuration."""
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)

        # Initialize providers
        self._sentence_transformer_model = None
        self._openai_client = None

        # Embedding cache
        self.cache = EmbeddingCache()

        # Cost tracking
        self.embedding_costs = {
            EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS: 0.0,  # Free
            EmbeddingProvider.AZURE_OPENAI_SMALL: 0.00002,  # $0.00002 per 1K tokens
            EmbeddingProvider.AZURE_OPENAI_LARGE: 0.00006,  # $0.00006 per 1K tokens
        }

        # Model configurations
        self.model_configs = {
            EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS: {
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384,
                "max_seq_length": 256,
            },
            EmbeddingProvider.AZURE_OPENAI_SMALL: {
                "model_name": "text-embedding-3-small",
                "dimension": 1536,
                "max_seq_length": 8192,
            },
            EmbeddingProvider.AZURE_OPENAI_LARGE: {
                "model_name": "text-embedding-3-large",
                "dimension": 3072,
                "max_seq_length": 8192,
            },
        }

    async def generate_embeddings(
        self,
        texts: List[str],
        provider: EmbeddingProvider = EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
        batch_size: Optional[int] = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts using specified provider.

        Args:
            texts: List of text strings to embed
            provider: Embedding provider to use
            batch_size: Batch size for processing (provider-specific defaults used if None)

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                provider=provider.value,
                model_name="",
                embedding_dimension=0,
                processing_time_ms=0,
                cost_usd=0.0,
                total_tokens=0,
                cached_count=0,
            )

        start_time = time.time()

        self.logger.info(
            "Starting embedding generation",
            provider=provider,
            text_count=len(texts),
            batch_size=batch_size,
        )

        try:
            if provider == EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS:
                result = await self._generate_sentence_transformer_embeddings(
                    texts, batch_size
                )
            elif provider == EmbeddingProvider.AZURE_OPENAI_SMALL:
                result = await self._generate_azure_openai_embeddings(
                    texts, self.settings.openai_embedding_deployment_small, batch_size
                )
            elif provider == EmbeddingProvider.AZURE_OPENAI_LARGE:
                result = await self._generate_azure_openai_embeddings(
                    texts, self.settings.openai_embedding_deployment_large, batch_size
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")

            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = int(processing_time)

            self.logger.info(
                "Embedding generation completed",
                provider=provider,
                text_count=len(texts),
                dimension=result.embedding_dimension,
                cost=result.cost_usd,
                cached_count=result.cached_count,
                processing_time=processing_time,
            )

            return result

        except Exception as e:
            self.logger.error(
                "Embedding generation failed",
                provider=provider,
                text_count=len(texts),
                error=str(e),
                exc_info=True,
            )
            raise

    async def _generate_sentence_transformer_embeddings(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> EmbeddingResult:
        """Generate embeddings using local SentenceTransformers model."""
        # Initialize model if needed
        if not self._sentence_transformer_model:
            model_name = self.model_configs[
                EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS
            ]["model_name"]
            self.logger.info("Loading SentenceTransformer model", model=model_name)
            self._sentence_transformer_model = SentenceTransformer(model_name)

        model_config = self.model_configs[EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS]
        batch_size = batch_size or 32  # Default batch size for local processing

        embeddings = []
        cached_count = 0

        # Process texts in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                # Check cache first
                cached_embedding = await self.cache.get(
                    text,
                    EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS.value,
                    model_config["model_name"],
                )

                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                    cached_count += 1
                else:
                    # Generate new embedding
                    text_embedding = self._sentence_transformer_model.encode(
                        [text], convert_to_numpy=True, normalize_embeddings=True
                    )[0].tolist()

                    batch_embeddings.append(text_embedding)

                    # Cache the embedding
                    await self.cache.set(
                        text,
                        EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS.value,
                        model_config["model_name"],
                        text_embedding,
                    )

            embeddings.extend(batch_embeddings)

        return EmbeddingResult(
            embeddings=embeddings,
            provider=EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS.value,
            model_name=model_config["model_name"],
            embedding_dimension=model_config["dimension"],
            processing_time_ms=0,  # Will be set by caller
            cost_usd=0.0,  # Free local processing
            total_tokens=sum(
                len(text.split()) for text in texts
            ),  # Rough token estimate
            cached_count=cached_count,
        )

    async def _generate_azure_openai_embeddings(
        self, texts: List[str], deployment_name: str, batch_size: Optional[int] = None
    ) -> EmbeddingResult:
        """Generate embeddings using Azure OpenAI."""
        # Initialize client if needed
        if not self._openai_client:
            self._openai_client = openai.AsyncAzureOpenAI(
                api_key=self.settings.openai_api_key,
                api_version=self.settings.openai_api_version,
                azure_endpoint=self.settings.openai_endpoint,
            )

        # Determine provider type for cost calculation
        if deployment_name == self.settings.openai_embedding_deployment_small:
            provider = EmbeddingProvider.AZURE_OPENAI_SMALL
        else:
            provider = EmbeddingProvider.AZURE_OPENAI_LARGE

        model_config = self.model_configs[provider]
        batch_size = batch_size or 16  # Smaller batch size for API calls

        embeddings = []
        total_tokens = 0
        cached_count = 0

        # Process texts in batches to stay within API limits
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = []
            batch_tokens = 0

            # Check cache for each text in batch
            texts_to_process = []
            indices_to_process = []

            for idx, text in enumerate(batch_texts):
                cached_embedding = await self.cache.get(
                    text, provider.value, model_config["model_name"]
                )

                if cached_embedding is not None:
                    batch_embeddings.append((idx, cached_embedding))
                    cached_count += 1
                else:
                    texts_to_process.append(text)
                    indices_to_process.append(idx)

            # Process non-cached texts
            if texts_to_process:
                try:
                    response = await self._openai_client.embeddings.create(
                        model=deployment_name, input=texts_to_process
                    )

                    # Extract embeddings and tokens
                    api_embeddings = [data.embedding for data in response.data]
                    batch_tokens = response.usage.total_tokens

                    # Add to batch results with correct indices
                    for api_idx, batch_idx in enumerate(indices_to_process):
                        embedding = api_embeddings[api_idx]
                        batch_embeddings.append((batch_idx, embedding))

                        # Cache the embedding
                        await self.cache.set(
                            texts_to_process[api_idx],
                            provider.value,
                            model_config["model_name"],
                            embedding,
                        )

                except Exception as e:
                    self.logger.error(
                        "Azure OpenAI embedding API call failed", error=str(e)
                    )
                    raise

            # Sort by original index and extract embeddings
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])
            total_tokens += batch_tokens

            # Rate limiting - small delay between batches
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)

        # Calculate cost
        cost_per_1k_tokens = self.embedding_costs[provider]
        total_cost = (total_tokens / 1000) * cost_per_1k_tokens

        return EmbeddingResult(
            embeddings=embeddings,
            provider=provider.value,
            model_name=model_config["model_name"],
            embedding_dimension=model_config["dimension"],
            processing_time_ms=0,  # Will be set by caller
            cost_usd=total_cost,
            total_tokens=total_tokens,
            cached_count=cached_count,
        )

    async def get_embedding_for_query(
        self,
        query: str,
        provider: EmbeddingProvider = EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
    ) -> List[float]:
        """
        Generate embedding for a single query string.

        Optimized for query processing in search scenarios.
        """
        result = await self.generate_embeddings([query], provider)
        return result.embeddings[0] if result.embeddings else []

    def calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Returns similarity score from -1 (opposite) to 1 (identical).
        """
        try:
            # Convert to numpy arrays for efficient computation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            self.logger.warning("Similarity calculation failed", error=str(e))
            return 0.0

    async def validate_embeddings(
        self, embeddings: List[List[float]], expected_dimension: int
    ) -> bool:
        """
        Validate that embeddings have correct dimensions and values.

        Returns True if embeddings are valid, False otherwise.
        """
        try:
            if not embeddings:
                return False

            for embedding in embeddings:
                # Check dimension
                if len(embedding) != expected_dimension:
                    self.logger.error(
                        "Embedding dimension mismatch",
                        expected=expected_dimension,
                        actual=len(embedding),
                    )
                    return False

                # Check for valid numeric values
                if not all(
                    isinstance(x, (int, float)) and not np.isnan(x) for x in embedding
                ):
                    self.logger.error("Invalid embedding values detected")
                    return False

                # Check for reasonable magnitude (embeddings should typically be normalized)
                magnitude = np.linalg.norm(embedding)
                if magnitude == 0 or magnitude > 10:  # Reasonable bounds
                    self.logger.warning(
                        "Unusual embedding magnitude", magnitude=magnitude
                    )

            return True

        except Exception as e:
            self.logger.error("Embedding validation failed", error=str(e))
            return False

    def get_provider_info(self, provider: EmbeddingProvider) -> Dict[str, Any]:
        """Get information about a specific embedding provider."""
        config = self.model_configs.get(provider, {})
        cost_per_1k = self.embedding_costs.get(provider, 0.0)

        return {
            "provider": provider.value,
            "model_name": config.get("model_name", "unknown"),
            "dimension": config.get("dimension", 0),
            "max_sequence_length": config.get("max_seq_length", 0),
            "cost_per_1k_tokens": cost_per_1k,
            "is_local": provider == EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
        }

    async def benchmark_providers(
        self,
        sample_texts: List[str],
        providers: Optional[List[EmbeddingProvider]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different embedding providers on sample texts.

        Useful for choosing the best provider for specific use cases.
        """
        if providers is None:
            providers = list(EmbeddingProvider)

        results = {}

        for provider in providers:
            try:
                start_time = time.time()
                result = await self.generate_embeddings(sample_texts, provider)
                end_time = time.time()

                results[provider.value] = {
                    "success": True,
                    "embedding_dimension": result.embedding_dimension,
                    "processing_time_ms": (end_time - start_time) * 1000,
                    "cost_usd": result.cost_usd,
                    "cost_per_text": (
                        result.cost_usd / len(sample_texts) if sample_texts else 0
                    ),
                    "cached_count": result.cached_count,
                    "provider_info": self.get_provider_info(provider),
                }

            except Exception as e:
                results[provider.value] = {
                    "success": False,
                    "error": str(e),
                    "provider_info": self.get_provider_info(provider),
                }

        return results

    async def cleanup(self):
        """Clean up resources and close connections."""
        if self._openai_client:
            await self._openai_client.close()

        # Clear memory cache to free up memory
        self.cache.clear_memory_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return self.cache.get_cache_stats()


# Utility functions for embedding operations
def chunk_text_for_embeddings(
    text: str, max_length: int = 1000, overlap: int = 100
) -> List[str]:
    """
    Split text into chunks suitable for embedding generation.

    Args:
        text: Text to chunk
        max_length: Maximum characters per chunk
        overlap: Character overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_length, len(text))

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for boundary in [". ", "! ", "? ", "\n\n"]:
                boundary_pos = text.rfind(boundary, start, end)
                if boundary_pos > start:
                    end = boundary_pos + len(boundary)
                    break
            else:
                # Look for any whitespace
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = max(end - overlap, start + 1)
        if start >= len(text):
            break

    return chunks
