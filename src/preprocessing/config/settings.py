"""
Configuration management for the cost-effective preprocessing pipeline.

This module provides comprehensive configuration for document processing strategies,
Azure service connections, cost optimization settings, and monitoring parameters.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum


class ProcessingStrategy(str, Enum):
    """Available document processing strategies for cost optimization."""

    TEXT_ONLY = "text_only"
    TEXT_OPTIMIZED = "text_optimized"
    IMAGE_HEAVY = "image_heavy"
    BALANCED = "balanced"
    AUTO = "auto"  # Automatic strategy selection


class EmbeddingProvider(str, Enum):
    """Available embedding generation providers."""

    LOCAL_SENTENCE_TRANSFORMERS = "local_sentence_transformers"
    AZURE_OPENAI_SMALL = "azure_openai_small"  # text-embedding-3-small
    AZURE_OPENAI_LARGE = "azure_openai_large"  # text-embedding-3-large


class OCRProvider(str, Enum):
    """Available OCR service providers."""

    MISTRAL = "mistral"  # Cost-effective OCR
    GPT4O_VISION = "gpt4o_vision"  # High-quality but expensive
    HYBRID = "hybrid"  # Intelligent selection based on image complexity


class PreprocessingSettings(BaseSettings):
    """Main configuration settings for the preprocessing pipeline."""

    # Environment and deployment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Azure service connections
    azure_subscription_id: str = Field(..., env="AZURE_SUBSCRIPTION_ID")
    azure_resource_group: str = Field(..., env="AZURE_RESOURCE_GROUP")
    azure_tenant_id: str = Field(..., env="AZURE_TENANT_ID")

    # Azure Search configuration
    search_service_name: str = Field(..., env="AZURE_SEARCH_SERVICE_NAME")
    search_service_key: str = Field(..., env="AZURE_SEARCH_SERVICE_KEY")
    search_service_endpoint: str = Field(..., env="AZURE_SEARCH_SERVICE_ENDPOINT")

    # Azure Storage configuration for documents
    storage_account_name: str = Field(..., env="AZURE_STORAGE_ACCOUNT_NAME")
    storage_account_key: str = Field(..., env="AZURE_STORAGE_ACCOUNT_KEY")
    storage_container_name: str = Field(
        default="documents", env="AZURE_STORAGE_CONTAINER_NAME"
    )

    # Azure OpenAI configuration (for embeddings and GPT-4o Vision)
    openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    openai_api_version: str = Field(
        default="2024-02-01", env="AZURE_OPENAI_API_VERSION"
    )
    openai_embedding_deployment_small: str = Field(
        default="text-embedding-3-small", env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT_SMALL"
    )
    openai_embedding_deployment_large: str = Field(
        default="text-embedding-3-large", env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE"
    )
    openai_gpt4o_deployment: str = Field(
        default="gpt-4o", env="AZURE_OPENAI_GPT4O_DEPLOYMENT"
    )

    # Mistral AI configuration (for cost-effective OCR)
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")
    mistral_endpoint: str = Field(
        default="https://api.mistral.ai/v1", env="MISTRAL_ENDPOINT"
    )

    # Processing strategy configuration
    default_processing_strategy: ProcessingStrategy = Field(
        default=ProcessingStrategy.AUTO, env="DEFAULT_PROCESSING_STRATEGY"
    )
    default_embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
        env="DEFAULT_EMBEDDING_PROVIDER",
    )
    default_ocr_provider: OCRProvider = Field(
        default=OCRProvider.MISTRAL, env="DEFAULT_OCR_PROVIDER"
    )

    # Document processing limits and thresholds
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    max_pages_per_document: int = Field(default=1000, env="MAX_PAGES_PER_DOCUMENT")
    batch_size: int = Field(default=10, env="BATCH_SIZE")

    # Strategy selection thresholds
    text_density_threshold: float = Field(
        default=0.7, env="TEXT_DENSITY_THRESHOLD"
    )  # Ratio for text-heavy detection
    image_ratio_threshold: float = Field(
        default=0.3, env="IMAGE_RATIO_THRESHOLD"
    )  # Ratio for image-heavy detection
    complex_layout_threshold: float = Field(
        default=0.5, env="COMPLEX_LAYOUT_THRESHOLD"
    )  # Layout complexity score

    # Cost optimization settings
    enable_cost_tracking: bool = Field(default=True, env="ENABLE_COST_TRACKING")
    cost_alert_threshold_usd: float = Field(
        default=100.0, env="COST_ALERT_THRESHOLD_USD"
    )
    max_gpt4o_calls_per_hour: int = Field(
        default=50, env="MAX_GPT4O_CALLS_PER_HOUR"
    )  # Rate limiting for expensive calls

    # Quality settings
    min_ocr_confidence: float = Field(default=0.8, env="MIN_OCR_CONFIDENCE")
    enable_quality_validation: bool = Field(
        default=True, env="ENABLE_QUALITY_VALIDATION"
    )
    citation_accuracy_threshold: float = Field(
        default=0.95, env="CITATION_ACCURACY_THRESHOLD"
    )

    # Monitoring and alerting
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    prometheus_port: int = Field(default=8000, env="PROMETHEUS_PORT")
    health_check_interval_seconds: int = Field(
        default=30, env="HEALTH_CHECK_INTERVAL_SECONDS"
    )

    # Processing parallelization
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    enable_parallel_processing: bool = Field(
        default=True, env="ENABLE_PARALLEL_PROCESSING"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("max_workers")
    def validate_max_workers(cls, v):
        """Ensure max_workers is reasonable for system resources."""
        import os

        cpu_count = os.cpu_count() or 4
        if v > cpu_count * 2:
            return cpu_count * 2
        return max(1, v)

    @validator("cost_alert_threshold_usd")
    def validate_cost_threshold(cls, v):
        """Ensure cost threshold is positive."""
        return max(0.0, v)


class StrategySettings:
    """Configuration for each processing strategy with cost and quality trade-offs."""

    STRATEGY_CONFIGS: Dict[ProcessingStrategy, Dict[str, Any]] = {
        ProcessingStrategy.TEXT_ONLY: {
            "description": "Pure text documents with minimal visual elements",
            "embedding_provider": EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
            "ocr_provider": None,  # Skip OCR entirely
            "process_images": False,
            "layout_analysis": False,
            "estimated_cost_per_page": 0.002,  # $0.002 per page
            "target_quality_score": 0.85,
            "use_cases": ["Text reports", "Articles", "Books"],
        },
        ProcessingStrategy.TEXT_OPTIMIZED: {
            "description": "Text-heavy with basic layout and minimal images",
            "embedding_provider": EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
            "ocr_provider": OCRProvider.MISTRAL,
            "process_images": True,
            "layout_analysis": True,
            "estimated_cost_per_page": 0.008,  # $0.008 per page
            "target_quality_score": 0.90,
            "use_cases": [
                "Forms",
                "Simple documents with headers",
                "Text with occasional images",
            ],
        },
        ProcessingStrategy.IMAGE_HEAVY: {
            "description": "Complex visuals, charts, diagrams requiring high-quality processing",
            "embedding_provider": EmbeddingProvider.AZURE_OPENAI_LARGE,
            "ocr_provider": OCRProvider.HYBRID,
            "process_images": True,
            "layout_analysis": True,
            "advanced_vision": True,
            "estimated_cost_per_page": 0.025,  # $0.025 per page
            "target_quality_score": 0.95,
            "use_cases": [
                "Scientific papers",
                "Financial reports",
                "Technical diagrams",
            ],
        },
        ProcessingStrategy.BALANCED: {
            "description": "Adaptive processing based on content analysis",
            "embedding_provider": EmbeddingProvider.AZURE_OPENAI_SMALL,
            "ocr_provider": OCRProvider.HYBRID,
            "process_images": True,
            "layout_analysis": True,
            "estimated_cost_per_page": 0.015,  # $0.015 per page (variable)
            "target_quality_score": 0.92,
            "use_cases": [
                "Mixed content",
                "General business documents",
                "Unknown document types",
            ],
        },
    }

    @classmethod
    def get_strategy_config(cls, strategy: ProcessingStrategy) -> Dict[str, Any]:
        """Get configuration for a specific processing strategy."""
        return cls.STRATEGY_CONFIGS.get(
            strategy, cls.STRATEGY_CONFIGS[ProcessingStrategy.BALANCED]
        )

    @classmethod
    def estimate_processing_cost(
        cls, strategy: ProcessingStrategy, page_count: int
    ) -> float:
        """Estimate processing cost for a document with given strategy and page count."""
        config = cls.get_strategy_config(strategy)
        return config["estimated_cost_per_page"] * page_count


def get_settings() -> PreprocessingSettings:
    """Get singleton instance of preprocessing settings."""
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = PreprocessingSettings()
    return get_settings._instance


# Export commonly used configurations
__all__ = [
    "PreprocessingSettings",
    "ProcessingStrategy",
    "EmbeddingProvider",
    "OCRProvider",
    "StrategySettings",
    "get_settings",
]
