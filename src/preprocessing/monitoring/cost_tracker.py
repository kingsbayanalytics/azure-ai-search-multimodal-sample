"""
Cost tracking module for the preprocessing pipeline.

This module tracks costs across different processing strategies and providers
to validate the 70-85% cost savings compared to Azure Document Intelligence.
It provides real-time cost monitoring and alerts to prevent budget overruns.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import structlog
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from config.settings import (
    get_settings,
    ProcessingStrategy,
    EmbeddingProvider,
    OCRProvider,
    StrategySettings,
)


@dataclass
class ProcessingCost:
    """Individual processing cost record."""

    timestamp: str
    document_id: str
    document_name: str
    strategy: ProcessingStrategy
    page_count: int
    image_count: int

    # Detailed cost breakdown
    text_processing_cost: float
    image_processing_cost: float
    embedding_cost: float
    ocr_cost: float
    total_cost: float

    # Provider details
    embedding_provider: EmbeddingProvider
    ocr_provider: Optional[OCRProvider]

    # Comparison metrics
    estimated_azure_doc_intel_cost: float
    cost_savings: float
    savings_percentage: float


@dataclass
class CostSummary:
    """Aggregated cost summary for reporting."""

    period_start: str
    period_end: str
    total_documents: int
    total_pages: int
    total_images: int

    # Cost breakdown by category
    total_processing_cost: float
    text_processing_cost: float
    image_processing_cost: float
    embedding_cost: float
    ocr_cost: float

    # Strategy breakdown
    strategy_costs: Dict[str, float]
    strategy_document_counts: Dict[str, int]

    # Savings analysis
    estimated_azure_doc_intel_cost: float
    total_savings: float
    average_savings_percentage: float

    # Performance metrics
    average_cost_per_document: float
    average_cost_per_page: float
    cost_efficiency_score: float  # Custom metric for cost optimization


class CostTracker:
    """
    Comprehensive cost tracking for the preprocessing pipeline.

    Tracks costs across all processing strategies and providers to validate
    cost savings and provide budget monitoring and alerts.
    """

    def __init__(self):
        """Initialize the cost tracker with configuration."""
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)

        # Cost tracking storage
        self.cost_records: List[ProcessingCost] = []
        self.daily_costs: Dict[str, float] = {}  # Date -> total cost
        self.hourly_costs: Dict[str, float] = {}  # Hour -> total cost

        # Cost rate configurations (USD)
        self.cost_rates = self._initialize_cost_rates()

        # Budget monitoring
        self.cost_alert_threshold = self.settings.cost_alert_threshold_usd
        self.daily_budget_limit = (
            self.cost_alert_threshold
        )  # Can be configured separately

    def _initialize_cost_rates(self) -> Dict[str, Dict[str, float]]:
        """Initialize cost rates for different services and operations."""
        return {
            "embedding": {
                "local_sentence_transformers": 0.0,  # Free local processing
                "azure_openai_small": 0.00002,  # $0.00002 per 1K tokens
                "azure_openai_large": 0.00006,  # $0.00006 per 1K tokens
            },
            "ocr": {
                "mistral": 0.0002,  # $0.0002 per image (estimated)
                "gpt4o_vision": 0.005,  # $0.005 per image (estimated)
                "hybrid": 0.002,  # Variable cost
            },
            "text_processing": {
                "pymupdf": 0.00001,  # $0.00001 per page
                "pdfplumber": 0.00002,  # $0.00002 per page
                "pymupdf4llm": 0.00001,  # $0.00001 per page
            },
            "azure_doc_intelligence": {
                "per_page": 0.10  # Estimated $0.10 per page for comparison
            },
        }

    async def calculate_document_cost(
        self,
        strategy: ProcessingStrategy,
        page_count: int,
        image_count: int,
        document_id: str = "",
        document_name: str = "",
    ) -> float:
        """
        Calculate the total processing cost for a document.

        Args:
            strategy: Processing strategy used
            page_count: Number of pages in document
            image_count: Number of images in document
            document_id: Unique document identifier
            document_name: Document filename

        Returns:
            Total processing cost in USD
        """
        strategy_config = StrategySettings.get_strategy_config(strategy)

        # Calculate individual cost components
        text_cost = await self._calculate_text_processing_cost(strategy, page_count)
        image_cost = await self._calculate_image_processing_cost(
            strategy_config.get("ocr_provider"), image_count
        )
        embedding_cost = await self._calculate_embedding_cost(
            strategy_config.get("embedding_provider"), page_count
        )

        total_cost = text_cost + image_cost + embedding_cost

        # Calculate comparison with Azure Document Intelligence
        azure_doc_intel_cost = (
            page_count * self.cost_rates["azure_doc_intelligence"]["per_page"]
        )
        cost_savings = azure_doc_intel_cost - total_cost
        savings_percentage = (
            (cost_savings / azure_doc_intel_cost * 100)
            if azure_doc_intel_cost > 0
            else 0
        )

        # Create cost record
        cost_record = ProcessingCost(
            timestamp=datetime.utcnow().isoformat(),
            document_id=document_id,
            document_name=document_name,
            strategy=strategy,
            page_count=page_count,
            image_count=image_count,
            text_processing_cost=text_cost,
            image_processing_cost=image_cost,
            embedding_cost=embedding_cost,
            ocr_cost=image_cost,  # Same as image cost for now
            total_cost=total_cost,
            embedding_provider=strategy_config.get("embedding_provider"),
            ocr_provider=strategy_config.get("ocr_provider"),
            estimated_azure_doc_intel_cost=azure_doc_intel_cost,
            cost_savings=cost_savings,
            savings_percentage=savings_percentage,
        )

        # Store cost record
        await self._store_cost_record(cost_record)

        # Check for cost alerts
        await self._check_cost_alerts(total_cost)

        self.logger.info(
            "Document cost calculated",
            document=document_name,
            strategy=strategy,
            total_cost=total_cost,
            savings_percentage=savings_percentage,
            pages=page_count,
            images=image_count,
        )

        return total_cost

    async def _calculate_text_processing_cost(
        self, strategy: ProcessingStrategy, page_count: int
    ) -> float:
        """Calculate cost for text processing based on strategy."""
        if strategy == ProcessingStrategy.TEXT_ONLY:
            # Use fastest PyMuPDF processing
            return page_count * self.cost_rates["text_processing"]["pymupdf"]
        elif strategy == ProcessingStrategy.TEXT_OPTIMIZED:
            # Use PDFPlumber for better layout analysis
            return page_count * self.cost_rates["text_processing"]["pdfplumber"]
        elif strategy == ProcessingStrategy.IMAGE_HEAVY:
            # Use PyMuPDF4LLM for LLM-optimized processing
            return page_count * self.cost_rates["text_processing"]["pymupdf4llm"]
        else:  # BALANCED
            # Mixed approach
            return (
                page_count
                * (
                    self.cost_rates["text_processing"]["pymupdf"]
                    + self.cost_rates["text_processing"]["pdfplumber"]
                )
                / 2
            )

    async def _calculate_image_processing_cost(
        self, ocr_provider: Optional[OCRProvider], image_count: int
    ) -> float:
        """Calculate cost for image processing and OCR."""
        if not ocr_provider or image_count == 0:
            return 0.0

        if ocr_provider == OCRProvider.MISTRAL:
            return image_count * self.cost_rates["ocr"]["mistral"]
        elif ocr_provider == OCRProvider.GPT4O_VISION:
            return image_count * self.cost_rates["ocr"]["gpt4o_vision"]
        elif ocr_provider == OCRProvider.HYBRID:
            # Assume 70% Mistral, 30% GPT-4o for complex images
            mistral_cost = image_count * 0.7 * self.cost_rates["ocr"]["mistral"]
            gpt4o_cost = image_count * 0.3 * self.cost_rates["ocr"]["gpt4o_vision"]
            return mistral_cost + gpt4o_cost

        return 0.0

    async def _calculate_embedding_cost(
        self, embedding_provider: EmbeddingProvider, page_count: int
    ) -> float:
        """Calculate cost for embedding generation."""
        if embedding_provider == EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS:
            return 0.0  # Free local processing

        # Estimate tokens per page (rough heuristic)
        estimated_tokens_per_page = 500
        total_tokens = page_count * estimated_tokens_per_page
        token_thousands = total_tokens / 1000

        if embedding_provider == EmbeddingProvider.AZURE_OPENAI_SMALL:
            return token_thousands * self.cost_rates["embedding"]["azure_openai_small"]
        elif embedding_provider == EmbeddingProvider.AZURE_OPENAI_LARGE:
            return token_thousands * self.cost_rates["embedding"]["azure_openai_large"]

        return 0.0

    async def _store_cost_record(self, cost_record: ProcessingCost):
        """Store cost record for tracking and reporting."""
        self.cost_records.append(cost_record)

        # Update daily tracking
        date_key = cost_record.timestamp[:10]  # YYYY-MM-DD
        self.daily_costs[date_key] = (
            self.daily_costs.get(date_key, 0.0) + cost_record.total_cost
        )

        # Update hourly tracking
        hour_key = cost_record.timestamp[:13]  # YYYY-MM-DDTHH
        self.hourly_costs[hour_key] = (
            self.hourly_costs.get(hour_key, 0.0) + cost_record.total_cost
        )

        # Persist to file if configured
        if self.settings.enable_cost_tracking:
            await self._persist_cost_record(cost_record)

    async def _persist_cost_record(self, cost_record: ProcessingCost):
        """Persist cost record to file for durability."""
        try:
            cost_log_file = Path("logs/cost_tracking.jsonl")
            cost_log_file.parent.mkdir(exist_ok=True)

            with open(cost_log_file, "a") as f:
                f.write(json.dumps(asdict(cost_record)) + "\n")

        except Exception as e:
            self.logger.warning("Failed to persist cost record", error=str(e))

    async def _check_cost_alerts(self, additional_cost: float):
        """Check if cost thresholds are exceeded and send alerts."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        today_total = self.daily_costs.get(today, 0.0)

        # Check daily budget limit
        if today_total > self.daily_budget_limit:
            await self._send_cost_alert(
                alert_type="daily_budget_exceeded",
                current_cost=today_total,
                threshold=self.daily_budget_limit,
                period="daily",
            )

        # Check hourly rate limits for expensive services
        current_hour = datetime.utcnow().strftime("%Y-%m-%dT%H")
        hourly_total = self.hourly_costs.get(current_hour, 0.0)

        # Example: Alert if hourly costs exceed $10
        hourly_threshold = 10.0
        if hourly_total > hourly_threshold:
            await self._send_cost_alert(
                alert_type="hourly_rate_high",
                current_cost=hourly_total,
                threshold=hourly_threshold,
                period="hourly",
            )

    async def _send_cost_alert(
        self, alert_type: str, current_cost: float, threshold: float, period: str
    ):
        """Send cost alert notification."""
        self.logger.warning(
            "Cost alert triggered",
            alert_type=alert_type,
            current_cost=current_cost,
            threshold=threshold,
            period=period,
        )

        # In production, this would integrate with monitoring systems
        # (e.g., Azure Monitor, Slack, email notifications)

    async def get_cost_summary(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> CostSummary:
        """
        Generate comprehensive cost summary for a time period.

        Args:
            start_date: Start of reporting period (defaults to 24 hours ago)
            end_date: End of reporting period (defaults to now)

        Returns:
            CostSummary with aggregated cost data
        """
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=1)
        if not end_date:
            end_date = datetime.utcnow()

        # Filter records by date range
        period_records = [
            record
            for record in self.cost_records
            if start_date
            <= datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
            <= end_date
        ]

        if not period_records:
            # Return empty summary
            return CostSummary(
                period_start=start_date.isoformat(),
                period_end=end_date.isoformat(),
                total_documents=0,
                total_pages=0,
                total_images=0,
                total_processing_cost=0.0,
                text_processing_cost=0.0,
                image_processing_cost=0.0,
                embedding_cost=0.0,
                ocr_cost=0.0,
                strategy_costs={},
                strategy_document_counts={},
                estimated_azure_doc_intel_cost=0.0,
                total_savings=0.0,
                average_savings_percentage=0.0,
                average_cost_per_document=0.0,
                average_cost_per_page=0.0,
                cost_efficiency_score=0.0,
            )

        # Aggregate data
        total_documents = len(period_records)
        total_pages = sum(record.page_count for record in period_records)
        total_images = sum(record.image_count for record in period_records)

        total_processing_cost = sum(record.total_cost for record in period_records)
        text_processing_cost = sum(
            record.text_processing_cost for record in period_records
        )
        image_processing_cost = sum(
            record.image_processing_cost for record in period_records
        )
        embedding_cost = sum(record.embedding_cost for record in period_records)
        ocr_cost = sum(record.ocr_cost for record in period_records)

        # Strategy breakdown
        strategy_costs = {}
        strategy_document_counts = {}
        for record in period_records:
            strategy_key = record.strategy.value
            strategy_costs[strategy_key] = (
                strategy_costs.get(strategy_key, 0.0) + record.total_cost
            )
            strategy_document_counts[strategy_key] = (
                strategy_document_counts.get(strategy_key, 0) + 1
            )

        # Savings analysis
        estimated_azure_doc_intel_cost = sum(
            record.estimated_azure_doc_intel_cost for record in period_records
        )
        total_savings = sum(record.cost_savings for record in period_records)
        average_savings_percentage = (
            sum(record.savings_percentage for record in period_records)
            / total_documents
        )

        # Performance metrics
        average_cost_per_document = total_processing_cost / total_documents
        average_cost_per_page = (
            total_processing_cost / total_pages if total_pages > 0 else 0.0
        )

        # Cost efficiency score (custom metric: higher is better)
        cost_efficiency_score = (
            (total_savings / estimated_azure_doc_intel_cost * 100)
            if estimated_azure_doc_intel_cost > 0
            else 0.0
        )

        return CostSummary(
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_documents=total_documents,
            total_pages=total_pages,
            total_images=total_images,
            total_processing_cost=total_processing_cost,
            text_processing_cost=text_processing_cost,
            image_processing_cost=image_processing_cost,
            embedding_cost=embedding_cost,
            ocr_cost=ocr_cost,
            strategy_costs=strategy_costs,
            strategy_document_counts=strategy_document_counts,
            estimated_azure_doc_intel_cost=estimated_azure_doc_intel_cost,
            total_savings=total_savings,
            average_savings_percentage=average_savings_percentage,
            average_cost_per_document=average_cost_per_document,
            average_cost_per_page=average_cost_per_page,
            cost_efficiency_score=cost_efficiency_score,
        )

    async def get_real_time_costs(self) -> Dict[str, Any]:
        """Get real-time cost metrics for monitoring dashboards."""
        now = datetime.utcnow()
        today = now.strftime("%Y-%m-%d")
        current_hour = now.strftime("%Y-%m-%dT%H")

        return {
            "current_daily_cost": self.daily_costs.get(today, 0.0),
            "current_hourly_cost": self.hourly_costs.get(current_hour, 0.0),
            "daily_budget_limit": self.daily_budget_limit,
            "daily_budget_utilization": (
                (self.daily_costs.get(today, 0.0) / self.daily_budget_limit * 100)
                if self.daily_budget_limit > 0
                else 0
            ),
            "total_documents_today": sum(
                1 for record in self.cost_records if record.timestamp.startswith(today)
            ),
            "last_24h_savings_percentage": await self._calculate_last_24h_savings(),
        }

    async def _calculate_last_24h_savings(self) -> float:
        """Calculate average savings percentage for last 24 hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_records = [
            record
            for record in self.cost_records
            if datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
            >= cutoff_time
        ]

        if not recent_records:
            return 0.0

        return sum(record.savings_percentage for record in recent_records) / len(
            recent_records
        )

    def export_cost_data(self, output_path: str) -> bool:
        """Export all cost data to a file for analysis."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_records": len(self.cost_records),
                "cost_records": [asdict(record) for record in self.cost_records],
                "daily_costs": self.daily_costs,
                "hourly_costs": self.hourly_costs,
                "cost_rates": self.cost_rates,
            }

            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            self.logger.info("Cost data exported", output_path=output_path)
            return True

        except Exception as e:
            self.logger.error("Failed to export cost data", error=str(e))
            return False
