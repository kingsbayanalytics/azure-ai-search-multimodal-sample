#!/usr/bin/env python3
"""
Deployment script for the cost-effective preprocessing pipeline.

This script demonstrates how to use the complete preprocessing system to
process documents and create Azure Search indexes at 70-85% cost savings
compared to Azure Document Intelligence.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import click
import structlog

# Add the preprocessing module to the path
sys.path.insert(0, str(Path(__file__).parent))

from document_processor import DocumentProcessor
from config.settings import get_settings, ProcessingStrategy


# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@click.group()
def cli():
    """Cost-Effective Document Preprocessing Pipeline"""
    pass


@cli.command()
@click.option(
    "--input-path",
    "-i",
    required=True,
    help="Path to input documents (file or directory)",
)
@click.option(
    "--output-index", "-o", required=True, help="Name for output Azure Search index"
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice([s.value for s in ProcessingStrategy]),
    default="auto",
    help="Processing strategy to use",
)
@click.option(
    "--batch-size", "-b", type=int, default=None, help="Batch size for processing"
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing index")
@click.option("--dry-run", is_flag=True, help="Analyze documents without processing")
def process(
    input_path: str,
    output_index: str,
    strategy: str,
    batch_size: Optional[int],
    overwrite: bool,
    dry_run: bool,
):
    """Process documents and create cost-effective Azure Search index."""

    async def run_processing():
        logger.info(
            "Starting preprocessing pipeline",
            input_path=input_path,
            output_index=output_index,
            strategy=strategy,
            dry_run=dry_run,
        )

        try:
            processor = DocumentProcessor()

            if dry_run:
                # Analyze documents without processing
                print(f"\n🔍 Analyzing documents in: {input_path}")
                documents = await processor._discover_documents(input_path)

                if not documents:
                    print("❌ No supported documents found")
                    return

                print(f"📄 Found {len(documents)} documents")

                # Analyze strategy distribution
                strategy_distribution = await processor.strategy_selector.analyze_batch_strategy_distribution(
                    documents
                )
                cost_estimate = await processor.strategy_selector.estimate_batch_cost(
                    documents
                )

                print("\n📊 Strategy Distribution:")
                for strat, count in strategy_distribution.items():
                    if count > 0:
                        print(f"  {strat}: {count} documents")

                print(f"\n💰 Cost Estimate:")
                print(
                    f"  Total estimated cost: ${cost_estimate['total_estimated_cost_usd']:.4f}"
                )
                print(
                    f"  Azure Doc Intelligence cost: ${cost_estimate['azure_doc_intel_estimated_cost_usd']:.2f}"
                )
                print(
                    f"  Estimated savings: {cost_estimate['estimated_savings_percentage']:.1f}%"
                )

            else:
                # Full processing
                processing_strategy = (
                    ProcessingStrategy(strategy) if strategy != "auto" else None
                )

                result = await processor.process_documents(
                    input_path=input_path,
                    output_index_name=output_index,
                    strategy=processing_strategy,
                    batch_size=batch_size,
                )

                # Display results
                print("\n" + "=" * 60)
                print("🎉 PREPROCESSING COMPLETE!")
                print("=" * 60)

                summary = result["processing_summary"]
                cost_analysis = result["cost_analysis"]

                print(
                    f"✅ Successfully processed: {summary['successfully_processed']} documents"
                )
                print(f"📄 Total pages processed: {summary['total_pages']}")
                print(
                    f"⏱️  Processing time: {summary['processing_duration_seconds']:.1f} seconds"
                )
                print(
                    f"📈 Processing rate: {summary['processing_rate_docs_per_minute']:.1f} docs/min"
                )

                print(f"\n💰 COST ANALYSIS:")
                print(
                    f"💵 Total processing cost: ${cost_analysis['total_processing_cost_usd']:.4f}"
                )
                print(
                    f"🏷️  Azure Doc Intelligence cost: ${cost_analysis['estimated_azure_doc_intel_cost_usd']:.2f}"
                )
                print(
                    f"💡 Cost savings: ${cost_analysis['estimated_savings_usd']:.2f} ({cost_analysis['estimated_savings_percentage']:.1f}%)"
                )
                print(
                    f"📊 Average cost per document: ${cost_analysis['average_cost_per_document']:.4f}"
                )
                print(
                    f"📋 Average cost per page: ${cost_analysis['average_cost_per_page']:.4f}"
                )

                print(f"\n🎯 STRATEGY USAGE:")
                for strategy_name, count in result["strategy_usage"].items():
                    if count > 0:
                        print(f"  {strategy_name}: {count} documents")

                print(f"\n🔍 AZURE SEARCH INDEX:")
                index_info = result["index_creation"]
                print(f"📇 Index name: {index_info['index_name']}")
                print(f"📊 Documents indexed: {index_info['documents_indexed']}")
                print(f"💾 Estimated index size: {index_info['index_size_mb']} MB")

                quality_metrics = result["quality_metrics"]
                print(f"\n✨ QUALITY METRICS:")
                print(
                    f"🎖️  Average quality score: {quality_metrics['average_quality_score']:.3f}"
                )
                print(
                    f"🔗 Citation compatibility: {'✅' if quality_metrics['citation_compatibility'] else '❌'}"
                )

                if cost_analysis["estimated_savings_percentage"] >= 70:
                    print("\n🎉 SUCCESS: Achieved target cost savings of 70-85%!")
                else:
                    print(
                        f"\n⚠️  WARNING: Cost savings ({cost_analysis['estimated_savings_percentage']:.1f}%) below 70% target"
                    )

                print(
                    f"\n🚀 READY: Your index '{output_index}' is ready for use with the existing app!"
                )

        except Exception as e:
            logger.error("Processing failed", error=str(e), exc_info=True)
            print(f"\n❌ Processing failed: {e}")
            sys.exit(1)

    asyncio.run(run_processing())


@cli.command()
@click.option(
    "--sample-docs", type=int, default=5, help="Number of sample documents to test"
)
def benchmark(sample_docs: int):
    """Benchmark different processing strategies and providers."""

    async def run_benchmark():
        logger.info("Starting benchmark", sample_docs=sample_docs)

        try:
            from strategy_selector import DocumentStrategySelector
            from embedding.embedding_generator import EmbeddingGenerator
            from config.settings import EmbeddingProvider

            # Sample texts for benchmarking
            sample_texts = [
                "This is a sample document for testing text processing capabilities.",
                "Another test document with different content to evaluate processing quality.",
                "A third document containing various types of content for comprehensive testing.",
                "Document four includes more complex text structures and formatting elements.",
                "The final test document contains mixed content types for strategy evaluation.",
            ][:sample_docs]

            print(f"\n🧪 Benchmarking with {len(sample_texts)} sample documents...")

            # Benchmark embedding providers
            embedding_generator = EmbeddingGenerator()
            providers = [
                EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
                EmbeddingProvider.AZURE_OPENAI_SMALL,
                EmbeddingProvider.AZURE_OPENAI_LARGE,
            ]

            benchmark_results = await embedding_generator.benchmark_providers(
                sample_texts, providers
            )

            print("\n📊 EMBEDDING PROVIDER BENCHMARK:")
            print("-" * 80)
            for provider, results in benchmark_results.items():
                if results["success"]:
                    print(f"\n🔹 {provider}:")
                    print(f"   ✅ Success: {results['success']}")
                    print(f"   📏 Dimension: {results['embedding_dimension']}")
                    print(
                        f"   ⏱️  Processing time: {results['processing_time_ms']:.1f}ms"
                    )
                    print(f"   💰 Cost: ${results['cost_usd']:.6f}")
                    print(f"   📊 Cost per text: ${results['cost_per_text']:.6f}")
                    print(f"   🗄️  Cached: {results['cached_count']}")
                else:
                    print(f"\n🔸 {provider}: ❌ {results['error']}")

            # Get cache statistics
            cache_stats = embedding_generator.get_cache_stats()
            print(f"\n💾 CACHE STATISTICS:")
            print(f"   Memory cache: {cache_stats['memory_cache_count']} items")
            print(f"   File cache: {cache_stats['file_cache_count']} items")

            print("\n🎯 RECOMMENDATIONS:")

            # Find best cost-effective option
            local_results = benchmark_results.get("local_sentence_transformers", {})
            if local_results.get("success"):
                print(
                    "   💡 Use LOCAL_SENTENCE_TRANSFORMERS for maximum cost savings (free)"
                )

            # Find best quality option
            large_results = benchmark_results.get("azure_openai_large", {})
            if large_results.get("success"):
                print("   🎖️  Use AZURE_OPENAI_LARGE for highest quality (higher cost)")

            # Find balanced option
            small_results = benchmark_results.get("azure_openai_small", {})
            if small_results.get("success"):
                print("   ⚖️  Use AZURE_OPENAI_SMALL for balanced quality and cost")

            await embedding_generator.cleanup()

        except Exception as e:
            logger.error("Benchmark failed", error=str(e), exc_info=True)
            print(f"\n❌ Benchmark failed: {e}")
            sys.exit(1)

    asyncio.run(run_benchmark())


@cli.command()
@click.option("--index-name", "-n", required=True, help="Index name to validate")
def validate(index_name: str):
    """Validate index compatibility with the existing app."""

    async def run_validation():
        logger.info("Starting index validation", index_name=index_name)

        try:
            from indexer.index_creator import IndexCreator

            index_creator = IndexCreator()

            # Check if index exists and get stats
            index_stats = await index_creator.get_index_stats(index_name)
            if not index_stats:
                print(f"❌ Index '{index_name}' not found or inaccessible")
                return

            # Validate app compatibility
            compatibility = index_creator.validate_app_compatibility(index_name)

            print(f"\n🔍 INDEX VALIDATION REPORT")
            print("=" * 50)
            print(f"Index name: {index_name}")
            print(f"Document count: {index_stats['document_count']}")
            print(f"Field count: {index_stats['field_count']}")
            print(f"Vector search enabled: {index_stats['vector_search_enabled']}")

            print(f"\n🔗 APP COMPATIBILITY:")
            if compatibility["compatible"]:
                print("✅ COMPATIBLE - Index can be used with the existing app")
                print("🎉 Visual citations will work correctly")
                print("🔍 Search functionality is fully supported")
            else:
                print("❌ NOT COMPATIBLE - Index cannot be used with the existing app")
                if compatibility.get("missing_fields"):
                    print(
                        f"   Missing required fields: {', '.join(compatibility['missing_fields'])}"
                    )
                if not compatibility.get("has_vector_search"):
                    print("   Missing vector search configuration")

            print(f"\n📊 TECHNICAL DETAILS:")
            print(
                f"   Vector search: {'✅' if compatibility['has_vector_search'] else '❌'}"
            )
            print(f"   Total fields: {compatibility['total_fields']}")
            print(f"   Validation time: {compatibility['validation_timestamp']}")

        except Exception as e:
            logger.error("Validation failed", error=str(e), exc_info=True)
            print(f"\n❌ Validation failed: {e}")
            sys.exit(1)

    asyncio.run(run_validation())


@cli.command()
def setup():
    """Set up the preprocessing environment and check configuration."""

    print("🚀 COST-EFFECTIVE PREPROCESSING PIPELINE SETUP")
    print("=" * 60)

    try:
        settings = get_settings()

        print("✅ Configuration loaded successfully")
        print(f"   Environment: {settings.environment}")
        print(f"   Debug mode: {settings.debug}")
        print(f"   Log level: {settings.log_level}")

        print("\n🔧 AZURE SERVICES:")
        print(f"   Search service: {settings.search_service_name}")
        print(f"   Storage account: {settings.storage_account_name}")
        print(f"   Resource group: {settings.azure_resource_group}")

        print("\n💰 COST OPTIMIZATION:")
        print(f"   Default strategy: {settings.default_processing_strategy}")
        print(f"   Default embedding: {settings.default_embedding_provider}")
        print(f"   Default OCR: {settings.default_ocr_provider}")
        print(f"   Cost alert threshold: ${settings.cost_alert_threshold_usd}")

        print("\n⚙️  PROCESSING LIMITS:")
        print(f"   Max file size: {settings.max_file_size_mb} MB")
        print(f"   Max pages per document: {settings.max_pages_per_document}")
        print(f"   Batch size: {settings.batch_size}")
        print(f"   Max workers: {settings.max_workers}")

        print("\n🎯 STRATEGY THRESHOLDS:")
        print(f"   Text density threshold: {settings.text_density_threshold}")
        print(f"   Image ratio threshold: {settings.image_ratio_threshold}")
        print(f"   Complex layout threshold: {settings.complex_layout_threshold}")

        print("\n📈 QUALITY SETTINGS:")
        print(f"   Enable quality validation: {settings.enable_quality_validation}")
        print(f"   Citation accuracy threshold: {settings.citation_accuracy_threshold}")
        print(f"   Min OCR confidence: {settings.min_ocr_confidence}")

        print("\n🎉 Setup complete! You can now:")
        print("   1. Run 'python deploy.py process --help' to see processing options")
        print("   2. Use 'python deploy.py benchmark' to test different strategies")
        print("   3. Use 'python deploy.py validate' to check index compatibility")

        print("\n💡 Example usage:")
        print("   python deploy.py process -i /path/to/docs -o my-cost-effective-index")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\n🔧 Please check your .env configuration file")
        sys.exit(1)


if __name__ == "__main__":
    cli()
