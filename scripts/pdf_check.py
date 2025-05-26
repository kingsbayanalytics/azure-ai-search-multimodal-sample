#!/usr/bin/env python3
"""
PDF Document Analysis and Cost Estimation Tool

This script provides comprehensive analysis of PDF documents for the Azure AI Search
Multimodal RAG Demo, including integrity checks, size analysis, and processing cost estimates.

Usage:
    python scripts/pdf_check.py                           # Check all PDFs in data/
    python scripts/pdf_check.py --file path/to/file.pdf   # Check specific file
    python scripts/pdf_check.py --directory data/books/   # Check specific directory
    python scripts/pdf_check.py --summary                 # Show summary only
"""

import argparse
import fitz  # PyMuPDF
import os
import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class PDFAnalyzer:
    """
    Comprehensive PDF analysis tool for document processing cost estimation
    and integrity verification for multimodal RAG applications.
    """

    def __init__(self):
        """Initialize the PDF analyzer with cost estimation parameters."""
        # Cost estimates based on Azure services (as of 2024)
        self.cost_per_1k_pages_doc_intelligence = 1.50  # Azure Document Intelligence
        self.cost_per_image_gpt4o = 0.05  # GPT-4o image processing (high estimate)
        self.cost_per_image_mistral_ocr = (
            0.005  # Mistral OCR (cost-effective alternative)
        )
        self.cost_per_1k_tokens_embedding = 0.00013  # text-embedding-3-large
        self.avg_tokens_per_page = 500  # Conservative estimate for text-heavy documents

    def verify_pdf_integrity(self, pdf_path: str) -> Dict:
        """
        Verify PDF integrity and extract basic metadata.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Dict: Analysis results including integrity status and metadata
        """
        result = {
            "file_path": pdf_path,
            "file_name": os.path.basename(pdf_path),
            "integrity_status": "unknown",
            "error_message": None,
            "pages": 0,
            "file_size_mb": 0,
            "metadata": {},
        }

        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                result["integrity_status"] = "file_not_found"
                result["error_message"] = f"File not found: {pdf_path}"
                return result

            # Get file size
            result["file_size_mb"] = os.path.getsize(pdf_path) / (1024 * 1024)

            # Try to open and analyze PDF
            doc = fitz.open(pdf_path)
            result["pages"] = len(doc)

            # Extract metadata
            metadata = doc.metadata
            result["metadata"] = {
                "title": metadata.get("title", "Unknown"),
                "author": metadata.get("author", "Unknown"),
                "subject": metadata.get("subject", "Unknown"),
                "creator": metadata.get("creator", "Unknown"),
                "producer": metadata.get("producer", "Unknown"),
                "creation_date": metadata.get("creationDate", "Unknown"),
                "modification_date": metadata.get("modDate", "Unknown"),
            }

            # Basic content validation - check if pages have content
            empty_pages = 0
            for page_num in range(min(5, len(doc))):  # Check first 5 pages
                page = doc[page_num]
                text = page.get_text().strip()
                if len(text) < 10:  # Very little text
                    empty_pages += 1

            doc.close()

            # Determine integrity status
            if result["pages"] == 0:
                result["integrity_status"] = "empty_document"
                result["error_message"] = "Document has no pages"
            elif empty_pages >= min(3, result["pages"]):
                result["integrity_status"] = "mostly_empty"
                result["error_message"] = (
                    f"Document appears to have mostly empty pages ({empty_pages} of {min(5, result['pages'])} checked)"
                )
            else:
                result["integrity_status"] = "valid"

        except Exception as e:
            result["integrity_status"] = "corrupted"
            result["error_message"] = f"Error opening PDF: {str(e)}"

        return result

    def count_actual_images(self, pdf_path: str) -> Dict:
        """
        Count the actual number of images in a PDF document using multiple detection methods.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Dict: Image count details including total images and per-page breakdown
        """
        try:
            doc = fitz.open(pdf_path)

            # Method 1: Standard image detection
            standard_images = 0
            pages_with_standard_images = 0

            # Method 2: XObject detection (includes more image types)
            xobject_images = 0
            pages_with_xobjects = 0

            # Method 3: Content stream analysis for vector graphics
            pages_with_drawings = 0

            max_images_per_page = 0
            detection_details = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_analysis = {
                    "page": page_num + 1,
                    "standard_images": 0,
                    "xobject_images": 0,
                    "has_drawings": False,
                    "content_analysis": "",
                }

                # Method 1: Standard image detection
                image_list = page.get_images()
                page_standard_images = len(image_list)
                standard_images += page_standard_images
                page_analysis["standard_images"] = page_standard_images

                if page_standard_images > 0:
                    pages_with_standard_images += 1

                # Method 2: XObject detection (more comprehensive)
                try:
                    # Get page resources and look for XObjects
                    page_dict = page.get_contents()
                    if page_dict:
                        # Look for image-related content in the page stream
                        page_content = (
                            page_dict[0].get_buffer().decode("latin-1", errors="ignore")
                        )

                        # Count XObject references (Do /Im1, Do /Im2, etc.)
                        import re

                        xobject_pattern = r"/Im\d+"
                        xobject_matches = re.findall(xobject_pattern, page_content)
                        page_xobject_images = len(
                            set(xobject_matches)
                        )  # Use set to avoid duplicates
                        xobject_images += page_xobject_images
                        page_analysis["xobject_images"] = page_xobject_images

                        if page_xobject_images > 0:
                            pages_with_xobjects += 1

                        # Method 3: Look for drawing operations that might indicate figures/charts
                        drawing_patterns = [
                            r"\d+\.?\d*\s+\d+\.?\d*\s+m",  # moveto operations
                            r"\d+\.?\d*\s+\d+\.?\d*\s+l",  # lineto operations
                            r"\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*\s+re",  # rectangle
                            r"[fFbB]\*?$",  # fill operations
                            r"[sS]$",  # stroke operations
                        ]

                        drawing_ops = 0
                        for pattern in drawing_patterns:
                            matches = re.findall(pattern, page_content, re.MULTILINE)
                            drawing_ops += len(matches)

                        # If there are many drawing operations, likely contains figures
                        if (
                            drawing_ops > 20
                        ):  # Threshold for "significant drawing content"
                            page_analysis["has_drawings"] = True
                            pages_with_drawings += 1
                            page_analysis["content_analysis"] = (
                                f"{drawing_ops} drawing operations"
                            )

                except Exception as e:
                    page_analysis["content_analysis"] = f"Analysis error: {str(e)}"

                # Track maximum images per page
                total_page_images = max(page_standard_images, page_xobject_images)
                max_images_per_page = max(max_images_per_page, total_page_images)

                detection_details.append(page_analysis)

            doc.close()

            # Use the highest count from different detection methods
            best_image_count = max(standard_images, xobject_images)
            best_pages_with_images = max(
                pages_with_standard_images, pages_with_xobjects
            )

            # Add pages with significant drawing content to the count
            estimated_figure_pages = pages_with_drawings

            return {
                "total_images": best_image_count,
                "pages_with_images": best_pages_with_images,
                "max_images_per_page": max_images_per_page,
                "average_images_per_page": (
                    best_image_count / len(doc) if len(doc) > 0 else 0
                ),
                "image_density": (
                    best_pages_with_images / len(doc) if len(doc) > 0 else 0
                ),
                "detection_methods": {
                    "standard_images": standard_images,
                    "xobject_images": xobject_images,
                    "pages_with_drawings": pages_with_drawings,
                    "pages_with_standard_images": pages_with_standard_images,
                    "pages_with_xobjects": pages_with_xobjects,
                },
                "estimated_figures": estimated_figure_pages,
                "detection_details": (
                    detection_details[:5]
                    if len(detection_details) > 5
                    else detection_details
                ),  # Show first 5 pages for debugging
            }

        except Exception as e:
            return {
                "total_images": 0,
                "pages_with_images": 0,
                "max_images_per_page": 0,
                "average_images_per_page": 0,
                "image_density": 0,
                "error": f"Error counting images: {str(e)}",
            }

    def estimate_processing_costs(
        self,
        analysis_result: Dict,
        strategy: str = "indexer-image-verbal",
        manual_figure_count: int = None,
    ) -> Dict:
        """
        Estimate processing costs based on document characteristics and processing strategy.

        Args:
            analysis_result (Dict): Result from verify_pdf_integrity
            strategy (str): Processing strategy ("indexer-image-verbal", "self-multimodal-embedding", "optimized")

        Returns:
            Dict: Detailed cost breakdown
        """
        if analysis_result["integrity_status"] != "valid":
            return {
                "total_cost": 0,
                "error": "Cannot estimate costs for invalid document",
            }

        pages = analysis_result["pages"]
        file_size_mb = analysis_result["file_size_mb"]
        pdf_path = analysis_result["file_path"]

        # Count actual images in the PDF
        image_analysis = self.count_actual_images(pdf_path)
        actual_images = image_analysis["total_images"]

        # Handle manual figure count override
        if manual_figure_count is not None:
            text_based_figures = manual_figure_count
            total_visual_elements = actual_images + text_based_figures
            estimated_images = max(1, total_visual_elements)
        else:
            # For cost estimation, consider both embedded images and text-based figures
            # Academic PDFs often have text-based tables/charts that will be processed by AI
            text_based_figures = 0
            if "detection_methods" in image_analysis:
                # Estimate text-based figures from figure references in text
                # This is a heuristic - in practice, you'd want to manually verify
                figure_mentions = len(
                    [
                        d
                        for d in image_analysis.get("detection_details", [])
                        if "figure" in d.get("content_analysis", "").lower()
                    ]
                )
                text_based_figures = max(0, figure_mentions)

            # Total visual elements that AI will process
            total_visual_elements = actual_images + text_based_figures

            # Use actual count, but ensure minimum of 1 for processing cost calculation
            estimated_images = max(1, total_visual_elements)

        cost_breakdown = {
            "strategy": strategy,
            "pages": pages,
            "actual_images": actual_images,
            "estimated_images": estimated_images,
            "image_analysis": image_analysis,
            "document_intelligence_cost": 0,
            "image_processing_cost": 0,
            "embedding_cost": 0,
            "total_cost": 0,
        }

        # Add manual figure count if provided
        if manual_figure_count is not None:
            cost_breakdown["manual_figures"] = manual_figure_count

        if strategy == "indexer-image-verbal":
            # Azure Document Intelligence
            cost_breakdown["document_intelligence_cost"] = (
                pages / 1000
            ) * self.cost_per_1k_pages_doc_intelligence

            # GPT-4o image processing
            cost_breakdown["image_processing_cost"] = (
                estimated_images * self.cost_per_image_gpt4o
            )

            # Text embeddings
            estimated_tokens = pages * self.avg_tokens_per_page
            cost_breakdown["embedding_cost"] = (
                estimated_tokens / 1000
            ) * self.cost_per_1k_tokens_embedding

        elif strategy == "self-multimodal-embedding":
            # Higher cost due to multimodal embeddings
            cost_breakdown["document_intelligence_cost"] = (
                pages / 1000
            ) * self.cost_per_1k_pages_doc_intelligence
            cost_breakdown["image_processing_cost"] = (
                estimated_images * self.cost_per_image_gpt4o * 1.5
            )  # Higher for multimodal
            estimated_tokens = (
                pages * self.avg_tokens_per_page * 1.2
            )  # More tokens for multimodal
            cost_breakdown["embedding_cost"] = (
                estimated_tokens / 1000
            ) * self.cost_per_1k_tokens_embedding

        elif strategy == "optimized":
            # Cost-optimized approach using local processing and Mistral OCR
            cost_breakdown["document_intelligence_cost"] = 0  # Local processing
            cost_breakdown["image_processing_cost"] = (
                estimated_images * self.cost_per_image_mistral_ocr
            )
            cost_breakdown["embedding_cost"] = 0  # Local embeddings

        cost_breakdown["total_cost"] = (
            cost_breakdown["document_intelligence_cost"]
            + cost_breakdown["image_processing_cost"]
            + cost_breakdown["embedding_cost"]
        )

        return cost_breakdown

    def analyze_single_pdf(
        self, pdf_path: str, strategy: str = "indexer-image-verbal"
    ) -> Dict:
        """
        Perform complete analysis of a single PDF file.

        Args:
            pdf_path (str): Path to the PDF file
            strategy (str): Processing strategy for cost estimation

        Returns:
            Dict: Complete analysis results
        """
        integrity_result = self.verify_pdf_integrity(pdf_path)
        cost_result = self.estimate_processing_costs(integrity_result, strategy)

        return {
            "integrity": integrity_result,
            "costs": cost_result,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def analyze_directory(
        self, directory_path: str, strategy: str = "indexer-image-verbal"
    ) -> Dict:
        """
        Analyze all PDF files in a directory.

        Args:
            directory_path (str): Path to directory containing PDFs
            strategy (str): Processing strategy for cost estimation

        Returns:
            Dict: Analysis results for all PDFs in directory
        """
        pdf_files = glob.glob(os.path.join(directory_path, "**/*.pdf"), recursive=True)

        if not pdf_files:
            return {
                "directory": directory_path,
                "files_found": 0,
                "files_analyzed": [],
                "summary": {"total_cost": 0, "total_pages": 0, "valid_files": 0},
            }

        results = {
            "directory": directory_path,
            "files_found": len(pdf_files),
            "files_analyzed": [],
            "summary": {
                "total_cost": 0,
                "total_pages": 0,
                "total_size_mb": 0,
                "valid_files": 0,
                "corrupted_files": 0,
                "empty_files": 0,
                "strategy": strategy,
            },
        }

        for pdf_file in pdf_files:
            analysis = self.analyze_single_pdf(pdf_file, strategy)
            results["files_analyzed"].append(analysis)

            # Update summary
            if analysis["integrity"]["integrity_status"] == "valid":
                results["summary"]["valid_files"] += 1
                results["summary"]["total_pages"] += analysis["integrity"]["pages"]
                results["summary"]["total_size_mb"] += analysis["integrity"][
                    "file_size_mb"
                ]
                results["summary"]["total_cost"] += analysis["costs"]["total_cost"]
            elif analysis["integrity"]["integrity_status"] == "corrupted":
                results["summary"]["corrupted_files"] += 1
            elif analysis["integrity"]["integrity_status"] in [
                "empty_document",
                "mostly_empty",
            ]:
                results["summary"]["empty_files"] += 1

        return results

    def print_analysis_report(self, analysis_result: Dict, detailed: bool = True):
        """
        Print a formatted analysis report.

        Args:
            analysis_result (Dict): Analysis results from analyze_single_pdf or analyze_directory
            detailed (bool): Whether to show detailed breakdown
        """
        if "files_analyzed" in analysis_result:
            # Directory analysis
            self._print_directory_report(analysis_result, detailed)
        else:
            # Single file analysis
            self._print_single_file_report(analysis_result, detailed)

    def _print_single_file_report(self, analysis: Dict, detailed: bool):
        """Print report for single file analysis."""
        integrity = analysis["integrity"]
        costs = analysis["costs"]

        print(f"\n📄 PDF Analysis Report")
        print(f"{'='*60}")
        print(f"File: {integrity['file_name']}")
        print(f"Path: {integrity['file_path']}")
        print(f"Status: {integrity['integrity_status'].replace('_', ' ').title()}")

        if integrity["integrity_status"] == "valid":
            print(f"Pages: {integrity['pages']}")
            print(f"Size: {integrity['file_size_mb']:.1f} MB")

            if detailed and integrity["metadata"]["title"] != "Unknown":
                print(f"Title: {integrity['metadata']['title']}")
                print(f"Author: {integrity['metadata']['author']}")

            # Show image analysis details
            if "image_analysis" in costs and detailed:
                img_analysis = costs["image_analysis"]
                print(f"\n🖼️  Image Analysis (Enhanced Detection)")
                print(f"{'─'*40}")
                print(f"Total Images: {costs['actual_images']}")
                print(f"Pages with Images: {img_analysis['pages_with_images']}")
                print(f"Image Density: {img_analysis['image_density']:.1%}")

                if "detection_methods" in img_analysis:
                    methods = img_analysis["detection_methods"]
                    print(f"\n🔍 Detection Methods:")
                    print(f"  Standard Images: {methods['standard_images']}")
                    print(f"  XObject Images: {methods['xobject_images']}")
                    print(f"  Pages with Drawings: {methods['pages_with_drawings']}")

                if img_analysis["pages_with_images"] > 0:
                    print(
                        f"Avg Images per Page: {img_analysis['average_images_per_page']:.1f}"
                    )
                    print(f"Max Images per Page: {img_analysis['max_images_per_page']}")

                # Show detection details for debugging
                if (
                    "detection_details" in img_analysis
                    and img_analysis["detection_details"]
                ):
                    print(f"\n🔬 Sample Page Analysis (first 5 pages):")
                    for detail in img_analysis["detection_details"]:
                        if (
                            detail["standard_images"] > 0
                            or detail["xobject_images"] > 0
                            or detail["has_drawings"]
                        ):
                            print(
                                f"  Page {detail['page']}: Std={detail['standard_images']}, XObj={detail['xobject_images']}, Drawings={detail['has_drawings']}"
                            )
                            if detail["content_analysis"]:
                                print(f"    {detail['content_analysis']}")

            print(f"\n💰 Cost Estimation (Strategy: {costs['strategy']})")
            print(f"{'─'*40}")
            if costs["strategy"] != "optimized":
                print(
                    f"Document Intelligence: ${costs['document_intelligence_cost']:.4f}"
                )
            print(f"Image Processing: ${costs['image_processing_cost']:.4f}")
            if "manual_figures" in costs:
                print(
                    f"  (Based on {costs['actual_images']} embedded images + {costs['manual_figures']} manual figures = {costs['estimated_images']} total)"
                )
            elif costs["actual_images"] != costs["estimated_images"]:
                print(
                    f"  (Based on {costs['actual_images']} actual images, min 1 for processing)"
                )
            else:
                print(f"  (Based on {costs['actual_images']} actual images)")
            if costs["strategy"] != "optimized":
                print(f"Text Embeddings: ${costs['embedding_cost']:.4f}")
            print(f"{'─'*40}")
            print(f"Total Estimated Cost: ${costs['total_cost']:.4f}")

        elif integrity["error_message"]:
            print(f"Error: {integrity['error_message']}")

    def _print_directory_report(self, results: Dict, detailed: bool):
        """Print report for directory analysis."""
        summary = results["summary"]

        print(f"\n📁 Directory Analysis Report")
        print(f"{'='*60}")
        print(f"Directory: {results['directory']}")
        print(f"Files Found: {results['files_found']}")
        print(f"Valid Files: {summary['valid_files']}")
        print(f"Corrupted Files: {summary['corrupted_files']}")
        print(f"Empty Files: {summary['empty_files']}")

        if summary["valid_files"] > 0:
            print(f"\n📊 Summary Statistics")
            print(f"{'─'*40}")
            print(f"Total Pages: {summary['total_pages']:,}")
            print(f"Total Size: {summary['total_size_mb']:.1f} MB")
            print(
                f"Average Pages per Document: {summary['total_pages'] / summary['valid_files']:.1f}"
            )

            print(f"\n💰 Total Cost Estimation (Strategy: {summary['strategy']})")
            print(f"{'─'*40}")
            print(f"Total Estimated Cost: ${summary['total_cost']:.2f}")
            print(
                f"Average Cost per Document: ${summary['total_cost'] / summary['valid_files']:.4f}"
            )

        # Enhanced per-file cost breakdown
        if (
            hasattr(self, "_show_per_file_costs")
            and self._show_per_file_costs
            and results["files_analyzed"]
        ):
            valid_files = [
                analysis
                for analysis in results["files_analyzed"]
                if analysis["integrity"]["integrity_status"] == "valid"
            ]

            if valid_files:
                print(f"\n💰 Per-File Cost Breakdown")
                print(f"{'='*80}")
                print(
                    f"{'File Name':<30} {'Pages':<6} {'Images':<7} {'Size(MB)':<8} {'Cost($)':<8}"
                )
                print(f"{'─'*30} {'─'*6} {'─'*7} {'─'*8} {'─'*8}")

                for analysis in valid_files:
                    integrity = analysis["integrity"]
                    costs = analysis["costs"]
                    file_name = integrity["file_name"]
                    if len(file_name) > 29:
                        file_name = file_name[:26] + "..."

                    print(
                        f"{file_name:<30} {integrity['pages']:<6} {costs['actual_images']:<7} "
                        f"{integrity['file_size_mb']:<8.1f} {costs['total_cost']:<8.4f}"
                    )

                print(f"{'─'*30} {'─'*6} {'─'*7} {'─'*8} {'─'*8}")
                print(
                    f"{'TOTAL':<30} {summary['total_pages']:<6} {'':<7} "
                    f"{summary['total_size_mb']:<8.1f} {summary['total_cost']:<8.2f}"
                )

        if detailed and results["files_analyzed"]:
            print(f"\n📋 Individual File Details")
            print(f"{'─'*60}")
            for analysis in results["files_analyzed"]:
                integrity = analysis["integrity"]
                costs = analysis["costs"]
                status_icon = "✅" if integrity["integrity_status"] == "valid" else "❌"
                print(f"{status_icon} {integrity['file_name']}")
                if integrity["integrity_status"] == "valid":
                    print(
                        f"   Pages: {integrity['pages']}, Size: {integrity['file_size_mb']:.1f}MB, "
                        f"Images: {costs['actual_images']}, Cost: ${costs['total_cost']:.4f}"
                    )
                    # Show cost breakdown for each file if detailed
                    if (
                        hasattr(self, "_show_per_file_costs")
                        and self._show_per_file_costs
                    ):
                        if costs["strategy"] != "optimized":
                            print(
                                f"     Doc Intelligence: ${costs['document_intelligence_cost']:.4f}"
                            )
                        print(
                            f"     Image Processing: ${costs['image_processing_cost']:.4f}"
                        )
                        if costs["strategy"] != "optimized":
                            print(
                                f"     Text Embeddings: ${costs['embedding_cost']:.4f}"
                            )
                else:
                    print(f"   Status: {integrity['integrity_status']}")
                    if integrity["error_message"]:
                        print(f"   Error: {integrity['error_message']}")


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="PDF Document Analysis and Cost Estimation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pdf_check.py                                    # Check all PDFs in data/
  python scripts/pdf_check.py --file data/books/mybook.pdf       # Check specific file
  python scripts/pdf_check.py --directory data/books/            # Check specific directory
  python scripts/pdf_check.py --directory data/books/ --per-file-costs  # Show detailed per-file cost table
  python scripts/pdf_check.py --strategy optimized               # Use cost-optimized strategy
  python scripts/pdf_check.py --summary                          # Show summary only
  python scripts/pdf_check.py --output analysis_report.json     # Save results to JSON
        """,
    )

    parser.add_argument("--file", "-f", type=str, help="Analyze a specific PDF file")

    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default="data",
        help="Directory to analyze (default: data)",
    )

    parser.add_argument(
        "--strategy",
        "-s",
        choices=["indexer-image-verbal", "self-multimodal-embedding", "optimized"],
        default="indexer-image-verbal",
        help="Processing strategy for cost estimation (default: indexer-image-verbal)",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary only (less detailed output)",
    )

    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")

    parser.add_argument(
        "--manual-figures",
        "-m",
        type=int,
        help="Manually specify number of figures/charts for cost estimation (overrides detection)",
    )

    parser.add_argument(
        "--per-file-costs",
        "-p",
        action="store_true",
        help="Show detailed per-file cost breakdown table (for directory analysis)",
    )

    args = parser.parse_args()

    analyzer = PDFAnalyzer()

    # Set the per-file costs flag on the analyzer instance
    analyzer._show_per_file_costs = args.per_file_costs

    try:
        if args.file:
            # Analyze single file
            if not os.path.exists(args.file):
                print(f"❌ Error: File not found: {args.file}")
                sys.exit(1)

            results = analyzer.analyze_single_pdf(args.file, args.strategy)

            # Apply manual figure override if specified
            if args.manual_figures is not None:
                results["costs"]["manual_figures"] = args.manual_figures
                results["costs"]["estimated_images"] = max(
                    1, results["costs"]["actual_images"] + args.manual_figures
                )
                # Recalculate costs with manual figure count
                results["costs"] = analyzer.estimate_processing_costs(
                    results["integrity"],
                    args.strategy,
                    manual_figure_count=args.manual_figures,
                )

            analyzer.print_analysis_report(results, detailed=not args.summary)

        else:
            # Analyze directory
            if not os.path.exists(args.directory):
                print(f"❌ Error: Directory not found: {args.directory}")
                sys.exit(1)

            results = analyzer.analyze_directory(args.directory, args.strategy)
            analyzer.print_analysis_report(results, detailed=not args.summary)

        # Save to JSON if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {args.output}")

    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
