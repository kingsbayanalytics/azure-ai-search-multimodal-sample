#!/usr/bin/env python3
"""
Debug script to investigate image detection in PDFs
"""

import fitz
import re
import sys


def debug_pdf_images(pdf_path):
    """Debug image detection in a PDF"""
    print(f"🔍 Debugging image detection for: {pdf_path}")
    print("=" * 60)

    doc = fitz.open(pdf_path)

    # Method 1: Standard image detection
    print("\n📊 Method 1: Standard Image Detection")
    total_standard = 0
    for page_num in range(min(10, len(doc))):  # Check first 10 pages
        page = doc[page_num]
        images = page.get_images()
        if images:
            print(f"  Page {page_num + 1}: {len(images)} images")
            total_standard += len(images)
    print(f"  Total standard images in first 10 pages: {total_standard}")

    # Method 2: XObject analysis
    print("\n📊 Method 2: XObject Analysis")
    total_xobjects = 0
    for page_num in range(min(10, len(doc))):
        page = doc[page_num]
        try:
            page_dict = page.get_contents()
            if page_dict:
                content = page_dict[0].get_buffer().decode("latin-1", errors="ignore")

                # Look for XObject references
                xobject_refs = re.findall(r"/Im\d+", content)
                figure_refs = re.findall(r"/Fig\d+", content)
                image_refs = re.findall(r"/Image\d+", content)

                all_refs = set(xobject_refs + figure_refs + image_refs)
                if all_refs:
                    print(
                        f"  Page {page_num + 1}: {len(all_refs)} XObject refs: {all_refs}"
                    )
                    total_xobjects += len(all_refs)
        except Exception as e:
            print(f"  Page {page_num + 1}: Error - {e}")
    print(f"  Total XObject references in first 10 pages: {total_xobjects}")

    # Method 3: Text analysis for figure references
    print("\n📊 Method 3: Text Analysis for Figure References")
    figure_mentions = []
    for page_num in range(min(50, len(doc))):  # Check more pages for text
        page = doc[page_num]
        text = page.get_text()

        # Look for figure references in text
        fig_patterns = [
            r"Figure\s+\d+",
            r"Fig\.\s+\d+",
            r"FIGURE\s+\d+",
            r"Chart\s+\d+",
            r"Diagram\s+\d+",
            r"Table\s+\d+",
        ]

        for pattern in fig_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                figure_mentions.append((page_num + 1, match))

    # Remove duplicates and sort
    unique_figures = list(set(figure_mentions))
    unique_figures.sort()

    print(f"  Found {len(unique_figures)} figure references in text:")
    for page, ref in unique_figures[:20]:  # Show first 20
        print(f"    Page {page}: {ref}")

    if len(unique_figures) > 20:
        print(f"    ... and {len(unique_figures) - 20} more")

    # Method 4: Content stream analysis
    print("\n📊 Method 4: Drawing Operations Analysis")
    pages_with_drawings = 0
    for page_num in range(min(10, len(doc))):
        page = doc[page_num]
        try:
            page_dict = page.get_contents()
            if page_dict:
                content = page_dict[0].get_buffer().decode("latin-1", errors="ignore")

                # Count drawing operations
                drawing_ops = 0
                drawing_ops += len(
                    re.findall(r"\d+\.?\d*\s+\d+\.?\d*\s+m", content)
                )  # moveto
                drawing_ops += len(
                    re.findall(r"\d+\.?\d*\s+\d+\.?\d*\s+l", content)
                )  # lineto
                drawing_ops += len(
                    re.findall(
                        r"\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*\s+re", content
                    )
                )  # rectangle

                if drawing_ops > 50:  # Significant drawing content
                    print(f"  Page {page_num + 1}: {drawing_ops} drawing operations")
                    pages_with_drawings += 1
        except Exception as e:
            print(f"  Page {page_num + 1}: Error - {e}")

    print(f"  Pages with significant drawing operations: {pages_with_drawings}")

    # Method 5: Resource analysis
    print("\n📊 Method 5: Page Resources Analysis")
    for page_num in range(min(5, len(doc))):
        page = doc[page_num]
        try:
            # Get page object
            page_obj = page.get_contents()
            if page_obj:
                # Look at the raw content
                content = page_obj[0].get_buffer().decode("latin-1", errors="ignore")

                # Look for resource references
                resource_patterns = [r"/XObject", r"/Image", r"/Form", r"/Pattern"]

                resources_found = []
                for pattern in resource_patterns:
                    if pattern in content:
                        resources_found.append(pattern)

                if resources_found:
                    print(f"  Page {page_num + 1}: Resources found: {resources_found}")

        except Exception as e:
            print(f"  Page {page_num + 1}: Error - {e}")

    doc.close()

    print(f"\n📋 Summary:")
    print(f"  - Standard images detected: {total_standard}")
    print(f"  - XObject references: {total_xobjects}")
    print(f"  - Figure mentions in text: {len(unique_figures)}")
    print(f"  - Pages with drawings: {pages_with_drawings}")

    if len(unique_figures) > 0 and total_standard == 0:
        print(
            f"\n💡 Conclusion: The PDF likely contains {len(unique_figures)} figures/charts"
        )
        print(f"   but they are embedded as vector graphics or text-based content,")
        print(f"   not as raster images that PyMuPDF can easily detect.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_images.py <pdf_path>")
        sys.exit(1)

    debug_pdf_images(sys.argv[1])
