# Cost-Effective Document Preprocessing Pipeline

## Overview
This preprocessing pipeline creates Azure Search indexes at 70-85% lower cost than Azure Document Intelligence by using Python libraries (PyMuPDF, PDFPlumber, SentenceTransformers) and intelligent document processing strategies.

## Architecture
- **Separate System**: Independent from the main app, only creates indexes
- **Strategy-Based Processing**: Intelligent selection of processing approaches based on document content
- **Cost Optimization**: Local embeddings, selective OCR, and smart image processing
- **Citation Preservation**: Maintains exact metadata format for visual citations

## Components
- `document_processor.py` - Main processing orchestrator
- `strategy_selector.py` - Document analysis and strategy selection
- `processors/` - Library-specific processing implementations
- `embedding/` - Local and cloud embedding generation
- `indexer/` - Azure Search index creation and management
- `config/` - Configuration management
- `monitoring/` - Cost tracking and quality metrics

## Processing Strategies
1. **text_only** - Pure text documents, local embeddings, skip images
2. **text_optimized** - Text-heavy with basic layout, Mistral OCR, local embeddings  
3. **image_heavy** - Complex visuals, hybrid Mistral+GPT-4o, Azure embeddings
4. **balanced** - Adaptive processing based on content analysis

## Usage
```bash
python document_processor.py --input-path /path/to/docs --strategy auto --output-index my-index
```

## Cost Savings Target
- 70-85% reduction vs Azure Document Intelligence Skills
- Intelligent strategy selection based on document characteristics
- Comprehensive cost tracking and reporting 