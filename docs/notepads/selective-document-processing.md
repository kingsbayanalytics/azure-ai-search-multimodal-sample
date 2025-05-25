# Selective Document Processing Guide

## Overview

The Azure AI Search Multimodal RAG Demo now supports selective document processing, allowing you to process specific folders or files within the `data` directory instead of processing all documents at once. This feature provides better control over processing costs, time, and allows for targeted index creation.

## Key Features

- **Folder Selection**: Process entire folders and their subdirectories
- **File Selection**: Process individual files
- **Recursive Processing**: Automatically finds files in subdirectories
- **Multiple File Types**: Supports PDF, DOCX, DOC, and TXT files
- **Path Flexibility**: Supports both relative and absolute paths
- **Error Handling**: Continues processing even if individual files fail
- **Progress Tracking**: Shows detailed progress and results

## Usage Examples

### Basic Usage

#### Process All Documents (Default Behavior)
```bash
# Windows
scripts\prepdocs.ps1

# Linux
scripts/prepdocs.sh
```

#### Process Specific Folder
```bash
# Windows
scripts\prepdocs.ps1 -DataPath "books"

# Linux  
scripts/prepdocs.sh self-multimodal-embedding "books"
```

#### Process Specific File
```bash
# Windows
scripts\prepdocs.ps1 -DataPath "books/technical-manual.pdf"

# Linux
scripts/prepdocs.sh self-multimodal-embedding "books/technical-manual.pdf"
```

### Advanced Usage

#### Combine with Indexer Strategies
```bash
# Windows - Use image-verbal strategy with specific folder
scripts\prepdocs.ps1 -IndexerStrategy "indexer-image-verbal" -DataPath "reports"

# Linux - Use image-verbal strategy with specific folder
scripts/prepdocs.sh indexer-image-verbal "reports"
```

#### Process Multiple Document Types
```bash
# Process a folder containing mixed document types
scripts/prepdocs.sh self-multimodal-embedding "mixed-documents"
```

## Path Options

### Relative Paths (Recommended)
Relative to the `data/` directory:
- `"books"` - Process all files in data/books/ and subdirectories
- `"reports/quarterly.pdf"` - Process specific file
- `"archive/2024"` - Process all files in data/archive/2024/

### Absolute Paths
Full system paths:
- `"/Users/username/documents/report.pdf"` - Process file outside data directory
- `"/full/path/to/folder"` - Process entire external folder

## Supported File Types

| Extension | Support Level | Notes |
|-----------|---------------|-------|
| `.pdf` | ✅ Full | Recommended format, best results |
| `.docx` | ✅ Good | Microsoft Word documents |
| `.doc` | ✅ Good | Legacy Word documents |
| `.txt` | ✅ Basic | Plain text files |

## Directory Structure Examples

```
data/
├── books/
│   ├── technical-manual.pdf      ← Process with: "books/technical-manual.pdf"
│   ├── research-paper.pdf
│   └── archive/
│       └── old-book.pdf          ← Included when processing "books"
├── reports/
│   ├── quarterly-2024.pdf        ← Process with: "reports/quarterly-2024.pdf"
│   └── annual-report.docx
└── mixed-documents/
    ├── presentation.pdf
    ├── notes.txt
    └── analysis.docx             ← All included when processing "mixed-documents"
```

## Processing Output

### Console Output Example
```
Processing documents from: /path/to/data/books
Found 3 documents to process in: /path/to/data/books
Documents to be processed:
  1. technical-manual.pdf
  2. research-paper.pdf
  3. archive/old-book.pdf

Processing file: /path/to/data/books/technical-manual.pdf
✅ Successfully processed: technical-manual.pdf

Processing file: /path/to/data/books/research-paper.pdf
✅ Successfully processed: research-paper.pdf

Processing file: /path/to/data/books/archive/old-book.pdf
✅ Successfully processed: old-book.pdf
```

### Error Handling
```
Processing file: /path/to/data/books/corrupted.pdf
❌ Error processing /path/to/data/books/corrupted.pdf: Invalid PDF format

Processing continues with next file...
```

## Cost Optimization Benefits

### Selective Processing Advantages
1. **Reduced Processing Costs**: Only process documents you need
2. **Faster Iteration**: Test with small document sets first
3. **Targeted Indexes**: Create specialized indexes for different document types
4. **Resource Management**: Better control over Azure service usage

### Cost Estimation Examples
```bash
# Small test run (5 documents)
scripts/prepdocs.sh self-multimodal-embedding "test-documents"
# Estimated cost: $0.50 - $2.00

# Full book collection (50 documents)  
scripts/prepdocs.sh indexer-image-verbal "books"
# Estimated cost: $25.00 - $100.00

# Single large document
scripts/prepdocs.sh self-multimodal-embedding "reports/annual-report-500pages.pdf"
# Estimated cost: $2.00 - $5.00
```

## Best Practices

### 1. Start Small
```bash
# Test with a small subset first
scripts/prepdocs.sh self-multimodal-embedding "test-folder"
```

### 2. Organize by Document Type
```
data/
├── technical-docs/     ← Use indexer-image-verbal
├── text-heavy/         ← Use self-multimodal-embedding  
└── mixed-content/      ← Use indexer-image-verbal
```

### 3. Use Appropriate Strategies
- **indexer-image-verbal**: For documents with diagrams, charts, images
- **self-multimodal-embedding**: For text-heavy documents, cost-sensitive processing

### 4. Monitor Processing
- Check console output for errors
- Verify document counts match expectations
- Monitor Azure costs during processing

## Troubleshooting

### Common Issues

#### "Path not found" Error
```bash
# Check if path exists
ls -la data/your-folder

# Use correct relative path
scripts/prepdocs.sh self-multimodal-embedding "correct-folder-name"
```

#### No Documents Found
```bash
# Check for supported file types
find data/your-folder -name "*.pdf" -o -name "*.docx" -o -name "*.doc" -o -name "*.txt"

# Verify files exist
ls -la data/your-folder/
```

#### Processing Failures
- Check document integrity (corrupted files)
- Verify Azure service quotas
- Monitor network connectivity
- Check available disk space

### Performance Tips

1. **Batch Similar Documents**: Process documents of similar type together
2. **Use Appropriate Hardware**: Ensure sufficient RAM for large documents
3. **Monitor Network**: Stable internet connection required for Azure services
4. **Check Quotas**: Verify Azure service limits before large processing runs

## Integration with Existing Workflows

### CI/CD Pipeline Integration
```yaml
# Example GitHub Actions workflow
- name: Process New Documents
  run: |
    scripts/prepdocs.sh indexer-image-verbal "new-documents"
```

### Automated Processing
```bash
#!/bin/bash
# Process different document types with appropriate strategies
scripts/prepdocs.sh indexer-image-verbal "technical-docs"
scripts/prepdocs.sh self-multimodal-embedding "text-documents"
scripts/prepdocs.sh indexer-image-verbal "mixed-content"
```

## Future Enhancements

Planned improvements for selective document processing:
- File filtering by date/size
- Batch processing with different strategies
- Progress persistence and resume capability
- Integration with document management systems
- Automated strategy selection based on content analysis 