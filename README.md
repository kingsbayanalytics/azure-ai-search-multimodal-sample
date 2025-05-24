# Azure AI Search Multimodal RAG Demo
[![Open in GitHub Codespaces](https://img.shields.io/static/v1?style=for-the-badge&label=GitHub+Codespaces&message=Open&color=brightgreen&logo=github)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=984945158&skip_quickstart=true)

## Table of Contents

- [Azure AI Search Multimodal RAG Demo](#azure-ai-search-multimodal-rag-demo)
- [Azure AI Search Portal: Bring your own index and resources](#azure-ai-search-portal-bring-your-own-index-and-resources)
- [Getting Started](#getting-started)
   - [General Requirements](#general-requirements)
- [Environment setup](#environment-setup)
   - [Github codespaces](#github-codespaces)
   - [Local development setup (Windows or Linux)](#local-development-setup-windows-or-linux)
   - [Provision resources and deploy working app](#provision-resources-and-deploy-working-app)
   - [Debug app locally](#debug-app-locally)
   - [Bring your own data (supports .pdf only)](#bring-your-own-data-supports-pdf-only)
- [Processing New Documents and Creating Additional Indexes](#processing-new-documents-and-creating-additional-indexes)
   - [Document Placement and Preparation](#document-placement-and-preparation)
   - [Index Creation Process](#index-creation-process)
   - [Preprocessing Pipeline Details](#preprocessing-pipeline-details)
   - [Cost Estimation for Document Processing](#cost-estimation-for-document-processing)
   - [Troubleshooting Document Processing](#troubleshooting-document-processing)
- [Azure Services Used for Deployment](#azure-services-used-for-deployment)
   - [Role Mapping for the Application](#role-mapping-for-the-application)
- [End-to-end app diagram](#end-to-end-app-diagram)
- [Troubleshooting](#troubleshooting)



Welcome to the **Azure AI Search Multimodal RAG Demo**. This repository contains the code for an application designed to showcase [multimodal](https://aka.ms/azs-multimodal) [Retrieval-Augmented Generation (RAG)](https://learn.microsoft.com/azure/search/retrieval-augmented-generation-overview) techniques using [Azure AI Search](https://learn.microsoft.com/azure/search/search-what-is-azure-search). This demo combines AI capabilities to create custom copilots / RAG applications that can query, retrieve, and reason over both text and image data.

With multimodal RAG, you can:

+ Extract relevant information from documents, screenshots, and visuals (like diagrams, charts, workflows, etc.).
+ Preserve and understand the relationships between entities in complex images to enable reasoning over structured content.
+ Generate grounded, accurate responses using Large Language Models (LLMs), integrating insights from both textual and visual modalities.

This demo is intentionally kept lean and simple, providing a hands-on experience with multimodal AI techniques. While not intended for production use, it serves as a powerful starting point for exploring how multimodal RAG can unlock new possibilities in building smarter, more context-aware applications.

Note that currently this sample doesn't have support for table extraction as a structure, but tables are extracted as plain text.

![image](docs/images/sample_snap_1.jpg) 
**Text citations**![image](docs/images/sample_snap_3.jpg) ![image](docs/images/sample_snap_2.jpg) 
**Image citations**![image](docs/images/image-cite-1.jpg) ![image](docs/images/image-cite-2.jpg) 

## Azure AI Search Portal: Bring your own index and resources
You can create an index using the AI Search portal's quick wizard for the multimodal scenario. Once the index is successfully created, you can integrate it with the app by running the following steps:

- Checkout a [code space](#azure-ai-search-multimodal-rag-demo) based on **main** branch
- Run ```az login --use-device-code```
- Run 
   ```pwsh
   scripts/portal-2-app.ps1 `
        -SearchIndexName "my-index" `
        -SearchServiceEndpoint "https://myservice.search.windows.net" `
        -StorageAccountUrl "https://myaccount.blob.core.windows.net" `
        -KnowledgeStoreContainerName "knowledgestore-artifacts" `
        -DataSourcesContainerName "data-sources" `
        -AzureOpenAiEndpoint "https://myopenai.openai.azure.com" `
        -AzureOpenAiDeploymentName "my-deployment" `
        -AzureOpenAiEndpointChatCompletionModelName "gpt-4o"
   ```

   Replace the placeholders (`<...>`) with your specific values. This script will configure the app to use the newly created index.  
   **Assumption**: For app simplicity, ensure 'KnowledgeStoreContainerName' and 'DataSourcesContainerName' must be from same storage account.
- Ensure your Azure Entra ID user object ID has been granted the necessary permissions for all required resources. See [Role Mapping for the Application](#role-mapping-for-the-application) for details.
- Run:
   ```bash
      src/start.sh
   ```

## Getting Started

### General Requirements  
To deploy and run this application, you will need the following:  
  
1. **Azure Account**  
   - If you're new to Azure, you can [sign up for a free Azure account](https://azure.microsoft.com/free) and receive some free credits to get started.   
   - Follow the guide to deploy using the free trial if applicable.  
  
2. **Azure Account Permissions**  
   - Your Azure account must have sufficient permissions to perform deployments. Specifically, you need:  
     - `Microsoft.Authorization/roleAssignments/write` permissions, such as those granted by the **Role Based Access Control (RBAC) Administrator**, **User Access Administrator**, or **Owner** roles.  
     - **Subscription-level permissions**. Alternatively, if you don't have subscription-level permissions, you must be granted RBAC access for an existing resource group where you'll deploy the application.  
     - `Microsoft.Resources/deployments/write` permissions at the subscription level.  
  
3. **Local Deployment Environment (Optional)**  
   - If a local deployment of the application is required, ensure you have one of the following operating systems set up:  
     - **Windows OS**  
     - **Linux OS**  
---  

## Environment setup

### Github codespaces
- Checkout a [code space](#azure-ai-search-multimodal-rag-demo) based on **main** branch

### Local development setup (Windows or Linux)
Install the below tools
- [Python 3.12.7](https://www.python.org/downloads/release/python-3127/)
- [Node.js > v.18](https://nodejs.org/)
- [az cli latest](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows?pivots=winget)
- [azd latest](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd?tabs=winget-windows%2Cbrew-mac%2Cscript-linux&pivots=os-windows)
- [Powershell 7 (Windows and Linux)](https://github.com/powershell/powershell)

### Provision resources and deploy working app
- Run below commands (One time setup per environment)
  - Run ```az login --use-device-code```
  - Run ```azd auth login```
  - Run ```azd env new <YOUR_ENVIRONMENT_NAME>```
  - Run ```azd env set AZURE_PRINCIPAL_ID  <USER_OBJECT_ID>``` (This needs to user's object ID from Azure Entra ID. Alternate you can use command from your local development box ```az ad signed-in-user show --query id -o tsv``` )
  - Run ```azd up```. This command will
    - Provision the azure resources
    - Package the application
    - Injest data into azure search index
    - Deploy the working app to webApp services
  - NOTE: You might encounter provisioning errors on cohere. Please visit troubleshooting section for more details.
  - Once deployment succeeds, you can use the app.
  
!['Output from running azd up'](docs/images/app_depl_success.png)

NOTE: It may take 5-10 minutes after you see 'SUCCESS' for the application to be fully deployed. If you see a "Python Developer" welcome screen or an error page, then wait a bit and refresh the page.


### Debug app locally
- You need to ***provision all the resources*** before your start to debug app locally
- To launch the app locally, run the below command. The website will open automatically and be served at [localhost:5000](http://localhost:5000).

- **On Windows:**
   ```powershell
   src/start.ps1
   ```

- **On Linux:**
   ```bash
   src/start.sh
   ```

### Bring your own data (supports .pdf only)
- To index your own data,
   - Place pdf's under ```/data``` folder
   - Run ```scripts\prepdocs.ps1```
- You could also use different indexer strategies **["indexer-image-verbal", "self-multimodal-embedding"]**
- To create new index with a different strategy
  - Run ```azd set SEARCH_INDEX_NAME <new-index-name>```
  - **On Windows** Run ```scripts\prepdocs.ps1 -IndexerStrategy indexer-image-verbal ```
  - **On Linux** Run ```scripts\prepdocs.sh indexer-image-verbal ```

---

## Processing New Documents and Creating Additional Indexes

This section provides comprehensive instructions for adding new documents (such as book PDFs) to your multimodal RAG system and creating specialized indexes for different document collections.

### Document Placement and Preparation

#### 1. Document Location Structure
```
azure-ai-search-multimodal-sample/
├── data/                          # Main document directory
│   ├── books/                     # Organize by document type (optional)
│   │   ├── technical-manual.pdf
│   │   └── research-paper.pdf
│   ├── reports/
│   └── general/
│       └── your-new-book.pdf
```

#### 2. Document Requirements and Specifications

**Supported Formats:**
- **PDF only** (current limitation)
- **Maximum file size**: 500 MB per document (Azure Document Intelligence limit)
- **Page limit**: 2,000 pages per document (recommended for cost optimization)

**Document Quality Guidelines:**
- **Text clarity**: Ensure text is readable and not heavily distorted
- **Image quality**: Minimum 150 DPI for embedded images/diagrams
- **Language support**: Primarily English, with limited support for other languages
- **Structure**: Documents with clear headings and consistent formatting process better

#### 3. Pre-Processing Checklist

Before placing documents in the `/data` folder:

1. **Verify PDF integrity:**
   ```bash
   # Test if PDF opens without errors
   python -c "
   import fitz  # PyMuPDF
   doc = fitz.open('path/to/your-book.pdf')
   print(f'Pages: {len(doc)}')
   doc.close()
   "
   ```

2. **Check document size and estimated costs:**
   ```bash
   # Get file size and page count
   ls -lh data/your-book.pdf
   python -c "
   import fitz
   doc = fitz.open('data/your-book.pdf')
   pages = len(doc)
   file_size_mb = os.path.getsize('data/your-book.pdf') / (1024*1024)
   estimated_cost = pages * 0.0015  # $1.50 per 1000 pages
   print(f'Pages: {pages}, Size: {file_size_mb:.1f}MB, Est. Processing Cost: ${estimated_cost:.2f}')
   "
   ```

### Index Creation Process

#### 1. Creating a New Index for Book Collection

**Step 1: Set Environment Variables**
```bash
# Set a descriptive index name
azd env set SEARCH_INDEX_NAME "book-collection-index"

# Optional: Set custom knowledge agent name
azd env set KNOWLEDGE_AGENT_NAME "book-collection-agent"
```

**Step 2: Place Documents**
```bash
# Create organized structure
mkdir -p data/books
cp /path/to/your-book.pdf data/books/

# Verify placement
ls -la data/books/
```

**Step 3: Execute Document Processing**

**On Windows:**
```powershell
# Basic processing with default strategy
scripts\prepdocs.ps1

# Or with specific indexer strategy for better image processing
scripts\prepdocs.ps1 -IndexerStrategy "indexer-image-verbal"

# For documents with lots of images, use multimodal embedding strategy
scripts\prepdocs.ps1 -IndexerStrategy "self-multimodal-embedding"
```

**On Linux:**
```bash
# Basic processing
scripts/prepdocs.sh

# With specific strategy
scripts/prepdocs.sh indexer-image-verbal

# For image-heavy documents
scripts/prepdocs.sh self-multimodal-embedding
```

#### 2. Understanding Indexer Strategies

**`indexer-image-verbal` (Recommended for most use cases):**
- Uses Azure AI Document Intelligence for layout analysis
- Generates verbal descriptions of images using GPT-4o
- Best for: Technical documents, reports with diagrams, mixed content
- Processing time: ~2-3 minutes per page
- Cost: Higher due to GPT-4o image processing

**`self-multimodal-embedding` (For image-heavy documents):**
- Creates both text and image embeddings
- Preserves visual context for complex diagrams
- Best for: Scientific papers, infographics, visual-heavy content
- Processing time: ~1-2 minutes per page
- Cost: Moderate, uses embedding models

**Default Strategy:**
- Basic text extraction with simple image handling
- Fastest processing
- Best for: Text-heavy documents with minimal images
- Processing time: ~30 seconds per page
- Cost: Lowest

### Preprocessing Pipeline Details

#### 1. Document Processing Workflow

The preprocessing pipeline (`src/backend/processfile.py`) performs these steps:

```
Document Input → Document Intelligence Analysis → Content Extraction → Embedding Generation → Index Upload
```

**Detailed Process Flow:**

1. **Document Intelligence Analysis** (`_process_pdf` method):
   ```python
   # Key processing parameters from processfile.py
   analyze_request = AnalyzeDocumentRequest(
       url_source=None,
       output_content_format="markdown",  # Preserves structure
       output_option=[AnalyzeOutputOption.FIGURES]  # Extracts images
   )
   ```

2. **Content Chunking** (`_chunk_text_with_metadata` method):
   - **Chunk size**: ~1000 characters with overlap
   - **Metadata preservation**: Page numbers, bounding polygons
   - **Structure retention**: Headings, paragraphs, lists

3. **Image Processing** (`_extract_figures` method):
   - **Image extraction**: Converts to base64 for analysis
   - **Visual description**: GPT-4o generates descriptions
   - **Embedding creation**: Cohere multimodal embeddings

4. **Index Schema Creation**:
   ```python
   # Core fields created for each document
   fields = [
       SimpleField(name="content_id", type=SearchFieldDataType.String, key=True),
       SearchableField(name="content_text", type=SearchFieldDataType.String),
       SearchField(name="content_embedding", vector_search_dimensions=1024),
       ComplexField(name="locationMetadata", fields=[
           SimpleField(name="pageNumber", type=SearchFieldDataType.Int32),
           SimpleField(name="boundingPolygons", type=SearchFieldDataType.String)
       ])
   ]
   ```

#### 2. Monitoring Processing Progress

**Real-time Monitoring:**
```bash
# Watch processing logs
tail -f /tmp/document_processing.log

# Monitor Azure costs during processing
az consumption usage list --start-date $(date -d '1 day ago' +%Y-%m-%d) --end-date $(date +%Y-%m-%d)
```

**Processing Status Indicators:**
- **Document Analysis Phase**: "Analyzing document with Document Intelligence..."
- **Image Extraction Phase**: "Extracting figures from document..."
- **Embedding Generation Phase**: "Generating embeddings for content..."
- **Index Upload Phase**: "Uploading documents to search index..."

#### 3. Customizing Processing for Specific Document Types

**For Academic Papers/Research Documents:**
```powershell
# Use image-verbal strategy with custom parameters
scripts\prepdocs.ps1 -IndexerStrategy "indexer-image-verbal" -ChunkSize 1500 -ChunkOverlap 200
```

**For Technical Manuals:**
```powershell
# Emphasize structure preservation
scripts\prepdocs.ps1 -IndexerStrategy "self-multimodal-embedding" -PreserveStructure $true
```

**For Legal Documents:**
```powershell
# Focus on precise text extraction
scripts\prepdocs.ps1 -IndexerStrategy "default" -HighPrecisionText $true
```

### Cost Estimation for Document Processing

#### Per-Document Processing Costs

**Small Document (10-20 pages):**
- Document Intelligence: $0.015 - $0.030
- GPT-4o Image Processing: $0.050 - $0.100 (if using image-verbal)
- Text Embeddings: $0.020 - $0.040
- Image Embeddings: $0.030 - $0.060 (if using multimodal)
- **Total: $0.115 - $0.230 per document**

**Medium Document (50-100 pages):**
- Document Intelligence: $0.075 - $0.150
- GPT-4o Image Processing: $0.250 - $0.500
- Text Embeddings: $0.100 - $0.200
- Image Embeddings: $0.150 - $0.300
- **Total: $0.575 - $1.150 per document**

**Large Document (200+ pages, like a book):**
- Document Intelligence: $0.300 - $0.600
- GPT-4o Image Processing: $1.000 - $2.000
- Text Embeddings: $0.400 - $0.800
- Image Embeddings: $0.600 - $1.200
- **Total: $2.300 - $4.600 per document**

#### Monthly Processing Volume Estimates

**Research Team (20 documents/month):**
- Average cost per document: $0.50
- Monthly processing cost: $10
- Annual cost: $120

**Enterprise Department (100 documents/month):**
- Average cost per document: $0.75
- Monthly processing cost: $75
- Annual cost: $900

**Large Organization (500 documents/month):**
- Average cost per document: $1.00
- Monthly processing cost: $500
- Annual cost: $6,000

#### Cost Optimization Strategies

1. **Batch Processing**: Process multiple documents together to reduce overhead
2. **Strategy Selection**: Use simpler strategies for text-heavy documents
3. **Document Preprocessing**: Remove unnecessary pages before processing
4. **Incremental Updates**: Only reprocess changed documents

### Troubleshooting Document Processing

#### Common Issues and Solutions

**1. "Document Intelligence Analysis Failed"**
```bash
# Check document integrity
python -c "
import fitz
try:
    doc = fitz.open('data/your-book.pdf')
    print(f'Document OK: {len(doc)} pages')
except:
    print('Document corrupted or unreadable')
"

# Solution: Re-download or repair PDF
```

**2. "Embedding Generation Timeout"**
```bash
# Check if document is too large
ls -lh data/your-book.pdf

# Solution: Split large documents or increase timeout
# Edit src/backend/processfile.py, line ~380:
# self.text_model = EmbeddingsClient(..., timeout=120)  # Increase timeout
```

**3. "Index Upload Failed - Field Validation Error"**
```bash
# Check for special characters in document content
grep -P '[^\x00-\x7F]' data/your-book.pdf

# Solution: Clean document or update field definitions
```

**4. "Insufficient Azure Credits/Quota"**
```bash
# Check current usage
az consumption usage list --start-date $(date -d '1 month ago' +%Y-%m-%d)

# Solution: Monitor and set up billing alerts
az consumption budget create --resource-group <rg-name> --budget-name "DocumentProcessing" --amount 100
```

#### Processing Performance Optimization

**1. Parallel Processing Configuration:**
```python
# Edit src/backend/processfile.py for concurrent processing
import asyncio
import aiofiles

class ProcessFile:
    def __init__(self, ...):
        self.max_concurrent_docs = 3  # Adjust based on quota limits
        self.semaphore = asyncio.Semaphore(self.max_concurrent_docs)
```

**2. Memory Optimization for Large Documents:**
```python
# Process documents in smaller chunks
CHUNK_SIZE = 50  # Process 50 pages at a time
for i in range(0, total_pages, CHUNK_SIZE):
    chunk_pages = pages[i:i+CHUNK_SIZE]
    await self._process_page_chunk(chunk_pages)
```

**3. Network Optimization:**
```bash
# Increase timeout for slow connections
export AZURE_CLIENT_TIMEOUT=300
export AZURE_RETRY_ATTEMPTS=5
```

#### Validation and Quality Assurance

**1. Post-Processing Validation:**
```bash
# Verify index was created successfully
az search index show --index-name "book-collection-index" --service-name <search-service>

# Check document count
az search index statistics --index-name "book-collection-index" --service-name <search-service>
```

**2. Content Quality Verification:**
```python
# Test search functionality with sample queries
import requests

test_query = {
    "search": "chapter 1 introduction",
    "queryType": "semantic",
    "semanticConfiguration": "semanticconfig",
    "queryLanguage": "en-us",
    "count": 5
}

response = requests.post(f"{search_endpoint}/indexes/{index_name}/docs/search", 
                        json=test_query, headers=headers)
print(f"Search results: {len(response.json()['value'])} documents found")
```

**3. Citation Functionality Test:**
```python
# Verify visual citations are working
test_citations = response.json()['value'][0].get('locationMetadata')
if test_citations and 'boundingPolygons' in test_citations:
    print("✅ Visual citations properly configured")
else:
    print("❌ Visual citations missing - check processing strategy")
```

#### Recovery and Rollback Procedures

**1. Index Recovery:**
```bash
# Backup current index definition
az search index show --index-name "book-collection-index" > backup-index-schema.json

# Delete corrupted index
az search index delete --index-name "book-collection-index" --yes

# Recreate from backup
az search index create --index-definition backup-index-schema.json
```

**2. Document Reprocessing:**
```bash
# Reprocess specific failed documents
scripts/prepdocs.ps1 -DocumentPath "data/books/specific-book.pdf" -ForceReprocess $true

# Clean and restart full processing
rm -rf data/.processing_cache/
scripts/prepdocs.ps1 -IndexerStrategy "indexer-image-verbal"
```

**3. Cost Management Recovery:**
```bash
# Set emergency budget alerts
az consumption budget create \
    --budget-name "EmergencyStop" \
    --amount 50 \
    --time-grain "Monthly" \
    --notifications '[{
        "enabled": true,
        "operator": "GreaterThan",
        "threshold": 80,
        "contactEmails": ["admin@company.com"]
    }]'
```

---

## Azure Services Used for Deployment  
The following Azure services are used as part of this deployment. Ensure you verify their billing and pricing details as part of the setup:  
  
1. **Azure AI Search**  
   - Service used for search functionalities within the application. Review [pricing](https://azure.microsoft.com/pricing/details/search/).
  
2. **Azure AI Document Intelligence**  
   - Service used for processing and extracting information from documents. Review [pricing](https://azure.microsoft.com/pricing/details/ai-document-intelligence/). 
  
3. Your provided:
   - **LLM Deployment**: For running the large language model (LLM) for verbalization and used by the RAG orchestrator. 
   - **Embedding Model Deployment**: Used for creating embeddings for vector search and other tasks.   
   - Ensure you check the pricing for both LLM and embedding deployments.
   - This sample currently supports gpt-4o, (AOAI) text-embedding-large, cohere-serverless-v3  
  
4. **Azure Blob Storage Account**  
   - Used to store extracted images and other data. Verify the pricing for storage and associated operations. Review [pricing](https://azure.microsoft.com/pricing/details/storage/blobs/).
  
5. **Azure App Service**  
   - Used to host and run the application in the cloud. Review [pricing](https://azure.microsoft.com/pricing/details/app-service/windows/). 

### Role Mapping for the Application  
The following table maps the roles used by the application to their respective functions:  
  
| **Role ID**                              | **Built-in Role Name**                  | **Purpose**                                                                                     |  
|------------------------------------------|------------------------------------------|-------------------------------------------------------------------------------------------------|  
| `5e0bd9bd-7b93-4f28-af87-19fc36ad61bd`   | **Cognitive Services OpenAI User**       | Read-only access to models, files, and deployments in an Azure OpenAI resource. Allows running completion/embedding/image-generation calls. |  
| `a97b65f3-24c7-4388-baec-2e87135dc908`   | **Cognitive Services User**              | Provides read access to an Azure Cognitive Services resource and the ability to list its access keys. (No write or manage permissions.) |  
| `ba92f5b4-2d11-453d-a403-e96b0029c9fe`   | **Storage Blob Data Contributor**        | Allows read, upload, modify, and delete operations on blobs and containers within an Azure Storage account (data-plane only). |  
| `7ca78c08-252a-4471-8644-bb5ff32d4ba0`   | **Search Service Contributor**           | Enables management of the Azure Cognitive Search service (e.g., create, scale, delete). Does not provide access to index data itself. |  
| `8ebe5a00-799e-43f5-93ac-243d3dce84a7`   | **Search Index Data Contributor**        | Provides full create, read, update, and delete access to all
| `64702f94-c441-49e6-a78b-ef80e0188fee`   | **Azure AI Developer**                   | Provides full create, read access to AI foundry projects.

## End-to-end app diagram

![image](https://github.com/user-attachments/assets/5984f2b7-e0d9-4d2c-a652-9a7b10085b79)

## Troubleshooting
- What is the region availability for Azure OpenAI service?  
  Please visit [available regions](https://learn.microsoft.com/azure/ai-services/openai/concepts/models?tabs=global-standard%2Cstandard-chat-completions#global-standard-model-availability)
- What is the region availability for Cohere Serverless?    
  Cohere serverless is supported only in [5 regions](https://learn.microsoft.com/azure/ai-foundry/how-to/deploy-models-serverless-availability#cohere-models)
- Deployment fails for 'Cohere' in marketplace subscription !['Error from azd up'](docs/images/marketplace_error.png)
  - Ensure your subscription is supported or enabled for Marketplace deployment [Learn more](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/deploy-models-serverless?tabs=azure-ai-studio#prerequisites)
  - There is a known issue of conflict operation between Marketplace subscription and endpoint deployment. **Rerun deployment** to fix it

