# Local .env Setup for Development

## Scenario 1: Starting Fresh (Prerequisites)

**If you haven't created an azd environment yet:**

1. **Login to Azure:**
   ```bash
   az login --use-device-code
   azd auth login
   ```

2. **Create an azd environment:**
   ```bash
   azd env new <YOUR_ENVIRONMENT_NAME>
   # Example: azd env new my-multimodal-env
   ```

3. **Set your Azure Principal ID:**
   ```bash
   # Get your user object ID
   az ad signed-in-user show --query id -o tsv
   
   # Set it in azd environment
   azd env set AZURE_PRINCIPAL_ID <USER_OBJECT_ID>
   ```

4. **Deploy to Azure:**
   ```bash
   azd up
   ```

## Scenario 2: You Already Have an azd Environment (Your Situation)

**If you created the environment in Azure Cloud Shell but now want to deploy from local:**

1. **Make sure you're in your local project directory:**
   ```bash
   cd /Users/mikewarren/azure-ai-search-multimodal-sample
   ```

2. **Login to Azure locally:**
   ```bash
   az login --use-device-code
   azd auth login
   ```

3. **Connect to your existing environment:**
   ```bash
   # List available environments to see yours
   azd env list
   
   # Select your existing environment
   azd env select <your-environment-name>
   ```

4. **Deploy the actual application from local code:**
   ```bash
   azd up
   ```
   This time it will deploy your local code to the existing Azure resources.

5. **Extract environment variables after deployment:**
   ```bash
   azd env get-values > src/backend/.env
   ```

## Method 1: Copy from Azure deployment (Recommended)

**After completing either scenario above:**

1. Extract environment variables:
   ```bash
   azd env get-values > src/backend/.env
   ```

2. Or view the values first:
   ```bash
   azd env get-values --output json
   ```

## Method 2: Manual local .env file

Create `src/backend/.env` with your Azure resource endpoints:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_MODEL_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Azure AI Search Configuration  
SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
SEARCH_INDEX_NAME=state-of-ai

# Azure AI Document Intelligence
DOCUMENTINTELLIGENCE_ENDPOINT=https://your-doc-intelligence.cognitiveservices.azure.com/

# Azure AI Inference (Cohere Embeddings)
AZURE_INFERENCE_EMBED_ENDPOINT=https://your-inference-endpoint.cognitiveservices.azure.com/
AZURE_INFERENCE_EMBED_MODEL_NAME=Cohere-embed-v3-multilingual

# Azure Storage Account
ARTIFACTS_STORAGE_ACCOUNT_URL=https://yourstorageaccount.blob.core.windows.net/
ARTIFACTS_STORAGE_CONTAINER=mm-knowledgestore-artifacts
SAMPLES_STORAGE_CONTAINER=mm-sample-docs-container

# Knowledge Agent Configuration
KNOWLEDGE_AGENT_NAME=state-of-ai-knowledge-agent

# Azure Environment Configuration
AZURE_ENV_NAME=your-environment-name
AZURE_LOCATION=your-azure-region
AZURE_PRINCIPAL_ID=your-user-object-id
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group-name

# Optional App Configuration
HOST=localhost
PORT=5000
```

## Troubleshooting

**Error: "environment not specified"**
- Make sure you've run `azd env select <name>` to connect to your existing environment
- Check you're in the project root directory (where `azure.yaml` exists)
- Verify with `azd env list` to see available environments

**Error: "No azd environment found"**
- The environment hasn't been created yet
- Run Scenario 1 above

**Blank resource group after Cloud Shell deployment:**
- This is normal - Cloud Shell created the environment but didn't deploy the code
- Follow Scenario 2 to deploy the actual application from your local machine

## Why you need this:

- The VS Code launch configuration expects `.env` in `src/backend/`
- Local Python development needs direct access to environment variables
- Allows debugging without running through the start.sh script

## Running locally:

With local .env file:
```bash
cd src/backend
python app.py
```

Or use VS Code debugger with the provided launch configurations. 