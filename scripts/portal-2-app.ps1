<#
.SYNOPSIS
    Configures the application with search index and storage parameters using Azure AI Search portal's multimodal wizard.
    This will also update the web app settings if provided.

.DESCRIPTION
    This script sets up application parameters including search index name, search service endpoint,
    storage account URL, and knowledge store container name.

.PARAMETER SearchIndexName
    Name of the Azure AI Search index to use.

.PARAMETER SearchServiceEndpoint
    Endpoint URL for the Azure AI Search service.

.PARAMETER StorageAccountUrl
    URL of the Azure Storage account.

.PARAMETER KnowledgeStoreContainerName
    Name of the container in Azure Storage that holds knowledge store artifacts.

.PARAMETER DataSourcesContainerName
    Name of the container in Azure Storage that holds your data.
.EXAMPLE
    .\portal-2-app.ps1 -SearchIndexName "my-index" -SearchServiceEndpoint "https://myservice.search.windows.net" -StorageAccountUrl "https://myaccount.blob.core.windows.net" -KnowledgeStoreContainerName "mm-knowledgestore-artifacts" -DataSourcesContainerName "mm-data-sources" -WebAppName "mywebapp" -ResourceGroupName "myResourceGroup" -SubscriptionId "mySubscriptionId"
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SearchIndexName,

    [Parameter(Mandatory = $true)]
    [string]$SearchServiceEndpoint,

    [Parameter(Mandatory = $true)]
    [string]$StorageAccountUrl,

    [Parameter(Mandatory = $true)]
    [string]$KnowledgeStoreContainerName,

    [Parameter(Mandatory = $true)]
    [string]$DataSourcesContainerName,

     [Parameter(Mandatory = $false)]
    [string]$WebAppName,

    [Parameter(Mandatory = $false)]
    [string]$ResourceGroupName,

    [Parameter(Mandatory = $false)]
    [string]$SubscriptionId
)

# Set AZD Environment Variables
azd env set SEARCH_INDEX_NAME $SearchIndexName
azd env set SEARCH_SERVICE_ENDPOINT $SearchServiceEndpoint
azd env set ARTIFACTS_STORAGE_ACCOUNT_URL $StorageAccountUrl
azd env set ARTIFACTS_STORAGE_CONTAINER $KnowledgeStoreContainerName
azd env set SAMPLES_STORAGE_CONTAINER $DataSourcesContainerName

# Update web app settings and restart if web app parameters are provided
if ($WebAppName -and $ResourceGroupName -and $SubscriptionId) {
    Write-Host "Updating web app configuration settings..."
    
    # Update web app settings
    az webapp config appsettings set --name $WebAppName --subscription $SubscriptionId  --resource-group $ResourceGroupName --settings `
        SEARCH_INDEX_NAME=$SearchIndexName `
        SEARCH_SERVICE_ENDPOINT=$SearchServiceEndpoint `
        ARTIFACTS_STORAGE_ACCOUNT_URL=$StorageAccountUrl `
        ARTIFACTS_STORAGE_CONTAINER=$KnowledgeStoreContainerName `
        SAMPLES_STORAGE_CONTAINER=$DataSourcesContainerName `

    # Restart the web app
    Write-Host "Restarting web app: $WebAppName..."
    az webapp restart --name $WebAppName --resource-group $ResourceGroupName --subscription $SubscriptionId
    
    Write-Host "Web app configuration updated and restarted successfully."
}