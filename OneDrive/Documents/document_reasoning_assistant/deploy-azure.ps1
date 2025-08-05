# Azure Deployment Script for Document Reasoning Assistant
# Run this script after installing Azure CLI and logging in

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName,
    
    [Parameter(Mandatory=$true)]
    [string]$AppName,
    
    [string]$Location = "East US"
)

Write-Host "Starting Azure deployment..." -ForegroundColor Green

# Login to Azure (if not already logged in)
Write-Host "Checking Azure login status..."
$loginStatus = az account show 2>$null
if (!$loginStatus) {
    Write-Host "Please log in to Azure..."
    az login
}

# Create resource group
Write-Host "Creating resource group: $ResourceGroupName"
az group create --name $ResourceGroupName --location $Location

# Create App Service Plan (Free tier for testing)
Write-Host "Creating App Service Plan..."
az appservice plan create --name "$AppName-plan" --resource-group $ResourceGroupName --sku F1 --is-linux

# Create Web App
Write-Host "Creating Web App: $AppName"
az webapp create --resource-group $ResourceGroupName --plan "$AppName-plan" --name $AppName --runtime "PYTHON|3.9"

# Configure startup command
Write-Host "Configuring startup command..."
az webapp config set --resource-group $ResourceGroupName --name $AppName --startup-file "uvicorn app.main:app --host 0.0.0.0 --port 8000"

# Deploy code
Write-Host "Deploying application code..."
Compress-Archive -Path .\* -DestinationPath .\app.zip -Force
az webapp deployment source config-zip --resource-group $ResourceGroupName --name $AppName --src .\app.zip

# Configure environment variables (you'll need to set these manually in Azure Portal)
Write-Host "Setting up environment variables..."
Write-Host "IMPORTANT: You need to set the following environment variables in Azure Portal:" -ForegroundColor Yellow
Write-Host "- GROQ_API_KEY" -ForegroundColor Yellow
Write-Host "- PINECONE_API_KEY" -ForegroundColor Yellow
Write-Host "- PINECONE_INDEX_NAME" -ForegroundColor Yellow
Write-Host "- LLM_PROVIDER=groq" -ForegroundColor Yellow
Write-Host "- MODEL_NAME=llama3-70b-8192" -ForegroundColor Yellow
Write-Host "- EMBEDDING_MODEL=BAAI/bge-small-en-v1.5" -ForegroundColor Yellow

# Clean up
Remove-Item .\app.zip -Force

Write-Host "Deployment completed!" -ForegroundColor Green
Write-Host "Your app URL: https://$AppName.azurewebsites.net" -ForegroundColor Cyan
Write-Host "Don't forget to configure environment variables in Azure Portal!" -ForegroundColor Yellow
