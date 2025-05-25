param(
    [string]$IndexerStrategy = 'self-multimodal-embedding',
    [string]$DataPath = ''
)

# Set the parent of the script as the current location.
Set-Location $PSScriptRoot
$directory = Get-Location
$root = "$directory/.."
$backendPath = "$root/src/backend"
$azurePath = "$root/.azure"

# find the .env file in the azurePath directory recursively
$envFile = Get-ChildItem -Path $azurePath -Filter ".env" -Recurse -Force -ErrorAction SilentlyContinue | Select-Object -First 1

if ($envFile) {
    Write-Host ".env file found at: $envFile , $($envFile.FullName)"
} else {
    Write-Host ".env file not found in $azurePath. Please run azd up and ensure it completes successfully."
    exit 1
}

# Load azd environment variables
Write-Host "Loading azd environment variables"
$azdEnv = azd env get-values --output json | ConvertFrom-Json
foreach ($key in $azdEnv.PSObject.Properties.Name) {
    [System.Environment]::SetEnvironmentVariable($key, $azdEnv.$key, [System.EnvironmentVariableTarget]::Process)
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Error "Python is not installed or not found in PATH. Please install Python and try again."
        exit 1
    }
}
Write-Host 'Creating python virtual environment ".venv"'

$venvTarget = "$root/.venv"
Start-Process -FilePath ($pythonCmd).Source -ArgumentList "-m venv $venvTarget" -Wait -NoNewWindow

$venvPythonPath = "$root/.venv/scripts/python.exe"

Write-Host 'Installing dependencies from "requirements.txt" into virtual environment'
Start-Process -FilePath $venvPythonPath -ArgumentList "-m pip install -r $backendPath/requirements.txt" -Wait -NoNewWindow

Write-Host 'Run the document preparation script'
$args_indexer_strategy = $IndexerStrategy

# Build the command with optional data_path parameter
if ($DataPath -ne '') {
    Write-Host "Processing documents from: $DataPath"
    $pythonArgs = "$backendPath/prepdocs.py --source files --indexer_strategy $args_indexer_strategy --data_path `"$DataPath`""
} else {
    Write-Host "Processing all documents in data directory"
    $pythonArgs = "$backendPath/prepdocs.py --source files --indexer_strategy $args_indexer_strategy"
}

Start-Process -FilePath $venvPythonPath -ArgumentList $pythonArgs -Wait -NoNewWindow