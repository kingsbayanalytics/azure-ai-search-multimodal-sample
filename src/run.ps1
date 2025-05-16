# set the parent of the script as the current location.
Set-Location $PSScriptRoot
$directory = Get-Location
$root = "$directory/.."
$backendPath = "$directory/backend"

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue

Write-Host 'Creating python virtual environment ".venv"'

$venvTarget = "$root/.venv"
Start-Process -FilePath ($pythonCmd).Source -ArgumentList "-m venv $venvTarget" -Wait -NoNewWindow

if ($IsWindows) {
    $venvPythonPath = "$root/.venv/scripts/python.exe"
} else {
    $venvPythonPath = "$root/.venv/bin/python"
}


Write-Host ""
Write-Host "Starting app"
Write-Host ""
Set-Location $backendPath
Start-Process http://localhost:5000
Start-Process -FilePath $venvPythonPath -ArgumentList "app.py" -Wait -NoNewWindow
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start backend"
    exit $LASTEXITCODE
}

