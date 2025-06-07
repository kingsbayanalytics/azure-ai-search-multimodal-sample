#!/bin/sh
root=$(pwd)
backendPath="$root/src/backend"
azurePath="$root/.azure"
frontendPath="$root/src/frontend"

# find the .env file in the azurePath directory recursively
envFile=$(find $azurePath -type f -name ".env"| head -n 1)

if [ -f "$envFile" ]; then
    echo ".env file found at: $envFile"
else
    echo ".env file not found. Please run azd up and ensure it completes successfully."
    exit 1
fi

# Load azd environment variables
echo 'Loading azd environment variables'
azdEnv=$(azd env get-values --output json)

# Check if the azd command succeeded
if [ $? -ne 0 ]; then
    echo "Failed to load azd environment variables. Ensure azd is installed and configured correctly."
    exit 1
fi

# Parse and export each environment variable in the current shell session
eval $(echo "$azdEnv" | jq -r 'to_entries | .[] | "export \(.key)=\(.value)"')

# Set PORT to 5001 to avoid conflicts with Control Center
export PORT=5001
echo "Setting application to use PORT=$PORT"

echo 'Cleaning up frontend artifacts'
rm -rf $frontendPath/node_modules
rm -rf $frontendPath/dist # Also remove the build output directory for good measure
# Vite's default cache is node_modules/.vite, which is covered by removing node_modules

echo 'Restore and build frontend'
cd $frontendPath
npm install
npm run build

echo 'Build and start backend'
cd $root

echo 'Creating Python virtual environment'
rm -rf .venv # Also ensure the backend venv is fully clean
python3.11 -m venv .venv

echo 'Installing dependencies from "requirements.txt" into virtual environment (in quiet mode)...'
.venv/bin/python -m pip --quiet --disable-pip-version-check install -r src/backend/requirements.txt

echo 'Starting the app'
.venv/bin/python "src/backend/app.py"