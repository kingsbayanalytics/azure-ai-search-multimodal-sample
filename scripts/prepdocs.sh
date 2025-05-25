#!/bin/sh

# Function to display usage
show_usage() {
    echo "Usage: $0 [indexer_strategy] [data_path]"
    echo ""
    echo "Parameters:"
    echo "  indexer_strategy  Optional. Choose from: 'indexer-image-verbal', 'self-multimodal-embedding'"
    echo "                    Default: 'self-multimodal-embedding'"
    echo "  data_path         Optional. Path to specific folder or file within data directory"
    echo "                    If not provided, processes all documents in data directory"
    echo ""
    echo "Examples:"
    echo "  $0                                                    # Process all documents"
    echo "  $0 indexer-image-verbal                              # Process all with specific strategy"
    echo "  $0 self-multimodal-embedding \"books\"                 # Process books folder"
    echo "  $0 indexer-image-verbal \"books/document.pdf\"        # Process specific file"
    echo ""
    echo "Alternative: Use only data path (uses default strategy):"
    echo "  $0 \"books/document.pdf\"                             # Process specific file with default strategy"
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

root=$(pwd)
backendPath="$root/src/backend"
azurePath="$root/.azure"

# find the .env file in the azurePath directory recursively
envFile=$(find "$azurePath" -type f -name ".env" | head -n 1)

if [ -f "$envFile" ]; then
    echo ".env file found at: $envFile"
else
    echo ".env file not found. Please run azd up and ensure it completes successfully."
    exit 1
fi

# Load azd environment variables
echo "Loading azd environment variables"
azdEnv=$(azd env get-values --output json)

# Check if the azd command succeeded
if [ $? -ne 0 ]; then
    echo "Failed to load azd environment variables. Ensure azd is installed and configured correctly."
    exit 1
fi

# Parse and export each environment variable in the current shell session
eval $(echo "$azdEnv" | jq -r 'to_entries | .[] | "export \(.key)=\(.value)"')

echo 'Creating Python virtual environment'
python3 -m venv .venv

echo 'Installing dependencies from "requirements.txt" into virtual environment (in quiet mode)...'
.venv/bin/python -m pip --quiet --disable-pip-version-check install -r src/backend/requirements.txt

echo 'Run the document preparation script'

# Smart argument parsing
args_indexer_strategy='self-multimodal-embedding'
args_data_path=''

if [ $# -eq 0 ]; then
    # No arguments - use defaults
    echo "No arguments provided, using defaults"
elif [ $# -eq 1 ]; then
    # One argument - could be strategy or data path
    case "$1" in
        "indexer-image-verbal"|"self-multimodal-embedding")
            # It's a valid strategy
            args_indexer_strategy="$1"
            echo "Using indexer strategy: $args_indexer_strategy"
            ;;
        *)
            # Assume it's a data path
            args_data_path="$1"
            echo "Using default strategy with data path: $args_data_path"
            ;;
    esac
elif [ $# -eq 2 ]; then
    # Two arguments - strategy and data path
    args_indexer_strategy="$1"
    args_data_path="$2"
    echo "Using indexer strategy: $args_indexer_strategy with data path: $args_data_path"
else
    echo "Error: Too many arguments provided"
    show_usage
    exit 1
fi

# Build the command with optional data_path parameter
if [ -n "$args_data_path" ]; then
    echo "Processing documents from: $args_data_path"
    .venv/bin/python "src/backend/prepdocs.py" --source files --indexer_strategy "$args_indexer_strategy" --data_path "$args_data_path"
else
    echo "Processing all documents in data directory"
    .venv/bin/python "src/backend/prepdocs.py" --source files --indexer_strategy "$args_indexer_strategy"
fi 