#!/bin/bash
# Script to set PromptFlow environment variables

# Set environment variables for the current session
export PROMPTFLOW_REQUEST_FIELD_NAME=chat_input
export PROMPTFLOW_RESPONSE_FIELD_NAME=output
export PROMPTFLOW_CITATIONS_FIELD_NAME=citations

echo "Set PromptFlow field name environment variables:"
echo "PROMPTFLOW_REQUEST_FIELD_NAME=$PROMPTFLOW_REQUEST_FIELD_NAME"
echo "PROMPTFLOW_RESPONSE_FIELD_NAME=$PROMPTFLOW_RESPONSE_FIELD_NAME"
echo "PROMPTFLOW_CITATIONS_FIELD_NAME=$PROMPTFLOW_CITATIONS_FIELD_NAME"

# Check if an .env file exists in any of the common locations
ENV_FILES=(
  ".env"
  ".azure/my-multimodal-env/.env"
  "src/backend/.env"
)

echo ""
echo "To make these settings permanent, add them to your .env file or environment configuration."
echo "You can use one of these commands to add them to an existing .env file:"

for env_file in "${ENV_FILES[@]}"; do
  if [ -f "$env_file" ]; then
    echo ""
    echo "For $env_file:"
    echo "echo \"PROMPTFLOW_REQUEST_FIELD_NAME=chat_input\" >> $env_file"
    echo "echo \"PROMPTFLOW_RESPONSE_FIELD_NAME=output\" >> $env_file"
    echo "echo \"PROMPTFLOW_CITATIONS_FIELD_NAME=citations\" >> $env_file"
  fi
done

# Suggest .env file locations
echo "Common .env file locations in this project:"
for env_file in "${ENV_FILES[@]}"; do
  if [ -f "$env_file" ]; then
    echo "- $env_file (exists)"
  else
    echo "- $env_file (does not exist)"
  fi
done 