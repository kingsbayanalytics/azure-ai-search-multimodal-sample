#!/bin/bash
# Script to restart the application with the correct PromptFlow field names

# Set the PromptFlow field names
source scripts/set_promptflow_env.sh

# Determine how to restart the application
# This depends on how the application is currently being run

echo "Looking for running application processes..."

# Check for common process names that might be running the application
app_processes=$(ps aux | grep -E 'python.*app.py|npm.*start|node.*server' | grep -v grep)

if [ -n "$app_processes" ]; then
  echo "Found running application processes:"
  echo "$app_processes"
  echo ""
  echo "Please stop these processes and restart the application with:"
  echo ""
  
  # Suggest restart commands based on common patterns
  if [[ "$app_processes" == *"python"*"app.py"* ]]; then
    echo "For backend: python src/backend/app.py"
  elif [[ "$app_processes" == *"npm"*"start"* ]]; then
    echo "For frontend: npm start"
  elif [[ "$app_processes" == *"node"* ]]; then
    echo "For node server: node server.js"
  else
    echo "Please restart your application using the same command you used to start it."
  fi
else
  echo "No running application processes found."
  echo ""
  echo "To start the application:"
  echo "1. Backend: python src/backend/app.py"
  echo "2. Frontend: (in another terminal) cd src/frontend && npm start"
fi

echo ""
echo "The PromptFlow field names have been set to:"
echo "PROMPTFLOW_REQUEST_FIELD_NAME=$PROMPTFLOW_REQUEST_FIELD_NAME"
echo "PROMPTFLOW_RESPONSE_FIELD_NAME=$PROMPTFLOW_RESPONSE_FIELD_NAME"
echo "PROMPTFLOW_CITATIONS_FIELD_NAME=$PROMPTFLOW_CITATIONS_FIELD_NAME"
echo ""
echo "These settings will be used when you restart the application." 