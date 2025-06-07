# PromptFlow Configuration Guide

This document explains how to configure the application to work with PromptFlow by setting the correct environment variables.

## Required Environment Variables

The following environment variables are required for basic PromptFlow functionality:

```
PROMPT_FLOW_ENDPOINT=your_endpoint_url
PROMPT_FLOW_FLOW_NAME=your_flow_name
PROMPT_FLOW_API_KEY=your_api_key
USE_PROMPT_FLOW=true
```

## Field Name Customization

PromptFlow endpoints can use different field names for inputs and outputs. You can customize these field names to match your PromptFlow endpoint's expectations using these environment variables:

```
PROMPTFLOW_REQUEST_FIELD_NAME=chat_input    # Default: "question"
PROMPTFLOW_RESPONSE_FIELD_NAME=output       # Default: "output"
PROMPTFLOW_CITATIONS_FIELD_NAME=citations   # Default: "citations"
```

### Verified Field Names

Through testing, we have confirmed that the following field names work with the current PromptFlow endpoint:

- Input field: `chat_input` - This is the field name used to send the user's question to the PromptFlow endpoint
- Response field: `output` - This is the field name where the PromptFlow response is found
- Citations field: `citations` - This is the field name where citation information is stored

The response structure from the PromptFlow endpoint looks like:

```json
{
  "chat_output": {
    "citations": [
      {
        "chunk_id": null,
        "content": "...",
        "filepath": "...",
        "id": "doc1",
        "metadata": {...}
      }
    ],
    "id": "...",
    "output": "Response text here"
  }
}
```

## Setting Environment Variables

You can set these environment variables in your .env file. The application looks for .env files in the following locations:

1. Root directory (`.env`)
2. Azure environment directory (`.azure/my-multimodal-env/.env`)
3. Backend directory (`src/backend/.env`)

### Using the Helper Script

A helper script is provided to set these environment variables temporarily for the current session:

```bash
source scripts/set_promptflow_env.sh
```

The script will also provide commands to add these settings permanently to your .env files.

## Testing Your Configuration

Use the provided test script to verify your configuration:

```bash
python scripts/test_promptflow_fields.py
```

This will check if your environment variables are set correctly and test the connection to your PromptFlow endpoint with the configured field names.

## Troubleshooting

If you're having issues with PromptFlow responses:

1. Check the logs for the response structure coming from the PromptFlow endpoint
2. Run the test script to check field name configurations:
   ```bash
   python scripts/test_promptflow_fields.py
   ```
3. Verify that the environment variables are set correctly
4. Restart the application after changing environment variables

## Looking at Logs

The application logs information about field name configuration:

```
PromptFlow field names - Request: chat_input, Response: output, Citations: citations
Looking for response in field 'output' and citations in field 'citations'
```

Check the logs to verify that the correct field names are being used. 