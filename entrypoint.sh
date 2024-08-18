#!/bin/bash

# Activate the virtual environment
source /app/venv/bin/activate

# Login to Hugging Face using the provided token
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  exit 1
fi

huggingface-cli login --token $HF_TOKEN
huggingface-cli download ojjsaw/reranking_model --token $HF_TOKEN
huggingface-cli download ojjsaw/embedding_model --token $HF_TOKEN
huggingface-cli download OpenVINO/Phi-3-mini-128k-instruct-int4-ov --token $HF_TOKEN

# Start the Gradio app
python3 gradio_app.py