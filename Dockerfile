# Use the official Ubuntu 22.04 image as the base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements-agents.txt .

# Create a virtual environment and install dependencies
RUN python3 -m venv venv \
    && . venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements-agents.txt

# Copy the rest of the application code
COPY llamaindex-minimal.py .
COPY llm_config.py .
COPY xeon6-e-cores-network-and-edge-brief.pdf .
COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh

# Activate the virtual environment and run the hugging-cli command
RUN . venv/bin/activate \
    && huggingface-cli download ojjsaw/reranking_model \
    && huggingface-cli download ojjsaw/embedding_model \
    && huggingface-cli download OpenVINO/Phi-3-mini-128k-instruct-int4-ov

# Set the entrypoint script
ENTRYPOINT ["/bin/bash", "-c", "source venv/bin/activate && uvicorn llamaindex-minimal:app --host 0.0.0.0"]