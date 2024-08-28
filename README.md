# ask-me-anything

### Minimal System Requirements
1. Storage >= 16GB
2. RAM >= 8GB
3. CPU cores >= 4
4. Core Frequency >= 3GHz

### Local Envr. Setup

```sh
python3 -m venv agents_env
source agents_env/bin/activate
python -m pip install --upgrade pip

pip install -r requirements-agents.txt

huggingface-cli download ojjsaw/reranking_model
huggingface-cli download ojjsaw/embedding_model
huggingface-cli download OpenVINO/Phi-3-mini-128k-instruct-int4-ov
```

### Local Run Steps
1. Ideal Setup: Run LLM model on GPU and other 2 models on CPU (default all models run on CPU)
2. To switch to gpu, update LLM_DEVICE to "GPU" in llamaindex-minimal.py
```sh
uvicorn llamaindex-minimal:app
```
3. To run via Swagger (non streaming responses), navigate to http://127.0.0.1:8000/docs
4. To use upload api via terminal CURL:
```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/upload-pdf-vector-index' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@xeon6-e-cores-network-and-edge-brief.pdf;type=application/pdf'
```
5. To run ask-me-anything api via terminal **preferred** for streaming responses.

Note: Use -N param to prevent buffer cache and see live chunked response data
```sh
curl -N -X 'POST' \
  'http://127.0.0.1:8000/ask-me-anything' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What is the range of Thermal Design Power (TDP) for Intel Xeon 6 processors with E-cores?"
}'
```

### Intel GPU Setup on Linux and WSL2 Linux for Ubuntu 22.04
1. Repo Signing
```sh
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key |
     sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
```

2. APT Repo Setup
```sh
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
sudo apt update
```

3. Install minimal GPU deps for AI workload
```sh
apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero
```

Troubleshooting/alternate OS Steps:
- https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html
- https://dgpu-docs.intel.com/driver/client/overview.html 

