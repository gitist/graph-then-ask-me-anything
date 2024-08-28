# ask-me-anything

```sh
python3 -m venv openvino_prod
source openvino_prod/bin/activate
python -m pip install --upgrade pip

# TODO: prevent nvidia installs by manual override of torch cpu installation
pip install -r requirements.txt

pip install -r requirements-prod.txt #shrink by 3.5GB

```
### Embedding Model
```sh
# TODO: Use INT8 quantized with large variant like https://huggingface.co/Intel/bge-large-en-v1.5-rag-int8-static
# Note: Switched to base model instead of small from nb example.
optimum-cli export openvino --model "BAAI/bge-base-en-v1.5" --task feature-extraction embedding_model

optimum-cli export openvino --model "BAAI/bge-base-en-v1.5" --task feature-extraction --weight-format int8 embedding_model

optimum-cli export openvino --model "BAAI/bge-base-en-v1.5" --task feature-extraction --weight-format int4 embedding_model
```

### Reranking Model
```sh
optimum-cli export openvino --model "BAAI/bge-reranker-base" --task text-classification reranking_model

optimum-cli export openvino --model "BAAI/bge-reranker-base" --task text-classification --weight-format int8 reranking_model

optimum-cli export openvino --model "BAAI/bge-reranker-base" --task text-classification --weight-format int4 reranking_model
```

### LLM Model
```sh
huggingface-cli download "OpenVINO/Phi-3-mini-128k-instruct-int4-ov"

#TODO:
huggingface-cli download "microsoft/Phi-3.5-mini-instruct"
optimum-cli export openvino --model "microsoft/Phi-3.5-mini-instruct" --task text-generation-with-past --trust-remote-code --weight-format int4 --group-size 128 --ratio 0.8 llm_model


```
### Utility Commands
```sh
du -h {PATH_TO_DIR}
huggingface-cli scan-cache
huggingface-cli download ojjsaw/reranking_model --token ?
huggingface-cli download ojjsaw/embedding_model --token ?
huggingface-cli login --token ?

# docker gets stuck on running
docker build -t ojjsaw/ask-me-anything .
docker run --rm -p 8000:8000 ojjsaw/ask-me-anything

curl -X 'POST' \
  'http://127.0.0.1:8000/upload-pdf-vector-index' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@xeon6-e-cores-network-and-edge-brief.pdf;type=application/pdf'

```

### better configs
```sh
chunk size 150
chunk overlap 50
similarity threshold 0.5
rerank top n 7
rerank top k 15
```
# RAG AGENTs

### Setup
```sh
python3 -m venv agents_env
source agents_env/bin/activate
python -m pip install --upgrade pip

pip install -r requirements-agents.txt

uvicorn fastapi-test:app --reload

uvicorn llamaindex-minimal:app --reload

curl -N -X 'POST' \
  'http://127.0.0.1:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What is the memory speed supported by Intel Xeon 6 processors with DDR5?"
}'

```