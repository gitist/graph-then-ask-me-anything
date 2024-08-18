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
docker run -e HF_TOKEN=? -p 7860:7860 ojjsaw/ask-me-anything
```

### better configs
```sh
chunk size 150
chunk overlap 50
similarity threshold 0.5
rerank top n 7
rerank top k 15
```