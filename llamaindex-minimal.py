import openvino as ov
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.postprocessor.openvino_rerank import OpenVINORerank
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core import PromptTemplate

import time
import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
import logging
import time
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)

# EMBEDDING CONFIG
EMBEDDING_DEVICE = "CPU"
EMBEDDING_CHUNK_SIZE = 900 #500
EMBEDDING_CHUNK_OVERLAP = 200 #100
EMBEDDING_MODEL_NAME = "ojjsaw/embedding_model"
EMBEDDING_BATCH_SIZE = 4
EMBEDDING_MAX_NEW_TOKENS = 512

# RERANKING CONFIG
RERANKING_DEVICE = "CPU"
RERANKING_MODEL_NAME = "ojjsaw/reranking_model"
RERANKING_TOP_N = 2

# LLM CONFIG
LLM_DEVICE = "CPU"# "GPU"
LLM_MODEL_NAME = "OpenVINO/Phi-3-mini-128k-instruct-int4-ov"
#LLM_MODEL_NAME = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"
LLM_MAX_NEW_TOKENS = 256 # 512
LLM_CONTEXT_WINDOW = 131072 #4096
CUSTOM_SYSTEM_PROMPT = "<|system|>\nYou are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.<|end|>\n"
OV_CONFIG = { "PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": ""}
QA_PROMPT_TMPL_STR = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query. Incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
            )
QA_PROMPT_TEMPLATE = PromptTemplate(QA_PROMPT_TMPL_STR)

def messages_to_prompt(messages):
    prompt = ""
    system_found = False
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}<|end|>\n"
            system_found = True
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}<|end|>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}<|end|>\n"
        else:
            prompt += f"<|user|>\n{message.content}<|end|>\n"

    # trailing prompt
    prompt += "<|assistant|>\n"

    if not system_found:
        prompt = (
            CUSTOM_SYSTEM_PROMPT + prompt
        )

    #print(f"Prompt: {prompt}")
    return prompt

def phi_completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"

logging.info("Loading LLM and embedding models")
core = ov.Core()
embedding = OpenVINOEmbedding(
    model_id_or_path=EMBEDDING_MODEL_NAME, 
    device=EMBEDDING_DEVICE,
    model_kwargs={"ov_config": OV_CONFIG, "trust_remote_code": True},
    max_length=EMBEDDING_MAX_NEW_TOKENS,
    embed_batch_size=EMBEDDING_BATCH_SIZE,
)

reranker = OpenVINORerank(
    model_id_or_path=RERANKING_MODEL_NAME, 
    device=RERANKING_DEVICE,
    model_kwargs={"ov_config": OV_CONFIG, "trust_remote_code": True},
    top_n=RERANKING_TOP_N
)

llm = OpenVINOLLM(
    model_id_or_path=LLM_MODEL_NAME,
    context_window=LLM_CONTEXT_WINDOW,
    max_new_tokens=LLM_MAX_NEW_TOKENS,
    model_kwargs={"ov_config": OV_CONFIG, "trust_remote_code": True },
    generate_kwargs={
        "do_sample": True, 
        "temperature": 0.1,
        #"top_k": 0, 
        #"top_p": 1.0 
        },
    device_map=LLM_DEVICE,
    query_wrapper_prompt=(
            "<|system|>\n" + CUSTOM_SYSTEM_PROMPT +
            "<|user|>\n" +
            "{query_str}<|end|>\n"
            "<|assistant|>\n"
        ),
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=phi_completion_to_prompt,
    is_chat_model=True,
)

logging.info("Updating Settings")
Settings.embed_model = embedding
Settings.llm = llm
Settings.chunk_size = EMBEDDING_CHUNK_SIZE
Settings.chunk_overlap = EMBEDDING_CHUNK_OVERLAP

app = FastAPI(
    title="Ask Me Anything API",
    description="""
    E2E llamaindex streaming RAG lightweight models and tradeoff accuracy vs latency:
    1. EMBEDDINGS MODEL: INT4 Quantized BAAI/bge-base-en-v1.5
    2. LLM MODEL: phi3-mini-128k-instruct
    3. RERANKING MODEL: BAAI/bge-reranker-base
    Note: Swagger UI waits to complete response, use example {CURL} command in description of API for non-buffered live streaming result.
    """,
)

@app.post("/upload-pdf-vector-index", tags=["Knowledge Base"])
async def upload_file(file: UploadFile = File(...)):
    logging.info("Initializing VectorStoreIndex and QueryEngine")
    if file.filename.endswith('.pdf'):
        start_time = time.time()
        loader = PyMuPDFReader()
        documents = loader.load(file_path=file.filename)
        vector_index = VectorStoreIndex.from_documents(documents)

        global query_engine
        query_engine = vector_index.as_query_engine(
            streaming=True, 
            similarity_top_k=RERANKING_TOP_N, 
            response_mode="compact", 
            node_postprocessors=[reranker]
        )
        query_engine.update_prompts({"response_synthesizer:text_qa_template": QA_PROMPT_TEMPLATE})
        end_time = time.time()
        status = f"Succesfully updated vector index with data from {file.filename} in {end_time - start_time:.2f} seconds\n"
        return {"status": status}
    else:
        return {"status": "error", "message": "Only PDF files are allowed."}

class QueryRequest(BaseModel):
    question: str = Query(..., description="The question to be queried")

def response_streamer(question: str):
    logging.info("Querying the engine")
    start_time = time.time()
    global query_engine
    streaming_response = query_engine.query(question)
    for line in streaming_response.response_gen:
        yield line
    end_time = time.time()
    yield f"\nDuration: {end_time - start_time:.2f} seconds\n"

@app.post("/ask-me-anything", tags=["QnA"])
async def query_post(request: QueryRequest):
    return StreamingResponse(response_streamer(request.question), media_type="text/event-stream")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

