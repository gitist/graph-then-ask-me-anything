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

logging.basicConfig(level=logging.INFO)

PDF_TEST_FILE_PATH = "xeon6-e-cores-network-and-edge-brief.pdf"

EMBEDDING_DEVICE = "CPU"
RERANKING_DEVICE = "CPU"
LLM_DEVICE = "GPU"
OV_CONFIG = { "PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": ""}
MAX_NEW_TOKENS = 512
CONTEXT_WINDOW = 127000
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
LLM_MODEL_NAME = "OpenVINO/Phi-3-mini-128k-instruct-int4-ov"
EMBEDDING_MODEL_NAME = "ojjsaw/embedding_model"
RERANKING_MODEL_NAME = "ojjsaw/reranking_model"
EMBEDDING_BATCH_SIZE = 4

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
            "<|system|>\nYou are a helpful AI assistant.<|end|>\n" + prompt
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
    max_length=MAX_NEW_TOKENS,
    embed_batch_size=EMBEDDING_BATCH_SIZE,
)

reranker = OpenVINORerank(
    model_id_or_path=RERANKING_MODEL_NAME, 
    device=RERANKING_DEVICE,
    model_kwargs={"ov_config": OV_CONFIG, "trust_remote_code": True},
    top_n=2
)

llm = OpenVINOLLM(
    model_id_or_path=LLM_MODEL_NAME,
    context_window=CONTEXT_WINDOW,
    max_new_tokens=MAX_NEW_TOKENS,
    model_kwargs={"ov_config": OV_CONFIG, "trust_remote_code": True },
    generate_kwargs={
        "do_sample": True, 
        "temperature": 0.1,
        #"top_k": 0, 
        #"top_p": 1.0 
        },
    device_map=LLM_DEVICE,
    query_wrapper_prompt=(
            "<|system|>\n"
            "You are a helpful AI assistant.<|end|>\n"
            "<|user|>\n"
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
Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = CHUNK_OVERLAP

logging.info("Initializing VectorStoreIndex and QueryEngine")
loader = PyMuPDFReader()
documents = loader.load(file_path=PDF_TEST_FILE_PATH)
vector_index = VectorStoreIndex.from_documents(documents)

query_engine = vector_index.as_query_engine(
    streaming=True, 
    similarity_top_k=2, 
    response_mode="compact", 
    node_postprocessors=[reranker]
)

qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
            )
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
# question = "what is the base power range of xeon 6 processors?"
# logging.info("Querying the engine")
# start_time = time.time()
# response = query_engine.query(question)
# response.print_response_stream()
# print("\n")
# end_time = time.time()
# logging.info(f"Execution time: {end_time - start_time} seconds")

try:
    while True:
        question = input("Enter your question (or 'q' to quit): ")
        if question == "q":
            break
        
        logging.info("Querying the engine")
        start_time = time.time()
        response = query_engine.query(question)
        response.print_response_stream()
        print("\n")
        end_time = time.time()
        logging.info(f"Execution time: {end_time - start_time} seconds")
except KeyboardInterrupt:
    logging.info("Program stopped by user")
    exit()

