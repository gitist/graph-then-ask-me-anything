import openvino as ov
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool

DEVICE = "CPU"
OV_CONFIG = { "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
MAX_NEW_TOKENS = 100
CONTEXT_WINDOW = 3900
#LLM_MODEL_NAME = "OpenVINO/Phi-3-mini-128k-instruct-int4-ov"
LLM_MODEL_NAME = "nsbendre25/llama-3-8B-Instruct-ov-fp16-int4-sym"
EMBEDDING_MODEL_NAME = "ojjsaw/embedding_model"
PDF_TEST_FILE_PATH = "xeon6-e-cores-network-and-edge-brief.pdf"

def phi_messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}<|end|>\n"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}<|end|>\n"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}<|end|>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n<|end|>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt

def phi_completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"

# Load the embeddings, LLM models and apply to the settings
core = ov.Core()
embedding = OpenVINOEmbedding(model_id_or_path=EMBEDDING_MODEL_NAME, device=DEVICE)
llm = OpenVINOLLM(
    model_id_or_path=LLM_MODEL_NAME,
    context_window=CONTEXT_WINDOW,
    max_new_tokens=MAX_NEW_TOKENS,
    model_kwargs={"ov_config": OV_CONFIG},
    generate_kwargs={"do_sample": True, "temperature": 0.1, "top_p": 1.0, "top_k": 50},
    device_map=DEVICE,
#   messages_to_prompt=phi_messages_to_prompt,
#   completion_to_prompt=phi_completion_to_prompt,
    stopping_ids=[128001],
)
Settings.embed_model = embedding
Settings.llm = llm

# Create the tools
def multiply(a: float, b: float) -> float:
    return a * b
def add(a: float, b: float) -> float:
    return a + b
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

# Create a rag query engine for the rag tool
loader = PyMuPDFReader()
documents = loader.load(file_path=PDF_TEST_FILE_PATH)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=2)
rag_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="vector_search",
    description="A RAG engine with some basic facts about Intel Xeon 6 processors with E-cores",
)

agent = ReActAgent.from_tools([multiply_tool, add_tool, rag_tool], llm=llm, verbose=True)
#response = agent.chat("What's the maximum number of cores in 3600 sockets of Intel Xeon 6 processor ? Go step by step, using a tool to do any math.")
response = agent.chat("What's the maximum number of cores in Intel Xeon 6 processor ?")
agent.reset()