import openvino as ov
import torch
from transformers import (
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from threading import Thread

DEFAULT_RAG_PROMPT = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""
PDF_TEST_FILE_PATH = "text_example_en.pdf"
LLM_MODEL_NAME = "OpenVINO/Phi-3-mini-128k-instruct-int4-ov"
STOP_TOKENS = ["<|end|>"]
RAG_PROMPT_TEMPLATE = """
<|system|> {DEFAULT_RAG_PROMPT }<|end|>
<|user|>
Question: {input} 
Context: {context} 
Answer: <|end|>
<|assistant|>
"""
EMBEDDING_MODEL_NAME = "embedding_model"
EMBEDDING_BATCH_SIZE = 4
RERANKING_MODEL_NAME = "reranking_model"
RERANK_TOP_N = 2
DEVICE = "CPU"
OV_CONFIG = { "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "" }

core = ov.Core()

embedding = OpenVINOBgeEmbeddings(
    model_name_or_path=EMBEDDING_MODEL_NAME,
    model_kwargs={ "device": DEVICE, "compile": False },
    encode_kwargs={ "mean_pooling": False, "normalize_embeddings": True, "batch_size": EMBEDDING_BATCH_SIZE},
)

embedding.ov_model.compile()

reranker = OpenVINOReranker(
    model_name_or_path=RERANKING_MODEL_NAME,
    model_kwargs={ "device": DEVICE },
    top_n=RERANK_TOP_N,
)

llm = HuggingFacePipeline.from_model_id(
    model_id=LLM_MODEL_NAME,
    task="text-generation",
    backend="openvino",
    model_kwargs={ "device": DEVICE, "ov_config": OV_CONFIG, "trust_remote_code": True },
    pipeline_kwargs={ "max_new_tokens": 2 }
)

### STOP TOKENS
class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

if STOP_TOKENS is not None:
    if isinstance(STOP_TOKENS[0], str):
        STOP_TOKENS = llm.pipeline.tokenizer.convert_tokens_to_ids(STOP_TOKENS)
    STOP_TOKENS = [StopOnTokens(STOP_TOKENS)]

### LOAD DOCs
def load_single_document(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()

def default_partial_text_processor(partial_text: str, new_text: str):
    partial_text += new_text
    return partial_text

text_processor = default_partial_text_processor

### Vector Store
def create_vectordb(docs, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, score_threshold):
    global db
    global retriever
    global combine_docs_chain
    global rag_chain

    if vector_rerank_top_n > vector_search_top_k:
        print("Search top k must >= Rerank top n")
        return False

    documents = []
    for doc in docs:
        if type(doc) is not str:
            doc = doc.name
        documents.extend(load_single_document(doc))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embedding)
    search_kwargs = {"k": vector_search_top_k, "score_threshold": score_threshold}
    retriever = db.as_retriever(search_kwargs=search_kwargs, search_type="similarity_score_threshold")
    reranker.top_n = vector_rerank_top_n
    retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return "Vector database is Ready"

def user(message, history):
    return "", history + [[message, ""]]

def bot(history, temperature, top_p, top_k, repetition_penalty, hide_full_prompt):
    streamer = TextIteratorStreamer(
        llm.pipeline.tokenizer,
        timeout=60.0,
        skip_prompt=hide_full_prompt,
        skip_special_tokens=True,
    )
    llm.pipeline._forward_params = dict(
        max_new_tokens=512,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )

    if STOP_TOKENS is not None:
        llm.pipeline._forward_params["stopping_criteria"] = StoppingCriteriaList(STOP_TOKENS)
    
    # t1 = Thread(target=rag_chain.invoke, args=({"input": history[-1][0]},))
    # t1.start()

    print(rag_chain.invoke(history))

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text = text_processor(partial_text, new_text)
        history[-1][1] = partial_text
        yield history
    
    print(history)

def request_cancel():
    llm.pipeline.model.request.cancel()


def clear_files():
    return "Vector Store is Not ready"

print(
    create_vectordb(
        [PDF_TEST_FILE_PATH],
        chunk_size=400,
        chunk_overlap=50,
        vector_search_top_k=10,
        vector_rerank_top_n=2,
        score_threshold=0.5,
    )
)

history = "what doe Intel vPro® Enterprise systems offer?"
repetition_penalty = 1.1 # 1.0 to 2.0, step 0.1
top_k = 50 # 0.0 to 200, step 1
top_p = 1.0 # 0.0 to 1.0, step 0.1
temperature = 0.1 # 0.0 to 1.0, step 0.1
hide_context = True
bot(history, temperature, top_p, top_k, repetition_penalty, hide_context)

del reranker
del llm
del core