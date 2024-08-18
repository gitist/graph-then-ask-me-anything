
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from threading import Thread
import gradio as gr
import re
from typing import List
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from pathlib import Path
import openvino as ov
import torch
from transformers import (
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from llm_config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_RERANK_MODELS,
    SUPPORTED_LLM_MODELS,
)

PDF_TEST_FILE_PATH = "text_example_en.pdf"
text_example_path = PDF_TEST_FILE_PATH
LLM_MODEL_NAME = "OpenVINO/Phi-3-mini-128k-instruct-int4-ov"
EMBEDDING_MODEL_NAME = "ojjsaw/embedding_model"
EMBEDDING_BATCH_SIZE = 4
RERANKING_MODEL_NAME = "ojjsaw/reranking_model"
RERANK_TOP_N = 2
DEVICE = "CPU"
OV_CONFIG = { "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "" }

llm_model_configuration = SUPPORTED_LLM_MODELS['English']['phi-3-mini-instruct']
rag_prompt_template = llm_model_configuration["rag_prompt_template"]
stop_tokens = llm_model_configuration.get("stop_tokens")

print(llm_model_configuration)

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


TEXT_SPLITERS = {
    "Character": CharacterTextSplitter,
    "RecursiveCharacter": RecursiveCharacterTextSplitter,
    "Markdown": MarkdownTextSplitter,
    "Chinese": ChineseTextSplitter,
}


LOADERS = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}
english_examples = [
    ["How much power consumption can Intel® Core™ Ultra Processors help save?"],
    ["Compared to Intel’s previous mobile processor, what is the advantage of Intel® Core™ Ultra Processors for Artificial Intelligence?"],
    ["What can Intel vPro® Enterprise systems offer?"],
]
examples = english_examples

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

if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = llm.pipeline.tokenizer.convert_tokens_to_ids(stop_tokens)
    stop_tokens = [StopOnTokens(stop_tokens)]

### LOAD DOCs
def load_single_document(file_path: str) -> List[Document]:
    """
    helper for loading a single document

    Params:
      file_path: document path
    Returns:
      documents loaded

    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADERS:
        loader_class, loader_args = LOADERS[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"File does not exist '{ext}'")

def default_partial_text_processor(partial_text: str, new_text: str):
    partial_text += new_text
    return partial_text

text_processor = llm_model_configuration.get("partial_text_processor", default_partial_text_processor)

### Vector Store
def create_vectordb(
    docs, spliter_name, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, run_rerank, search_method, score_threshold, progress=gr.Progress()
):
    """
    Initialize a vector database

    Params:
      doc: orignal documents provided by user
      spliter_name: spliter method
      chunk_size:  size of a single sentence chunk
      chunk_overlap: overlap size between 2 chunks
      vector_search_top_k: Vector search top k
      vector_rerank_top_n: Search rerank top n
      run_rerank: whether run reranker
      search_method: top k search method
      score_threshold: score threshold when selecting 'similarity_score_threshold' method

    """
    global db
    global retriever
    global combine_docs_chain
    global rag_chain

    if vector_rerank_top_n > vector_search_top_k:
        gr.Warning("Search top k must >= Rerank top n")

    documents = []
    for doc in docs:
        if type(doc) is not str:
            doc = doc.name
        documents.extend(load_single_document(doc))

    text_splitter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embedding)
    if search_method == "similarity_score_threshold":
        search_kwargs = {"k": vector_search_top_k, "score_threshold": score_threshold}
    else:
        search_kwargs = {"k": vector_search_top_k}
    retriever = db.as_retriever(search_kwargs=search_kwargs, search_type=search_method)
    if run_rerank:
        reranker.top_n = vector_rerank_top_n
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
    prompt = PromptTemplate.from_template(rag_prompt_template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return "Vector database is Ready"


def update_retriever(vector_search_top_k, vector_rerank_top_n, run_rerank, search_method, score_threshold):
    """
    Update retriever

    Params:
      vector_search_top_k: Vector search top k
      vector_rerank_top_n: Search rerank top n
      run_rerank: whether run reranker
      search_method: top k search method
      score_threshold: score threshold when selecting 'similarity_score_threshold' method

    """
    global db
    global retriever
    global combine_docs_chain
    global rag_chain

    if vector_rerank_top_n > vector_search_top_k:
        gr.Warning("Search top k must >= Rerank top n")

    if search_method == "similarity_score_threshold":
        search_kwargs = {"k": vector_search_top_k, "score_threshold": score_threshold}
    else:
        search_kwargs = {"k": vector_search_top_k}
    retriever = db.as_retriever(search_kwargs=search_kwargs, search_type=search_method)
    if run_rerank:
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
        reranker.top_n = vector_rerank_top_n
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return "Vector database is Ready"

def user(message, history):
    return "", history + [[message, ""]]

def bot(history, temperature, top_p, top_k, repetition_penalty, hide_full_prompt, do_rag):
    """
    callback function for running chatbot on submit button click

    Params:
      history: conversation history
      temperature:  parameter for control the level of creativity in AI-generated text.
                    By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
      top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
      top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
      repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
      hide_full_prompt: whether to show searching results in promopt.
      do_rag: whether do RAG when generating texts.

    """
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
    if stop_tokens is not None:
        llm.pipeline._forward_params["stopping_criteria"] = StoppingCriteriaList(stop_tokens)

    if do_rag:
        t1 = Thread(target=rag_chain.invoke, args=({"input": history[-1][0]},))
    else:
        input_text = rag_prompt_template.format(input=history[-1][0], context="")
        t1 = Thread(target=llm.invoke, args=(input_text,))
    t1.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text = text_processor(partial_text, new_text)
        history[-1][1] = partial_text
        yield history

def request_cancel():
    llm.pipeline.model.request.cancel()


def clear_files():
    return "Vector Store is Not ready"

# initialize the vector store with example document
create_vectordb(
    [text_example_path],
    "RecursiveCharacter",
    chunk_size=400,
    chunk_overlap=50,
    vector_search_top_k=10,
    vector_rerank_top_n=2,
    run_rerank=True,
    search_method="similarity_score_threshold",
    score_threshold=0.5,
)


### StartAPP

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    gr.Markdown("""<h1><center>QA over Document</center></h1>""")
    gr.Markdown(f"""<center>Powered by OpenVINO and {LLM_MODEL_NAME} </center>""")
    with gr.Row():
        with gr.Column(scale=1):
            docs = gr.File(
                label="Step 1: Load text files",
                value=[text_example_path],
                file_count="multiple",
                file_types=[
                    ".csv",
                    ".doc",
                    ".docx",
                    ".enex",
                    ".epub",
                    ".html",
                    ".md",
                    ".odt",
                    ".pdf",
                    ".ppt",
                    ".pptx",
                    ".txt",
                ],
            )
            load_docs = gr.Button("Step 2: Build Vector Store", variant="primary")
            db_argument = gr.Accordion("Vector Store Configuration", open=False)
            with db_argument:
                spliter = gr.Dropdown(
                    ["Character", "RecursiveCharacter", "Markdown", "Chinese"],
                    value="RecursiveCharacter",
                    label="Text Spliter",
                    info="Method used to splite the documents",
                    multiselect=False,
                )

                chunk_size = gr.Slider(
                    label="Chunk size",
                    value=150,
                    minimum=50,
                    maximum=2000,
                    step=50,
                    interactive=True,
                    info="Size of sentence chunk",
                )

                chunk_overlap = gr.Slider(
                    label="Chunk overlap",
                    value=50,
                    minimum=0,
                    maximum=400,
                    step=10,
                    interactive=True,
                    info=("Overlap between 2 chunks"),
                )

            langchain_status = gr.Textbox(
                label="Vector Store Status",
                value="Vector Store is Ready",
                interactive=False,
            )
            do_rag = gr.Checkbox(
                value=True,
                label="RAG is ON",
                interactive=True,
                info="Whether to do RAG for generation",
            )
            with gr.Accordion("Generation Configuration", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=50,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=300,
                label="Step 3: Input Query",
            )
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        msg = gr.Textbox(
                            label="QA Message Box",
                            placeholder="Chat Message Box",
                            show_label=False,
                            container=False,
                        )
                with gr.Column():
                    with gr.Row():
                        submit = gr.Button("Submit", variant="primary")
                        stop = gr.Button("Stop")
                        clear = gr.Button("Clear")
            gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")
            retriever_argument = gr.Accordion("Retriever Configuration", open=True)
            with retriever_argument:
                with gr.Row():
                    with gr.Row():
                        do_rerank = gr.Checkbox(
                            value=True,
                            label="Rerank searching result",
                            interactive=True,
                        )
                        hide_context = gr.Checkbox(
                            value=True,
                            label="Hide searching result in prompt",
                            interactive=True,
                        )
                    with gr.Row():
                        search_method = gr.Dropdown(
                            ["similarity_score_threshold", "similarity", "mmr"],
                            value="similarity_score_threshold",
                            label="Searching Method",
                            info="Method used to search vector store",
                            multiselect=False,
                            interactive=True,
                        )
                    with gr.Row():
                        score_threshold = gr.Slider(
                            0.01,
                            0.99,
                            value=0.5,
                            step=0.01,
                            label="Similarity Threshold",
                            info="Only working for 'similarity score threshold' method",
                            interactive=True,
                        )
                    with gr.Row():
                        vector_rerank_top_n = gr.Slider(
                            1,
                            10,
                            value=7,
                            step=1,
                            label="Rerank top n",
                            info="Number of rerank results",
                            interactive=True,
                        )
                    with gr.Row():
                        vector_search_top_k = gr.Slider(
                            1,
                            50,
                            value=15,
                            step=1,
                            label="Search top k",
                            info="Search top k must >= Rerank top n",
                            interactive=True,
                        )
    docs.clear(clear_files, outputs=[langchain_status], queue=False)
    load_docs.click(
        create_vectordb,
        inputs=[docs, spliter, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        outputs=[langchain_status],
        queue=False,
    )
    submit_event = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot,
        [chatbot, temperature, top_p, top_k, repetition_penalty, hide_context, do_rag],
        chatbot,
        queue=True,
    )
    submit_click_event = submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot,
        [chatbot, temperature, top_p, top_k, repetition_penalty, hide_context, do_rag],
        chatbot,
        queue=True,
    )
    stop.click(
        fn=request_cancel,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    vector_search_top_k.release(
        update_retriever,
        [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        outputs=[langchain_status],
    )
    vector_rerank_top_n.release(
        update_retriever,
        inputs=[vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        outputs=[langchain_status],
    )
    do_rerank.change(
        update_retriever,
        inputs=[vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        outputs=[langchain_status],
    )
    search_method.change(
        update_retriever,
        inputs=[vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        outputs=[langchain_status],
    )
    score_threshold.change(
        update_retriever,
        inputs=[vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
        outputs=[langchain_status],
    )


demo.queue()
# if you are launching remotely, specify server_name and server_port
#  demo.launch(server_name='your server name', server_port='server port in int')
# if you have any issue to launch on your platform, you can pass share=True to launch method:
# demo.launch(share=True)
# it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
try:
    demo.launch()
except Exception:
    demo.launch(share=True)