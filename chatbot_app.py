import streamlit as st
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from streamlit_chat import message
from Blockchain import Blockchain
from Block import Block

# Initialize the blockchain
blockchain = Blockchain()

st.set_page_config(layout="wide")

device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float32
)

# Define the directory to save PDF files
persist_directory = "docs"
os.makedirs(persist_directory, exist_ok=True)

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None 


def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def summarize_pdf(pdf_file_path):
    loader = PDFMinerLoader(pdf_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    final_texts = "".join(text.page_content for text in texts)

    summarization_pipe = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        device=-1  # For CPU, use 0 for GPU
    )
    summary_text = summarization_pipe(final_texts, max_length=130, min_length=30, length_penalty=2.0, num_beams=4)[0]['summary_text']
    return summary_text

def add_summary_to_blockchain(summary, filename):
    transaction = {
        "filename": filename,
        "summary": summary,
        "timestamp": time.time()
    }
    blockchain.add_new_transaction(transaction)
    new_block = blockchain.mine()
    return new_block

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 512,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    # Assuming the function `qa_llm` is defined and correctly set up as provided in the original code.
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>Built By Bhanu </h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        file_details = {"Filename": uploaded_file.name, "File size": get_file_size(uploaded_file)}
        st.json(file_details)

        file_path = os.path.join(persist_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        displayPDF(file_path)

        summary = summarize_pdf(file_path)
        st.write(summary)
        new_block = add_summary_to_blockchain(summary, uploaded_file.name)
        if new_block:
            st.success(f"A new block has been added to the blockchain: {new_block}")
        else:
            st.error("Failed to mine a new block.")

    st.markdown("<h4 style='color:black;'>Chat Here</h4>", unsafe_allow_html=True)
    user_input = st.text_input("Enter your question here:", key="input")
    send_button = st.button(label="Send", key="send_button")
    
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    if send_button and user_input:
        answer = process_answer(user_input)
        st.session_state["past"].append(user_input)
        response = answer
        st.session_state["generated"].append(response)

    display_conversation(st.session_state)

if __name__ == "__main__":
    main()