import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import jinja2  
from langchain.vectorstores import Chroma
from constants import Settings
persist_directory = "db"

def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

persist_directory = "db"

def get_vector_store(text_chunks):
    client = Settings(chroma_db_impl="duckdb+parquet", persist_directory="db/", anonymized_telemetry=False)
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    # Convert the text chunks to a list of Document objects.
    documents = [Document(page_content=text_chunk, metadata={}) for text_chunk in text_chunks]
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory="db/", client_settings=client)
    return vector_store



def get_conversation_chain(vector_store):
    from transformers import AutoModelForSeq2SeqLM
    llm = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def render_template(template_name, context):
    """Renders a template using Jinja2."""
    template_loader = jinja2.FileSystemLoader('templates')  # Make sure to place your template.html file in a 'templates' folder
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(template_name)
    return template.render(context)

def handle_user_input(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message_type = "user-message"  # Define the message type for user messages
        else:
            message_type = "bot-message"  # Define the message type for bot messages
        st.write(message.content, message_type)

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with Your own PDFs :books:')
    question = st.text_input("Ask anything to your PDF: ")

    if question:
        handle_user_input(question)

    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        pdf_files = st.file_uploader("Choose your PDF Files and Press OK", type=['pdf'], accept_multiple_files=True)

        if st.button("OK"):
            with st.spinner("Processing your PDFs..."):
                # Get PDF Text
                raw_text = get_pdf_text(pdf_files)

                # Get Text Chunks
                text_chunks = get_chunk_text(raw_text)

                # Create Vector Store
                vector_store = get_vector_store(text_chunks)
                st.write("DONE")

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == "__main__":
    main()