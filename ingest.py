import os
from PIL import Image
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader, RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders.image import UnstructuredImageLoader
from constants import CHROMA_SETTINGS

def extract_text_from_images(list_img):
    image_content = []
    for image_path in list_img:
        try:
            image = Image.open(image_path)
            image_text = process_image(image)
            image_content.append(image_text)
        except Exception as e:
            print(f"Error processing image: {image_path} - {str(e)}")
    return "\n".join(image_content)

def extract_text_from_pdf(pdf_file):
    pdf_page_content = ''
    for pdf_path in pdf_file:
        try:
            pdf_text = process_pdf(pdf_path)
            pdf_page_content += pdf_text
        except Exception as e:
            print(f"Error processing PDF: {pdf_path} - {str(e)}")
    return pdf_page_content

def process_image(image):
    loader = UnstructuredImageLoader(image)
    data = loader.load()
    return data[0].page_content  # Assuming only one image page

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return '\n'.join(doc.page_content for doc in documents)

def create_embeddings_and_store(docs):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chuck_overlap=200,
        length_function=len,
    )

    embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    db = Chroma.from_documents(docs, embeddings)
    return db

def main():
    docs_folder = "docs"
    
    # Find image and PDF files in the "docs" folder
    list_img = [os.path.join(docs_folder, filename) for filename in os.listdir(docs_folder) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    pdf_file = [os.path.join(docs_folder, filename) for filename in os.listdir(docs_folder) if filename.endswith('.pdf')]
    
    # Extract text from images and PDFs
    image_text = extract_text_from_images(list_img)
    pdf_text = extract_text_from_pdf(pdf_file)
    
    if image_text:
        create_embeddings_and_store([image_text])
    if pdf_text:
        create_embeddings_and_store([pdf_text])

if __name__ == "__main__":
    main()
