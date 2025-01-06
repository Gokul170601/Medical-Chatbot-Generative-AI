from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def load_pdf_file(data):
    loader=DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader,)
    documents=loader.load()
    return documents

def split_text(extracted_data):
    splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks=splitter.split_documents(extracted_data)
    return chunks

def huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings