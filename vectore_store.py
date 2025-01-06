from src.helper import load_pdf_file, split_text, huggingface_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

extracted_data = load_pdf_file(data="data/")
chunks = split_text(extracted_data)
embeddings = huggingface_embeddings()



pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "medibot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

doc_storing = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings,
)