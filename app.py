from flask import Flask,request, render_template
from src.helper import huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from dotenv import load_dotenv
import os


app = Flask(__name__)

load_dotenv()

GROG_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings=huggingface_embeddings()

index_name = "medibot"

doc_search=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,)

retriver=doc_search.as_retriever(
    search_type="similarity", search_kwaegs={"k":3})

llm = ChatGroq(temperature=0, groq_api_key=GROG_API_KEY, model_name="llama-3.1-70b-versatile")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ('human' ,"{input}") ] )


qanda_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriver,qanda_chain)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    # print(msg)
    response=rag_chain.invoke({"input":msg})
    # print(response['answer'])
    return str(response['answer'])

if __name__ == '__main__':
    app.run(debug=True)