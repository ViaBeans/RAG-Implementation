# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH =  os.getenv("DATA_PATH")

RCTS = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

def generate_data_store():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    chunks = RCTS.split_documents(documents)
    #persists when you make it by default
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

generate_data_store()