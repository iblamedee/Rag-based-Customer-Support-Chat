import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings


file_path = r"X:\api_trail\langchain\LLMs @PROVIDERHUB0.pdf"
loader = PyPDFLoader(file_path)
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(

chunk_size = 800,
chunk_overlap = 200

)

chunks = text_splitter.split_documents(documents=doc)



embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key= os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
    
)

vector_store = QdrantVectorStore.from_documents(
    documents = chunks,
    embedding= embedding_model,
    url = "http://localhost:6333",
    collection_name = "learning-rag4",
    force_recreate=True

)

print("Vector store created successfully!")