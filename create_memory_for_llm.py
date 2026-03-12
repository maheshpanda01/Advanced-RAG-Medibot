from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()

 #Reading the PDF File
file_path = "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND (2).pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# Preserve page numbers
for doc in docs:
    doc.metadata["page_label"] = doc.metadata.get("page", "Unknown")

#Splitting The Parsed Data into chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

texts = text_splitter.split_documents(docs)
                       

# Embedding Model for Vector Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
)
 
#Using embedding_model for creating embedding of split docs and storing them in faiss
DB_FAISS_PATH="vectorstore/db-faiss"
os.makedirs("vectorstore", exist_ok=True)

db = FAISS.from_documents(texts, embedding_model)
db.save_local(DB_FAISS_PATH)