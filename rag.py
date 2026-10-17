from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os 
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_chroma  import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader

Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vectorstore = Chroma(
    collection_name="tripmate_docs",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

def read_file_text(file_path: str) -> str:
    path=Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        
        for page in reader.pages:
            text+= page.extract_text() or ""
            text+="\n"
            
        return text
    elif suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
        
    raise ValueError("Unsupported file type.")


def add_document_to_rag(file_path: str):
    text = read_file_text(file_path)
    
    if not text.strip():
        raise ValueError("NO text could be extracted from this file")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )
    
    chunks = splitter.split_text(text)
    
    docs: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "source": Path(file_path).name
            }
        )
        
        for chunk in chunks
    ]
    
    vectorstore.add_documents(docs)
    
    return {
        "filename": Path(file_path).name,
        "chunks": len(docs)
    }
    
def retrieve_from_rag(query: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(
        query,
        k=k,
    )
    
    if not docs:
        return "No relevent uploaded document content found"

    results = []
    
    for i, docs in enumerate(docs, start=1):
        source = docs.metadata.get("source", "uploaded document")
        results.append(
            f"[Source {i}: {source}]\n{docs.page_content}"
        )
        
    return "\n\n".join(results)