from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer   
import os
import uuid
from typing import List
import google.generativeai as genai

load_dotenv()
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
HF_API_KEY = os.getenv("HF_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "localhost")
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "./data")
CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", "500"))
PORT = int(os.getenv("PORT", "8000"))

os.makedirs(RAG_DATA_DIR, exist_ok=True)

app = FastAPI()

def init_embedder(model_name: str):
    return SentenceTransformer(model_name)

def init_llm(api_key: str, model_name: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def init_chromadb(persistent_dir="chroma_data"):
    # client = chromadb.HttpClient(
    #     host = host,
    #     port = PORT,
    #     settings = Settings(anonymized_telemetry=False)
    # )
    client = Client(Settings(
        persist_directory=persistent_dir,
        anonymized_telemetry=False
    ))
    if "documents" not in [c.name for c in client.list_collections()]:
        client.create_collection("documents")
    return client

embedder = init_embedder(EMBED_MODEL_NAME)
llm = init_llm(GEMINI_API_KEY, LLM_MODEL_NAME)
collection = init_chromadb(CHROMA_DB_HOST)

def save_files(file: UploadFile, base_dir: str) -> str:
    content = file.file.read().decode("utf-8", errors="ignore")
    path = os.path.join(base_dir, file.filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return content

def semantic_chunk(text: str, chunk_length: int) -> List[str]:
    sentences = text.split('. ')
    chunks = []
    buffer = ""

    for sentence in sentences:
        if len(buffer) + len(sentence) + 2 <= chunk_length:
            buffer += sentence + '. '
        else:
            chunks.append(buffer.strip())
            buffer = sentence + '. '
    
    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    return embedder.encode(
        chunks,
        convert_to_numpy=True
    ).tolist()

def store_chunks(
        chunks: List[str], 
        embeddings: List[List[float]], 
        source: str
):
    ids = []
    metadatas = []
    for i in range(len(chunks)):
        ids.append(f"{source}-{i}-{str(uuid.uuid4())}")
        metadatas.append({"source": source})

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def retrieve_context(query: str, k: int = 5) -> List[str]:
    query_embedding = embed_chunks([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results.get("documents", [[]])[0]

def generate_response(query: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return "No relevant information found."
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a retrieval-augmented assistant. Use ONLY the provided context. 
    If answer is not contained in the context, say you don't know. 
    Context: {context}
    Question: {query}
    Answer:
    """
    response = llm.generate_content(prompt)
    return response.text

def delete_chunks():
    collection.delete(where={})

def load_documents(base_dir: str) -> dict:
    documents = {}

    for filename in os.listdir(base_dir):
        path = os.path.join(base_dir, filename)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            documents[filename] = f.read()
    return documents

class ChatPayload(BaseModel):
    query: str

class RechunkPayload(BaseModel):
    chunk_length: int

@app.post("/upload")
async def upload(file: UploadFile):
    # TODO: implement upload handling

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    contents = await file.read()
    text = contents.decode("utf-8", errors="ignore")

    file_path = os.path.join(RAG_DATA_DIR, file.filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    chunks = semantic_chunk(text, CHUNK_LENGTH)
    embeddings = embed_chunks(chunks)
    store_chunks(chunks, embeddings, file.filename)

    return{
        "filename": file.filename,
        "chunks_created": len(chunks)
    }

@app.post("/chat")
def chat(payload: ChatPayload):
    # TODO: implement prompt handling
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    context_chunks = retrieve_context(query)
    answer = generate_response(query, context_chunks)
    return {"answer": answer}

@app.post("/rechunk")
def rechunk(payload: RechunkPayload):
    # TODO: implement rechunk handling
    new_chunk_length = payload.chunk_length

    if new_chunk_length <= 0:
        raise HTTPException(status_code=400, detail="Chunk length must be positive")
    
    delete_chunks()
    documents = load_documents(RAG_DATA_DIR)
    total_chunks = 0
    for source, text in documents.items():
        chunks = semantic_chunk(text, new_chunk_length)
        embeddings = embed_chunks(chunks)
        store_chunks(chunks, embeddings, source)
        total_chunks += len(chunks)
    return {
        "status": "Rechunking completed",
        "new_chunk_length": new_chunk_length,
        "documents_processed": len(documents),
        "chunks_created": total_chunks
    }

@app.get("/health")
def health():
    return {"status": "alive"}