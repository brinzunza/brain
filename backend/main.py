from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import os
from datetime import datetime
import json
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
import io
import requests

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize chromadb client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="brain_collection",
    metadata={"hnsw:space": "cosine"}
)

# initialize openai client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")


class TextInput(BaseModel):
    text: str


class Question(BaseModel):
    question: str
    llm_provider: str = "chatgpt"


def get_embedding(text: str):
    """generate embedding using openai"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200):
    """split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


@app.post("/api/add-text")
async def add_text(input_data: TextInput):
    """store text input in vector database"""
    try:
        text = input_data.text
        embedding = get_embedding(text)

        # generate unique id
        doc_id = f"text_{datetime.now().timestamp()}"

        # add to chromadb
        collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "type": "text",
                "timestamp": datetime.now().isoformat()
            }],
            ids=[doc_id]
        )

        return {"status": "success", "message": "text stored successfully", "id": doc_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def extract_text_from_file(content: bytes, filename: str) -> str:
    """extract text from different file types"""
    # pdf files
    if filename.lower().endswith('.pdf'):
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()

    # docx files
    elif filename.lower().endswith('.docx'):
        doc = Document(io.BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()

    # text files
    elif filename.lower().endswith(('.txt', '.md', '.csv', '.json', '.xml')):
        return content.decode('utf-8')

    # try utf-8 decode as fallback
    else:
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError(f"unsupported file type: {filename}")


@app.post("/api/add-file")
async def add_file(file: UploadFile = File(...)):
    """store file content in vector database"""
    try:
        content = await file.read()
        text_content = extract_text_from_file(content, file.filename)

        if not text_content.strip():
            return {"status": "error", "message": "no text content found in file"}

        chunks = chunk_text(text_content)
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)

            doc_id = f"file_{datetime.now().timestamp()}_{i}"

            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "type": "file",
                    "filename": file.filename,
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[doc_id]
            )
            chunk_ids.append(doc_id)

        return {"status": "success", "message": f"file {file.filename} stored successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_ollama(prompt: str, system_prompt: str):
    """query ollama local llm"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\n{prompt}",
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        raise Exception(f"ollama error: {str(e)}")


@app.post("/api/ask")
async def ask_question(question_data: Question):
    """answer questions based on stored data using rag"""
    try:
        question = question_data.question
        llm_provider = question_data.llm_provider

        # get embedding for question
        question_embedding = get_embedding(question)

        # query chromadb for relevant documents
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=5
        )

        # prepare context from retrieved documents
        context_docs = results['documents'][0] if results['documents'] else []
        context = "\n\n".join(context_docs)

        # generate answer using selected llm
        system_prompt = """you are a personal assistant helping the user recall information they've stored.
answer questions based only on the provided context. if you cannot find relevant information in the context,
say you don't have that information. keep answers concise and direct. use lowercase only."""

        user_prompt = f"""context from user's stored data:
{context}

question: {question}

answer based on the context above:"""

        if llm_provider == "ollama":
            # use ollama
            answer = query_ollama(user_prompt, system_prompt)
        else:
            # use openai/chatgpt
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            answer = response.choices[0].message.content

        return {
            "status": "success",
            "answer": answer,
            "sources": len(context_docs)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/clear")
async def clear_brain():
    """delete all stored data from the brain"""
    try:
        # delete the collection and recreate it
        chroma_client.delete_collection(name="brain_collection")

        global collection
        collection = chroma_client.get_or_create_collection(
            name="brain_collection",
            metadata={"hnsw:space": "cosine"}
        )

        return {"status": "success", "message": "all data cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
