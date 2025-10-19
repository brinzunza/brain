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

# create separate collections for different embedding providers
# openai uses 1536 dimensions, ollama (nomic-embed-text) uses 768 dimensions
def get_collection(provider: str):
    """get the appropriate collection for the embedding provider"""
    collection_name = f"brain_collection_{provider}"
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

# initialize openai client (optional)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = None
if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        print(f"warning: failed to initialize openai client: {e}")
else:
    print("warning: OPENAI_API_KEY not found. openai features will be disabled.")

# ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")


class TextInput(BaseModel):
    text: str
    embedding_provider: str = "openai"


class FileInput(BaseModel):
    embedding_provider: str = "openai"


class Question(BaseModel):
    question: str
    llm_provider: str = "chatgpt"
    embedding_provider: str = "openai"


def get_embedding(text: str, provider: str = "openai"):
    """generate embedding using selected provider"""
    if provider == "ollama":
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={
                    "model": OLLAMA_EMBEDDING_MODEL,
                    "prompt": text
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            raise Exception(f"ollama embedding error: {str(e)}")
    else:
        # use openai
        if not openai_client:
            raise Exception("openai api key not configured. please set OPENAI_API_KEY environment variable or use ollama embeddings.")
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
        embedding_provider = input_data.embedding_provider
        embedding = get_embedding(text, embedding_provider)

        # generate unique id
        doc_id = f"text_{datetime.now().timestamp()}"

        # get the appropriate collection for this provider
        collection = get_collection(embedding_provider)

        # add to chromadb
        collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "type": "text",
                "timestamp": datetime.now().isoformat(),
                "embedding_provider": embedding_provider
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
async def add_file(file: UploadFile = File(...), embedding_provider: str = Form("openai")):
    """store file content in vector database"""
    try:
        content = await file.read()
        text_content = extract_text_from_file(content, file.filename)

        if not text_content.strip():
            return {"status": "error", "message": "no text content found in file"}

        chunks = chunk_text(text_content)
        chunk_ids = []

        # get the appropriate collection for this provider
        collection = get_collection(embedding_provider)

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk, embedding_provider)

            doc_id = f"file_{datetime.now().timestamp()}_{i}"

            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "type": "file",
                    "filename": file.filename,
                    "timestamp": datetime.now().isoformat(),
                    "embedding_provider": embedding_provider
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
        embedding_provider = question_data.embedding_provider

        # get embedding for question
        question_embedding = get_embedding(question, embedding_provider)

        # get the appropriate collection for this provider
        collection = get_collection(embedding_provider)

        # query chromadb for relevant documents
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=5
        )

        # prepare context from retrieved documents
        context_docs = []
        if results and results.get('documents') and len(results['documents']) > 0:
            context_docs = results['documents'][0]

        context = "\n\n".join(context_docs) if context_docs else ""

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
            if not openai_client:
                raise Exception("openai api key not configured. please set OPENAI_API_KEY environment variable or use ollama.")
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


@app.get("/api/inputs")
async def get_all_inputs():
    """retrieve all stored inputs from the brain"""
    try:
        # get all items from both collections
        inputs = []

        for provider in ["openai", "ollama"]:
            try:
                collection = get_collection(provider)
                results = collection.get()

                # format the data
                if results and results.get('ids') and len(results['ids']) > 0:
                    for i in range(len(results['ids'])):
                        inputs.append({
                            "id": results['ids'][i],
                            "content": results['documents'][i] if i < len(results['documents']) else "",
                            "metadata": results['metadatas'][i] if i < len(results['metadatas']) else {},
                            "provider": provider
                        })
            except Exception as e:
                print(f"warning: could not fetch from {provider} collection: {e}")
                continue

        return {"status": "success", "inputs": inputs, "count": len(inputs)}
    except Exception as e:
        import traceback
        print(f"error in get_all_inputs: {e}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}


@app.delete("/api/inputs/{input_id}")
async def delete_input(input_id: str):
    """delete a specific input from the brain"""
    try:
        # try to delete from both collections (one will have it)
        deleted = False
        for provider in ["openai", "ollama"]:
            try:
                collection = get_collection(provider)
                collection.delete(ids=[input_id])
                deleted = True
            except Exception as e:
                continue

        if deleted:
            return {"status": "success", "message": "input deleted"}
        else:
            return {"status": "error", "message": "input not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/clear")
async def clear_brain():
    """delete all stored data from the brain"""
    try:
        # delete both provider collections and recreate them
        for provider in ["openai", "ollama"]:
            try:
                collection_name = f"brain_collection_{provider}"
                chroma_client.delete_collection(name=collection_name)
                # recreate it
                get_collection(provider)
            except Exception as e:
                print(f"warning: could not clear {provider} collection: {e}")
                continue

        return {"status": "success", "message": "all data cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/providers")
async def get_available_providers():
    """check which providers are available"""
    providers = {
        "openai_available": openai_client is not None,
        "ollama_available": False
    }

    # check if ollama is available
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        providers["ollama_available"] = response.status_code == 200
    except:
        pass

    return {"status": "success", "providers": providers}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
