from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import TextInput, Question, StorageResult, RetrievalResult
from databases.vector_store import VectorStore
from databases.graph_store import GraphStore
from utils.entity_extraction import EntityExtractor
from utils.text_processing import chunk_text
from langchain_openai import ChatOpenAI
from config import get_settings
import uuid

app = FastAPI(title="Brain - Duo Storage RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
print("Initializing Brain Duo Storage System...")
settings = get_settings()

try:
    vector_store = VectorStore()
    print("✓ Vector store initialized")
except Exception as e:
    print(f"✗ Vector store initialization failed: {e}")
    raise

try:
    graph_store = GraphStore()
    if graph_store.available:
        print("✓ Graph store initialized")
    else:
        print("⚠ Graph store not available (optional)")
except Exception as e:
    print(f"⚠ Graph store initialization failed: {e}")
    graph_store.available = False

try:
    entity_extractor = EntityExtractor()
    print("✓ Entity extractor initialized")
except Exception as e:
    print(f"✗ Entity extractor initialization failed: {e}")
    raise

try:
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        openai_api_key=settings.OPENAI_API_KEY
    )
    print("✓ LLM initialized")
except Exception as e:
    print(f"✗ LLM initialization failed: {e}")
    raise

print("Brain system ready!\n")


@app.post("/api/add-text", response_model=StorageResult)
async def add_text(input_data: TextInput):
    """Store text in vector DB (always) and graph DB (if entities found)"""
    try:
        text = input_data.text
        doc_id = f"doc_{uuid.uuid4()}"

        # 1. Always store in Vector DB
        chunks = chunk_text(text)
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        vector_store.add_documents(
            texts=chunks,
            metadatas=[{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))],
            ids=chunk_ids
        )
        print(f"✓ Stored in vector DB: {len(chunks)} chunks")

        # 2. Extract entities and store in Graph DB
        entities_stored = 0
        if graph_store.available:
            entities = entity_extractor.extract(text)
            if entities:
                # Create document node
                graph_store.create_document_node(doc_id, text[:500], {})

                # Create entity nodes and relationships
                for entity in entities:
                    graph_store.create_entity_node(entity["name"], entity["type"])
                    graph_store.create_relationship(doc_id, entity["name"])

                entities_stored = len(entities)
                print(f"✓ Stored in graph DB: {entities_stored} entities")

        return StorageResult(
            status="success",
            message=f"Stored in vector DB ({len(chunks)} chunks)" +
                   (f" and graph DB ({entities_stored} entities)" if entities_stored > 0 else "")
        )

    except Exception as e:
        print(f"Storage error: {e}")
        return StorageResult(
            status="error",
            message=str(e)
        )


@app.post("/api/ask", response_model=RetrievalResult)
async def ask_question(question_data: Question):
    """Answer questions by querying both vector and graph databases"""
    try:
        question = question_data.question
        all_context = []

        # 1. Query Vector DB
        print(f"\nQuerying vector DB for: {question}")
        vector_results = vector_store.similarity_search(question, k=5)
        vector_sources = []
        for doc, score in vector_results:
            all_context.append(f"[Vector] {doc.page_content}")
            vector_sources.append({
                "type": "vector",
                "content": doc.page_content[:200],
                "score": score
            })
        print(f"✓ Found {len(vector_results)} vector results")

        # 2. Query Graph DB (if available)
        graph_context = []
        if graph_store.available:
            print("Querying graph DB...")
            # Simple graph query - get some graph data
            try:
                graph_data = graph_store.get_knowledge_graph(limit=10)
                if graph_data:
                    for record in graph_data[:3]:
                        graph_context.append(f"[Graph] {str(record)}")
                print(f"✓ Found {len(graph_context)} graph results")
            except Exception as e:
                print(f"Graph query error: {e}")

        all_context.extend(graph_context)

        # 3. Generate answer using LLM
        if not all_context:
            return RetrievalResult(
                answer="I don't have any information to answer that question.",
                sources=[],
                status="success"
            )

        context_str = "\n\n".join(all_context)
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context_str}

Question: {question}

Answer:"""

        print("Generating answer...")
        response = llm.invoke(prompt)

        return RetrievalResult(
            answer=response.content,
            sources=vector_sources[:3],
            status="success"
        )

    except Exception as e:
        print(f"Query error: {e}")
        return RetrievalResult(
            answer="",
            sources=[],
            status="error",
            message=str(e)
        )


@app.get("/api/inputs")
async def get_all_inputs():
    """Retrieve stored documents from vector database"""
    try:
        # Get all documents from ChromaDB
        collection = vector_store.vectorstore._collection
        results = collection.get()

        # Group by doc_id
        docs_map = {}
        for i, chunk_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            doc_id = metadata.get('doc_id', chunk_id)

            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "id": doc_id,
                    "content": results['documents'][i],
                    "chunks": []
                }
            docs_map[doc_id]["chunks"].append(results['documents'][i])

        # Combine chunks for each document
        inputs = []
        for doc_id, doc_data in docs_map.items():
            inputs.append({
                "id": doc_id,
                "content": " ".join(doc_data["chunks"])[:500],  # First 500 chars
                "chunk_count": len(doc_data["chunks"])
            })

        return {"status": "success", "inputs": inputs, "count": len(inputs)}
    except Exception as e:
        print(f"Error getting inputs: {e}")
        return {"status": "error", "message": str(e), "inputs": [], "count": 0}


@app.delete("/api/inputs/{input_id}")
async def delete_input(input_id: str):
    """Delete a specific document"""
    try:
        # Delete from vector store
        collection = vector_store.vectorstore._collection
        results = collection.get()

        # Find all chunk IDs for this document
        chunk_ids_to_delete = []
        for i, metadata in enumerate(results['metadatas']):
            if metadata.get('doc_id') == input_id:
                chunk_ids_to_delete.append(results['ids'][i])

        if chunk_ids_to_delete:
            vector_store.delete(chunk_ids_to_delete)

        # Delete from graph store if available
        if graph_store.available:
            try:
                with graph_store.driver.session() as session:
                    session.run("MATCH (d:Document {id: $doc_id}) DETACH DELETE d", doc_id=input_id)
            except:
                pass

        return {"status": "success", "message": "Document deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/clear")
async def clear_brain():
    """Delete all stored data"""
    try:
        # Clear vector database
        vector_store.clear_all()
        print("✓ Vector store cleared")

        # Clear graph database
        if graph_store.available:
            graph_store.clear_all()
            print("✓ Graph store cleared")

        return {"status": "success", "message": "All data cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/providers")
async def check_providers():
    """Check which LLM providers are available"""
    settings = get_settings()

    openai_available = bool(settings.OPENAI_API_KEY and
                           settings.OPENAI_API_KEY != "your-openai-api-key-here")

    ollama_available = False
    try:
        import httpx
        response = httpx.get(settings.OLLAMA_URL, timeout=2.0)
        ollama_available = response.status_code == 200
    except:
        pass

    return {
        "status": "success",
        "providers": {
            "openai_available": openai_available,
            "ollama_available": ollama_available
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "duo-storage-rag",
        "databases": {
            "vector": "available",
            "graph": "available" if graph_store.available else "unavailable"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
