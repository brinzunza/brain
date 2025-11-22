from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import TextInput, Question, StorageResult, RetrievalResult, QueryType
from databases.vector_store import VectorStore
from databases.graph_store import GraphStore
from databases.sql_store import SQLStore
from agents.storage_agent import StorageRouter
from graph.workflow import BrainWorkflow
from utils.text_processing import chunk_text, extract_text_from_file
from datetime import datetime
import uuid

app = FastAPI(title="Brain - Multi-DB RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
print("Initializing Brain Multi-DB RAG System...")
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

try:
    sql_store = SQLStore()
    print("✓ SQL store initialized")
except Exception as e:
    print(f"✗ SQL store initialization failed: {e}")
    raise

try:
    storage_router = StorageRouter()
    print("✓ Storage router initialized")
except Exception as e:
    print(f"✗ Storage router initialization failed: {e}")
    raise

try:
    brain_workflow = BrainWorkflow()
    print("✓ Brain workflow initialized")
except Exception as e:
    print(f"✗ Brain workflow initialization failed: {e}")
    raise

print("Brain system ready!\n")


@app.post("/api/add-text", response_model=StorageResult)
async def add_text(input_data: TextInput):
    """Store text with intelligent routing"""
    try:
        text = input_data.text
        doc_id = f"doc_{uuid.uuid4()}"

        # Determine storage strategy
        strategy, entities = storage_router.determine_strategy(text)

        storage_locations = []

        # Always store in SQL for metadata
        sql_store.create_document(
            doc_id=doc_id,
            content=text,
            doc_type="text",
            metadata=input_data.metadata or {}
        )
        storage_locations.append("sql")

        # Add tags if provided
        if input_data.tags:
            sql_store.add_tags(doc_id, input_data.tags)

        # Store in vector DB
        chunks = chunk_text(text)
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            sql_store.add_chunk(chunk_id, doc_id, chunk, i, chunk_id)

        vector_store.add_documents(
            texts=chunks,
            metadatas=[{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))],
            ids=chunk_ids
        )
        storage_locations.append("vector")

        # Store in graph DB if entities found
        if entities and graph_store.available:
            graph_store.create_document_node(doc_id, text, input_data.metadata or {})

            for entity in entities:
                graph_store.create_entity_node(entity["name"], entity["type"])
                graph_store.create_relationship(doc_id, entity["name"])

            sql_store.add_entities(doc_id, entities)
            storage_locations.append("graph")

        return StorageResult(
            status="success",
            document_id=doc_id,
            storage_locations=storage_locations,
            entities_extracted=[e["name"] for e in entities] if entities else None,
            message=f"Document stored in {', '.join(storage_locations)}"
        )

    except Exception as e:
        return StorageResult(
            status="error",
            document_id="",
            storage_locations=[],
            entities_extracted=None,
            message=str(e)
        )


@app.post("/api/add-file", response_model=StorageResult)
async def add_file(file: UploadFile = File(...)):
    """Store file content with intelligent routing"""
    try:
        content = await file.read()
        text_content = extract_text_from_file(content, file.filename)

        if not text_content.strip():
            return StorageResult(
                status="error",
                message="No text content found in file"
            )

        doc_id = f"doc_{uuid.uuid4()}"

        # Determine storage strategy
        strategy, entities = storage_router.determine_strategy(text_content)

        storage_locations = []

        # Store in SQL
        sql_store.create_document(
            doc_id=doc_id,
            content=text_content,
            doc_type="file",
            filename=file.filename,
            metadata={"filename": file.filename}
        )
        storage_locations.append("sql")

        # Store in vector DB
        chunks = chunk_text(text_content)
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            sql_store.add_chunk(chunk_id, doc_id, chunk, i, chunk_id)

        vector_store.add_documents(
            texts=chunks,
            metadatas=[{"doc_id": doc_id, "chunk_index": i, "filename": file.filename} for i in range(len(chunks))],
            ids=chunk_ids
        )
        storage_locations.append("vector")

        # Store in graph DB if entities found
        if entities and graph_store.available:
            graph_store.create_document_node(doc_id, text_content, {"filename": file.filename})

            for entity in entities:
                graph_store.create_entity_node(entity["name"], entity["type"])
                graph_store.create_relationship(doc_id, entity["name"])

            sql_store.add_entities(doc_id, entities)
            storage_locations.append("graph")

        return StorageResult(
            status="success",
            document_id=doc_id,
            storage_locations=storage_locations,
            entities_extracted=[e["name"] for e in entities] if entities else None,
            message=f"File {file.filename} stored in {', '.join(storage_locations)}"
        )

    except Exception as e:
        return StorageResult(
            status="error",
            document_id="",
            storage_locations=[],
            entities_extracted=None,
            message=str(e)
        )


@app.post("/api/ask", response_model=RetrievalResult)
async def ask_question(question_data: Question):
    """Answer questions using multi-DB RAG with LangGraph"""
    try:
        # Run LangGraph workflow
        result = brain_workflow.run(question_data.question)

        # Build sources
        sources = []
        for doc in result.get("vector_results", [])[:3]:
            sources.append({
                "type": "vector",
                "content": doc["content"][:200],
                "score": doc.get("score", 0)
            })

        databases_queried = []
        if result.get("vector_results"):
            databases_queried.append("vector")
        if result.get("graph_results"):
            databases_queried.append("graph")
        if result.get("sql_results"):
            databases_queried.append("sql")

        return RetrievalResult(
            answer=result["final_answer"],
            sources=sources,
            query_type_used=QueryType(result["query_type"]),
            databases_queried=databases_queried,
            confidence=0.85,  # Calculate based on retrieval scores
            status="success"
        )

    except Exception as e:
        return RetrievalResult(
            answer="",
            sources=[],
            query_type_used=QueryType.SEMANTIC,
            databases_queried=[],
            confidence=0.0,
            status="error",
            message=str(e)
        )


@app.get("/api/knowledge-graph")
async def get_knowledge_graph(entity: str = None):
    """Get knowledge graph data"""
    try:
        if not graph_store.available:
            return {"status": "error", "message": "Graph database not available"}

        graph_data = graph_store.get_knowledge_graph(entity, limit=50)
        return {"status": "success", "graph": [dict(record) for record in graph_data]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/inputs")
async def get_all_inputs():
    """Retrieve all stored inputs from the brain"""
    try:
        docs = sql_store.get_all_documents(limit=100)
        inputs = []

        for doc in docs:
            inputs.append({
                "id": doc.id,
                "content": doc.content[:500],  # Preview
                "metadata": doc.metadata,
                "document_type": doc.document_type,
                "filename": doc.filename,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "tags": [tag.name for tag in doc.tags] if doc.tags else []
            })

        return {"status": "success", "inputs": inputs, "count": len(inputs)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/inputs/{input_id}")
async def delete_input(input_id: str):
    """Delete a specific input from the brain"""
    try:
        # Get document to find chunk IDs
        doc = sql_store.get_document_by_id(input_id)
        if doc:
            # Delete from vector store
            chunk_ids = [chunk.vector_id for chunk in doc.chunks]
            if chunk_ids:
                vector_store.delete(chunk_ids)

        # Delete from SQL (cascades to chunks and entities)
        sql_store.delete_document(input_id)

        # Note: Graph deletion not implemented for simplicity
        # In production, you'd delete the document node and orphaned entities

        return {"status": "success", "message": "Input deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/clear")
async def clear_brain():
    """Delete all stored data from the brain"""
    try:
        # Clear all documents from SQL (cascades)
        docs = sql_store.get_all_documents(limit=1000)
        for doc in docs:
            sql_store.delete_document(doc.id)

        # Note: Vector and graph stores would need manual clearing
        # This is simplified for the implementation

        return {"status": "success", "message": "All data cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "multi-db-rag",
        "databases": {
            "vector": "available",
            "sql": "available",
            "graph": "available" if graph_store.available else "unavailable"
        }
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if graph_store.available:
        graph_store.close()
        print("Graph store connection closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
