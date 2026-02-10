from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from models.schemas import TextInput, Question, StorageResult, RetrievalResult
from databases.vector_store import VectorStore
from databases.graph_store import GraphStore
from databases.sqlite_graph_store import SQLiteGraphStore
from utils.entity_extraction import EntityExtractor
from utils.text_processing import chunk_text, extract_text_from_file
from utils.token_counter import count_tokens, count_tokens_for_chunks
from langchain_openai import ChatOpenAI
from config import get_settings
import httpx
import uuid
import json
import asyncio

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

# Initialize graph store based on configuration
try:
    if settings.GRAPH_STORE_TYPE == "sqlite":
        graph_store = SQLiteGraphStore(db_path=settings.SQLITE_GRAPH_PATH)
        if graph_store.available:
            print(f"✓ SQLite graph store initialized at {settings.SQLITE_GRAPH_PATH}")
        else:
            print("⚠ SQLite graph store not available (optional)")
    elif settings.GRAPH_STORE_TYPE == "neo4j":
        graph_store = GraphStore()
        if graph_store.available:
            print("✓ Neo4j graph store initialized")
        else:
            print("⚠ Neo4j graph store not available (optional)")
    else:
        print(f"⚠ Unknown graph store type: {settings.GRAPH_STORE_TYPE}, using SQLite")
        graph_store = SQLiteGraphStore(db_path=settings.SQLITE_GRAPH_PATH)
        if graph_store.available:
            print(f"✓ SQLite graph store initialized (fallback)")
except Exception as e:
    print(f"⚠ Graph store initialization failed: {e}")
    # Create a dummy graph store that's not available
    graph_store = type('obj', (object,), {'available': False})()

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


# helper: call Ollama for LLM inference
async def call_ollama(prompt: str) -> str:
    """Send prompt to local Ollama instance and return the response text"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.OLLAMA_URL}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


#api endpoints

#inputting text
@app.post("/api/add-text", response_model=StorageResult)
async def add_text(input_data: TextInput):
    """Store text in vector DB (always) and graph DB (if entities found)"""
    try:
        text = input_data.text
        doc_id = f"doc_{uuid.uuid4()}"

        # 1. Always store in Vector DB
        chunks = chunk_text(text)
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        # Count tokens
        token_info = count_tokens_for_chunks(chunks, model="gpt-4")
        total_tokens = token_info["total_tokens"]

        vector_store.add_documents(
            texts=chunks,
            metadatas=[{"doc_id": doc_id, "chunk_index": i, "token_count": token_info["chunk_tokens"][i]}
                      for i in range(len(chunks))],
            ids=chunk_ids
        )
        print(f"✓ Stored in vector DB: {len(chunks)} chunks ({total_tokens} tokens)")

        # 2. Extract entities and store in Graph DB
        entities_stored = 0
        if graph_store.available:
            entities = entity_extractor.extract(text)
            if entities:
                # Create document node
                graph_store.create_document_node(doc_id, text[:500], {"token_count": total_tokens})

                # Create entity nodes and relationships
                for entity in entities:
                    graph_store.create_entity_node(entity["name"], entity["type"])
                    graph_store.create_relationship(doc_id, entity["name"])

                entities_stored = len(entities)
                print(f"✓ Stored in graph DB: {entities_stored} entities")

        return StorageResult(
            status="success",
            message=f"Stored in vector DB ({len(chunks)} chunks, {total_tokens} tokens)" +
                   (f" and graph DB ({entities_stored} entities)" if entities_stored > 0 else ""),
            token_count=total_tokens,
            chunk_count=len(chunks)
        )

    except Exception as e:
        print(f"Storage error: {e}")
        return StorageResult(
            status="error",
            message=str(e)
        )

#inputting files
@app.post("/api/add-file", response_model=StorageResult)
async def add_file(file: UploadFile = File(...)):
    """Parse uploaded file, store content in vector DB and graph DB"""
    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename)

        doc_id = f"doc_{uuid.uuid4()}"

        # 1. Store in Vector DB
        chunks = chunk_text(text)
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        # Count tokens
        token_info = count_tokens_for_chunks(chunks, model="gpt-4")
        total_tokens = token_info["total_tokens"]

        vector_store.add_documents(
            texts=chunks,
            metadatas=[{"doc_id": doc_id, "chunk_index": i, "filename": file.filename, "token_count": token_info["chunk_tokens"][i]}
                      for i in range(len(chunks))],
            ids=chunk_ids
        )
        print(f"✓ Stored file '{file.filename}' in vector DB: {len(chunks)} chunks ({total_tokens} tokens)")

        # 2. Extract entities and store in Graph DB
        entities_stored = 0
        if graph_store.available:
            entities = entity_extractor.extract(text)
            if entities:
                graph_store.create_document_node(doc_id, text[:500], {"filename": file.filename, "token_count": total_tokens})
                for entity in entities:
                    graph_store.create_entity_node(entity["name"], entity["type"])
                    graph_store.create_relationship(doc_id, entity["name"])
                entities_stored = len(entities)
                print(f"✓ Stored in graph DB: {entities_stored} entities")

        return StorageResult(
            status="success",
            message=f"Stored '{file.filename}' ({len(chunks)} chunks, {total_tokens} tokens)" +
                   (f" and graph DB ({entities_stored} entities)" if entities_stored > 0 else ""),
            token_count=total_tokens,
            chunk_count=len(chunks)
        )

    except Exception as e:
        print(f"File storage error: {e}")
        return StorageResult(
            status="error",
            message=str(e)
        )

#asking questions / querying database
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

        # Count tokens
        query_tokens = count_tokens(question, model="gpt-4")
        context_tokens = count_tokens(context_str, model="gpt-4")

        # Route to the requested LLM provider
        print(f"Generating answer via {question_data.llm_provider}...")
        if question_data.llm_provider == "ollama":
            answer = await call_ollama(prompt)
        else:
            response = llm.invoke(prompt)
            answer = response.content

        # Count answer tokens
        answer_tokens = count_tokens(answer, model="gpt-4")
        total_tokens = query_tokens + context_tokens + answer_tokens

        print(f"✓ Token usage - Query: {query_tokens}, Context: {context_tokens}, Answer: {answer_tokens}, Total: {total_tokens}")

        return RetrievalResult(
            answer=answer,
            sources=vector_sources[:3],
            status="success",
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            answer_tokens=answer_tokens,
            total_tokens=total_tokens
        )

    except Exception as e:
        print(f"Query error: {e}")
        return RetrievalResult(
            answer="",
            sources=[],
            status="error",
            message=str(e)
        )

#asking questions / querying database with streaming
@app.post("/api/ask-stream")
async def ask_question_stream(question_data: Question):
    """Answer questions with streaming response"""

    async def generate_stream():
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
                try:
                    graph_data = graph_store.get_knowledge_graph(limit=10)
                    if graph_data:
                        for record in graph_data[:3]:
                            graph_context.append(f"[Graph] {str(record)}")
                    print(f"✓ Found {len(graph_context)} graph results")
                except Exception as e:
                    print(f"Graph query error: {e}")

            all_context.extend(graph_context)

            # Send metadata first
            query_tokens = count_tokens(question, model="gpt-4")
            context_str = "\n\n".join(all_context) if all_context else ""
            context_tokens = count_tokens(context_str, model="gpt-4")

            metadata = {
                "type": "metadata",
                "sources": vector_sources[:3],
                "query_tokens": query_tokens,
                "context_tokens": context_tokens
            }
            yield f"data: {json.dumps(metadata)}\n\n"

            # 3. Generate answer using LLM with streaming
            if not all_context:
                answer_chunk = {
                    "type": "content",
                    "content": "I don't have any information to answer that question."
                }
                yield f"data: {json.dumps(answer_chunk)}\n\n"

                done_chunk = {
                    "type": "done",
                    "answer_tokens": 0,
                    "total_tokens": query_tokens + context_tokens
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                return

            prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context_str}

Question: {question}

Answer:"""

            # Route to the requested LLM provider
            print(f"Generating answer via {question_data.llm_provider}...")

            full_answer = ""

            if question_data.llm_provider == "ollama":
                # Stream from Ollama
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{settings.OLLAMA_URL}/api/chat",
                        json={
                            "model": settings.OLLAMA_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": True
                        },
                        timeout=120.0
                    ) as response:
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    if "message" in data and "content" in data["message"]:
                                        content = data["message"]["content"]
                                        full_answer += content
                                        chunk = {
                                            "type": "content",
                                            "content": content
                                        }
                                        yield f"data: {json.dumps(chunk)}\n\n"
                                except json.JSONDecodeError:
                                    continue
            else:
                # Stream from OpenAI
                stream = llm.stream(prompt)
                for chunk in stream:
                    content = chunk.content
                    full_answer += content
                    answer_chunk = {
                        "type": "content",
                        "content": content
                    }
                    yield f"data: {json.dumps(answer_chunk)}\n\n"

            # Send completion with token counts
            answer_tokens = count_tokens(full_answer, model="gpt-4")
            total_tokens = query_tokens + context_tokens + answer_tokens

            print(f"✓ Token usage - Query: {query_tokens}, Context: {context_tokens}, Answer: {answer_tokens}, Total: {total_tokens}")

            done_chunk = {
                "type": "done",
                "answer_tokens": answer_tokens,
                "total_tokens": total_tokens
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"

        except Exception as e:
            print(f"Streaming error: {e}")
            error_chunk = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# getting history of inputted data
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
                    "chunks": [],
                    "token_count": 0
                }
            docs_map[doc_id]["chunks"].append(results['documents'][i])
            # Add token count from metadata if available
            chunk_tokens = metadata.get('token_count', 0)
            docs_map[doc_id]["token_count"] += chunk_tokens

        # Combine chunks for each document
        inputs = []
        for doc_id, doc_data in docs_map.items():
            inputs.append({
                "id": doc_id,
                "content": " ".join(doc_data["chunks"])[:500],  # First 500 chars
                "chunk_count": len(doc_data["chunks"]),
                "token_count": doc_data["token_count"]
            })

        return {"status": "success", "inputs": inputs, "count": len(inputs)}
    except Exception as e:
        print(f"Error getting inputs: {e}")
        return {"status": "error", "message": str(e), "inputs": [], "count": 0}

# search feature for inputted data
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

# reset databases
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

# check llm providers (offline or api)
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

# affirm health
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


@app.get("/api/graph/stats")
async def graph_stats():
    """Get statistics about the graph database"""
    if not graph_store.available:
        return {
            "available": False,
            "message": "Graph store is not available"
        }

    try:
        # Check if graph store has get_stats method
        if hasattr(graph_store, 'get_stats'):
            stats = graph_store.get_stats()
            return {
                "available": True,
                "store_type": settings.GRAPH_STORE_TYPE,
                **stats
            }
        else:
            return {
                "available": True,
                "store_type": settings.GRAPH_STORE_TYPE,
                "message": "Stats not available for this graph store type"
            }
    except Exception as e:
        return {
            "available": True,
            "error": str(e)
        }


@app.get("/api/graph/visualization")
async def graph_visualization(limit: int = 100):
    """Get knowledge graph data for visualization"""
    if not graph_store.available:
        return {
            "available": False,
            "nodes": [],
            "edges": [],
            "message": "Graph store is not available"
        }

    try:
        # Get knowledge graph
        kg = graph_store.get_knowledge_graph(limit=limit)

        # Format for frontend visualization
        if isinstance(kg, dict):
            nodes = kg.get('nodes', [])
            relationships = kg.get('relationships', [])
        else:
            # Handle Neo4j result format if needed
            nodes = []
            relationships = []

        # Format nodes for visualization
        formatted_nodes = []
        for node in nodes:
            formatted_nodes.append({
                "id": node.get('id', node.get('name', '')),
                "label": node.get('name', node.get('id', '')),
                "type": node.get('type', 'unknown'),
                "group": node.get('type', 'unknown')  # For color grouping
            })

        # Format edges for visualization
        formatted_edges = []
        for rel in relationships:
            formatted_edges.append({
                "from": rel.get('source', ''),
                "to": rel.get('target', ''),
                "label": rel.get('type', 'RELATED'),
                "weight": rel.get('weight', 1.0)
            })

        return {
            "available": True,
            "nodes": formatted_nodes,
            "edges": formatted_edges,
            "stats": {
                "node_count": len(formatted_nodes),
                "edge_count": len(formatted_edges)
            }
        }

    except Exception as e:
        print(f"Error getting graph visualization: {e}")
        return {
            "available": False,
            "nodes": [],
            "edges": [],
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
