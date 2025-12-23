from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Annotated
import operator
from databases.vector_store import VectorStore
from databases.graph_store import GraphStore
from databases.sql_store import SQLStore
from agents.classifier_agent import QueryClassifier
from langchain_openai import ChatOpenAI
from models.schemas import QueryType
from config import get_settings

# Define State
class RetrievalState(TypedDict):
    question: str
    query_type: str
    vector_results: List[Dict]
    graph_results: List[Dict]
    sql_results: List[Dict]
    merged_context: str
    final_answer: str

class BrainWorkflow:
    """LangGraph workflow for multi-DB retrieval"""

    def __init__(self):
        settings = get_settings()
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()
        self.sql_store = SQLStore()
        self.classifier = QueryClassifier()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )

        # Build graph
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(RetrievalState)

        # Add nodes
        workflow.add_node("classify", self._classify_query)
        workflow.add_node("retrieve_vector", self._retrieve_from_vector)
        workflow.add_node("retrieve_graph", self._retrieve_from_graph)
        workflow.add_node("retrieve_sql", self._retrieve_from_sql)
        workflow.add_node("merge_results", self._merge_results)
        workflow.add_node("generate_answer", self._generate_answer)

        # Add edges
        workflow.set_entry_point("classify")

        # From classify, go to all retrieval nodes
        workflow.add_edge("classify", "retrieve_vector")
        workflow.add_edge("classify", "retrieve_graph")
        workflow.add_edge("classify", "retrieve_sql")

        # All retrieval nodes go to merge
        workflow.add_edge("retrieve_vector", "merge_results")
        workflow.add_edge("retrieve_graph", "merge_results")
        workflow.add_edge("retrieve_sql", "merge_results")

        workflow.add_edge("merge_results", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def _classify_query(self, state: RetrievalState) -> RetrievalState:
        """Classify the query type"""
        query_type = self.classifier.classify(state["question"])
        state["query_type"] = query_type.value
        print(f"Query classified as: {query_type.value}")
        return state

    def _retrieve_from_vector(self, state: RetrievalState) -> RetrievalState:
        """Retrieve from vector database"""
        if state["query_type"] in [QueryType.SEMANTIC.value, QueryType.HYBRID.value]:
            print("Retrieving from vector database...")
            results = self.vector_store.similarity_search(state["question"], k=5)
            state["vector_results"] = [
                {"content": doc.page_content, "metadata": doc.metadata, "score": score}
                for doc, score in results
            ]
        else:
            state["vector_results"] = []
        return state

    def _retrieve_from_graph(self, state: RetrievalState) -> RetrievalState:
        """Retrieve from graph database"""
        if state["query_type"] in [QueryType.RELATIONAL.value, QueryType.HYBRID.value]:
            print("Retrieving from graph database...")
            # Extract potential entities from question
            # For now, simplified - in production use NER
            graph_results = []

            # Try to find any entity mentions and get related docs
            # This is a simplified approach
            if self.graph_store.available:
                try:
                    # Get some graph data (simplified)
                    docs = self.graph_store.get_knowledge_graph(limit=10)
                    if docs:
                        graph_results = [{"content": str(doc), "source": "graph"} for doc in docs[:3]]
                except Exception as e:
                    print(f"Graph retrieval error: {e}")

            state["graph_results"] = graph_results
        else:
            state["graph_results"] = []
        return state

    def _retrieve_from_sql(self, state: RetrievalState) -> RetrievalState:
        """Retrieve from SQL database"""
        if state["query_type"] in [QueryType.STRUCTURED.value, QueryType.HYBRID.value]:
            print("Retrieving from SQL database...")
            # For now, get recent documents
            # In production, parse filters from question
            docs = self.sql_store.get_all_documents(limit=10)
            state["sql_results"] = [
                {"content": doc.content[:300], "metadata": doc.doc_metadata, "id": doc.id}
                for doc in docs
            ]
        else:
            state["sql_results"] = []
        return state

    def _merge_results(self, state: RetrievalState) -> RetrievalState:
        """Merge results from all databases"""
        all_contexts = []

        # Add vector results
        for result in state.get("vector_results", []):
            all_contexts.append(f"[Vector Search] {result['content']}")

        # Add graph results
        for result in state.get("graph_results", []):
            content = result.get("content", "")
            if content:
                all_contexts.append(f"[Graph Relationship] {content}")

        # Add SQL results
        for result in state.get("sql_results", [])[:3]:  # Limit SQL results
            all_contexts.append(f"[Structured Data] {result['content']}")

        state["merged_context"] = "\n\n".join(all_contexts)
        print(f"Merged {len(all_contexts)} results from databases")
        return state

    def _generate_answer(self, state: RetrievalState) -> RetrievalState:
        """Generate final answer using LLM"""
        prompt = f"""You are a personal knowledge assistant. Answer the question based on the provided context.

Context from multiple sources:
{state['merged_context']}

Question: {state['question']}

Provide a comprehensive answer based on the context. If the context doesn't contain relevant information, say so."""

        print("Generating answer...")
        response = self.llm.invoke(prompt)
        state["final_answer"] = response.content
        return state

    def run(self, question: str) -> Dict:
        """Run the workflow"""
        initial_state = {
            "question": question,
            "query_type": "",
            "vector_results": [],
            "graph_results": [],
            "sql_results": [],
            "merged_context": "",
            "final_answer": ""
        }

        print(f"\n=== Starting BrainWorkflow for question: {question} ===")
        final_state = self.workflow.invoke(initial_state)
        print("=== Workflow completed ===\n")
        return final_state
