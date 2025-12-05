import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import get_settings
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.DEFAULT_EMBEDDING_MODEL,
            openai_api_key=self.settings.OPENAI_API_KEY
        )

        self.client = chromadb.PersistentClient(
            path=self.settings.CHROMA_PERSIST_DIR
        )

        self.vectorstore = Chroma(
            client=self.client,
            collection_name="brain_vectors",
            embedding_function=self.embeddings
        )

    def add_documents(self, texts: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to vector store"""
        try:
            self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict] = None):
        """Search for similar documents"""
        try:
            return self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def delete(self, ids: List[str]):
        """Delete documents by IDs"""
        try:
            self.vectorstore.delete(ids=ids)
        except Exception as e:
            print(f"Error deleting from vector store: {e}")
            raise
