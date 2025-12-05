from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class QueryType(str, Enum):
    SEMANTIC = "semantic"          # Vector similarity
    RELATIONAL = "relational"      # Graph relationships
    STRUCTURED = "structured"      # SQL filtering/aggregation
    HYBRID = "hybrid"              # Multiple databases

class StorageStrategy(str, Enum):
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    SQL_ONLY = "sql_only"
    FULL_HYBRID = "full_hybrid"

class TextInput(BaseModel):
    text: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class Question(BaseModel):
    question: str
    query_type: Optional[QueryType] = None  # Auto-detect if None
    filters: Optional[Dict[str, Any]] = None

class StorageResult(BaseModel):
    status: str
    document_id: str = ""
    storage_locations: List[str] = []
    entities_extracted: Optional[List[str]] = None
    message: Optional[str] = None

class RetrievalResult(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    query_type_used: QueryType
    databases_queried: List[str] = []
    confidence: float = 0.0
    status: str = "success"
    message: Optional[str] = None
