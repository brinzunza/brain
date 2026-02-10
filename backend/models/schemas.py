from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class TextInput(BaseModel):
    text: str

class Question(BaseModel):
    question: str
    llm_provider: Optional[str] = "openai"

class StorageResult(BaseModel):
    status: str
    message: Optional[str] = None
    token_count: Optional[int] = None
    chunk_count: Optional[int] = None

class RetrievalResult(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    status: str = "success"
    message: Optional[str] = None
    query_tokens: Optional[int] = None
    context_tokens: Optional[int] = None
    answer_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
