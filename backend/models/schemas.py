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

class RetrievalResult(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    status: str = "success"
    message: Optional[str] = None
