from typing import List
import io
from pypdf import PdfReader
from docx import Document

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks if chunks else [text]

def extract_text_from_file(content: bytes, filename: str) -> str:
    """Extract text from various file types"""
    if filename.lower().endswith('.pdf'):
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()

    elif filename.lower().endswith('.docx'):
        doc = Document(io.BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()

    elif filename.lower().endswith(('.txt', '.md', '.csv', '.json', '.xml')):
        return content.decode('utf-8')

    else:
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported file type: {filename}")
