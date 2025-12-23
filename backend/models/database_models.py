from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

# Association table for many-to-many relationship
document_tags = Table(
    'document_tags',
    Base.metadata,
    Column('document_id', String, ForeignKey('documents.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

class Document(Base):
    """SQL table for document metadata"""
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    document_type = Column(String)  # 'text', 'file', 'url'
    filename = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    doc_metadata = Column(JSON)

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    tags = relationship("Tag", secondary=document_tags, back_populates="documents")
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    """SQL table for chunk metadata"""
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"))
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer)
    vector_id = Column(String)  # Reference to ChromaDB

    document = relationship("Document", back_populates="chunks")

class Tag(Base):
    """Tags for filtering"""
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)

    documents = relationship("Document", secondary=document_tags, back_populates="tags")

class Entity(Base):
    """Extracted entities for graph DB"""
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id"))
    entity_name = Column(String, nullable=False)
    entity_type = Column(String)  # 'person', 'organization', 'concept', etc.

    document = relationship("Document", back_populates="entities")
