from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from config import get_settings
from models.database_models import Base, Document, DocumentChunk, Tag, Entity
from typing import List, Dict, Optional
from datetime import datetime

class SQLStore:
    def __init__(self):
        self.settings = get_settings()
        self.engine = create_engine(self.settings.DATABASE_URL)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def create_document(self, doc_id: str, content: str, doc_type: str,
                       filename: Optional[str] = None, metadata: Optional[Dict] = None):
        """Create document record"""
        session = self.get_session()
        try:
            doc = Document(
                id=doc_id,
                content=content,
                document_type=doc_type,
                filename=filename,
                metadata=metadata or {}
            )
            session.add(doc)
            session.commit()
            return doc
        except Exception as e:
            session.rollback()
            print(f"Error creating document: {e}")
            raise
        finally:
            session.close()

    def add_chunk(self, chunk_id: str, doc_id: str, chunk_text: str,
                  chunk_index: int, vector_id: str):
        """Add chunk metadata"""
        session = self.get_session()
        try:
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=doc_id,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                vector_id=vector_id
            )
            session.add(chunk)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error adding chunk: {e}")
            raise
        finally:
            session.close()

    def add_tags(self, doc_id: str, tags: List[str]):
        """Add tags to document"""
        session = self.get_session()
        try:
            doc = session.query(Document).filter_by(id=doc_id).first()
            if not doc:
                return
            for tag_name in tags:
                tag = session.query(Tag).filter_by(name=tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    session.add(tag)
                if tag not in doc.tags:
                    doc.tags.append(tag)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error adding tags: {e}")
        finally:
            session.close()

    def add_entities(self, doc_id: str, entities: List[Dict]):
        """Add extracted entities"""
        session = self.get_session()
        try:
            for entity in entities:
                ent = Entity(
                    document_id=doc_id,
                    entity_name=entity["name"],
                    entity_type=entity["type"]
                )
                session.add(ent)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error adding entities: {e}")
        finally:
            session.close()

    def search_documents(self, filters: Dict) -> List[Document]:
        """Search documents with SQL filters"""
        session = self.get_session()
        try:
            query = session.query(Document)

            if "tags" in filters:
                query = query.join(Document.tags).filter(Tag.name.in_(filters["tags"]))

            if "start_date" in filters:
                query = query.filter(Document.created_at >= filters["start_date"])

            if "end_date" in filters:
                query = query.filter(Document.created_at <= filters["end_date"])

            if "document_type" in filters:
                query = query.filter(Document.document_type == filters["document_type"])

            return query.all()
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
        finally:
            session.close()

    def get_all_documents(self, limit: int = 100) -> List[Document]:
        """Get all documents"""
        session = self.get_session()
        try:
            return session.query(Document).order_by(Document.created_at.desc()).limit(limit).all()
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
        finally:
            session.close()

    def delete_document(self, doc_id: str):
        """Delete document and related records"""
        session = self.get_session()
        try:
            doc = session.query(Document).filter_by(id=doc_id).first()
            if doc:
                session.delete(doc)
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error deleting document: {e}")
            raise
        finally:
            session.close()

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        session = self.get_session()
        try:
            return session.query(Document).filter_by(id=doc_id).first()
        except Exception as e:
            print(f"Error getting document: {e}")
            return None
        finally:
            session.close()
