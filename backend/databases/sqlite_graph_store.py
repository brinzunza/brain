import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class SQLiteGraphStore:
    """
    SQLite-based graph database for storing documents, entities, and their relationships.
    Optimized for privacy, local-first architecture, and good performance up to millions of nodes.
    """

    def __init__(self, db_path: str = "data/graph.db"):
        self.db_path = db_path
        self.available = False

        try:
            # Ensure data directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Access columns by name

            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")

            # Performance optimizations
            self.conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better concurrency
            self.conn.execute("PRAGMA synchronous = NORMAL")  # Faster writes, still safe
            self.conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
            self.conn.execute("PRAGMA temp_store = MEMORY")  # Store temp tables in memory

            # Create tables
            self._create_tables()
            self.available = True

        except Exception as e:
            print(f"Warning: SQLite graph store not available: {e}")
            self.conn = None
            self.available = False

    def _create_tables(self):
        """Create database tables with proper indexes"""
        cursor = self.conn.cursor()

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Relationships table (document -> entity)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                relationship_type TEXT DEFAULT 'MENTIONS',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (entity_name) REFERENCES entities(name) ON DELETE CASCADE,
                UNIQUE(doc_id, entity_name, relationship_type)
            )
        """)

        # Entity-to-entity relationships (for knowledge graph)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                relationship_type TEXT DEFAULT 'RELATED_TO',
                weight REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity) REFERENCES entities(name) ON DELETE CASCADE,
                FOREIGN KEY (target_entity) REFERENCES entities(name) ON DELETE CASCADE,
                UNIQUE(source_entity, target_entity, relationship_type)
            )
        """)

        # Create indexes for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_doc ON relationships(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_entity ON relationships(entity_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_rel_source ON entity_relationships(source_entity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_rel_target ON entity_relationships(target_entity)")

        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def create_document_node(self, doc_id: str, content: str, metadata: Dict):
        """Create a document node"""
        if not self.available:
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO documents (id, content, metadata)
                VALUES (?, ?, ?)
                """,
                (doc_id, content[:500], json.dumps(metadata))  # Store preview only
            )
            self.conn.commit()
        except Exception as e:
            print(f"Error creating document node: {e}")
            self.conn.rollback()

    def create_entity_node(self, entity_name: str, entity_type: str):
        """Create an entity node"""
        if not self.available:
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO entities (name, type)
                VALUES (?, ?)
                """,
                (entity_name, entity_type)
            )
            self.conn.commit()
        except Exception as e:
            print(f"Error creating entity node: {e}")
            self.conn.rollback()

    def create_relationship(self, doc_id: str, entity_name: str, relationship_type: str = "MENTIONS"):
        """Create relationship between document and entity"""
        if not self.available:
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO relationships (doc_id, entity_name, relationship_type)
                VALUES (?, ?, ?)
                """,
                (doc_id, entity_name, relationship_type)
            )
            self.conn.commit()

            # Also create entity-to-entity relationships for entities in the same document
            self._update_entity_relationships(doc_id)

        except Exception as e:
            print(f"Error creating relationship: {e}")
            self.conn.rollback()

    def _update_entity_relationships(self, doc_id: str):
        """Create/strengthen relationships between entities that appear in the same document"""
        if not self.available:
            return

        try:
            cursor = self.conn.cursor()

            # Get all entities in this document
            cursor.execute(
                """
                SELECT entity_name FROM relationships WHERE doc_id = ?
                """,
                (doc_id,)
            )
            entities = [row[0] for row in cursor.fetchall()]

            # Create relationships between all pairs
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    # Insert or increment weight if relationship exists
                    cursor.execute(
                        """
                        INSERT INTO entity_relationships (source_entity, target_entity, relationship_type, weight)
                        VALUES (?, ?, 'CO_OCCURS', 1.0)
                        ON CONFLICT(source_entity, target_entity, relationship_type)
                        DO UPDATE SET weight = weight + 0.1
                        """,
                        (entity1, entity2)
                    )

            self.conn.commit()
        except Exception as e:
            print(f"Error updating entity relationships: {e}")
            self.conn.rollback()

    def find_related_entities(self, entity_name: str, max_depth: int = 2):
        """Find entities related to a given entity"""
        if not self.available:
            return []

        try:
            cursor = self.conn.cursor()

            if max_depth == 1:
                # Direct relationships only
                cursor.execute(
                    """
                    SELECT e.name as entity, e.type as type, 1 as distance, er.weight
                    FROM entity_relationships er
                    JOIN entities e ON (er.target_entity = e.name OR er.source_entity = e.name)
                    WHERE (er.source_entity = ? OR er.target_entity = ?)
                      AND e.name != ?
                    ORDER BY er.weight DESC, e.name
                    LIMIT 20
                    """,
                    (entity_name, entity_name, entity_name)
                )
            else:
                # Use recursive CTE for multi-hop queries (depth 2+)
                cursor.execute(
                    """
                    WITH RECURSIVE entity_path(entity, type, distance, weight, path) AS (
                        -- Base case: direct relationships
                        SELECT e.name, e.type, 1, er.weight, e.name
                        FROM entity_relationships er
                        JOIN entities e ON (er.target_entity = e.name OR er.source_entity = e.name)
                        WHERE (er.source_entity = ? OR er.target_entity = ?)
                          AND e.name != ?

                        UNION

                        -- Recursive case: relationships of related entities
                        SELECT e.name, e.type, ep.distance + 1, er.weight * 0.5, ep.path || ',' || e.name
                        FROM entity_path ep
                        JOIN entity_relationships er ON (er.source_entity = ep.entity OR er.target_entity = ep.entity)
                        JOIN entities e ON (er.target_entity = e.name OR er.source_entity = e.name)
                        WHERE ep.distance < ?
                          AND e.name != ?
                          AND ep.path NOT LIKE '%' || e.name || '%'
                    )
                    SELECT DISTINCT entity, type, MIN(distance) as distance, MAX(weight) as weight
                    FROM entity_path
                    GROUP BY entity
                    ORDER BY distance, weight DESC, entity
                    LIMIT 20
                    """,
                    (entity_name, entity_name, entity_name, max_depth, entity_name)
                )

            results = cursor.fetchall()
            return [
                {
                    "entity": row["entity"],
                    "type": row["type"],
                    "distance": row["distance"]
                }
                for row in results
            ]

        except Exception as e:
            print(f"Error finding related entities: {e}")
            return []

    def find_documents_by_entity(self, entity_name: str):
        """Find all documents mentioning an entity"""
        if not self.available:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT d.id as doc_id, d.content as preview
                FROM documents d
                JOIN relationships r ON d.id = r.doc_id
                WHERE r.entity_name = ?
                ORDER BY d.created_at DESC
                """,
                (entity_name,)
            )

            results = cursor.fetchall()
            return [
                {
                    "doc_id": row["doc_id"],
                    "preview": row["preview"]
                }
                for row in results
            ]

        except Exception as e:
            print(f"Error finding documents by entity: {e}")
            return []

    def get_knowledge_graph(self, center_entity: Optional[str] = None, limit: int = 50):
        """Get knowledge graph for visualization"""
        if not self.available:
            return {"nodes": [], "relationships": []}

        try:
            cursor = self.conn.cursor()

            if center_entity:
                # Get entities related to center entity
                cursor.execute(
                    """
                    WITH related_entities AS (
                        SELECT DISTINCT e.name, e.type
                        FROM entity_relationships er
                        JOIN entities e ON (er.target_entity = e.name OR er.source_entity = e.name)
                        WHERE (er.source_entity = ? OR er.target_entity = ?)
                        LIMIT ?
                    )
                    SELECT name, type FROM related_entities
                    UNION
                    SELECT name, type FROM entities WHERE name = ?
                    """,
                    (center_entity, center_entity, limit - 1, center_entity)
                )
            else:
                # Get all entities (limited)
                cursor.execute(
                    """
                    SELECT name, type FROM entities LIMIT ?
                    """,
                    (limit,)
                )

            nodes = [
                {
                    "id": row["name"],
                    "name": row["name"],
                    "type": row["type"]
                }
                for row in cursor.fetchall()
            ]

            # Get relationships between these nodes
            node_names = [node["name"] for node in nodes]
            if node_names:
                placeholders = ",".join("?" * len(node_names))
                cursor.execute(
                    f"""
                    SELECT source_entity, target_entity, relationship_type, weight
                    FROM entity_relationships
                    WHERE source_entity IN ({placeholders})
                      AND target_entity IN ({placeholders})
                    ORDER BY weight DESC
                    LIMIT ?
                    """,
                    node_names + node_names + [limit * 2]
                )

                relationships = [
                    {
                        "source": row["source_entity"],
                        "target": row["target_entity"],
                        "type": row["relationship_type"],
                        "weight": row["weight"]
                    }
                    for row in cursor.fetchall()
                ]
            else:
                relationships = []

            return {
                "nodes": nodes,
                "relationships": relationships
            }

        except Exception as e:
            print(f"Error getting knowledge graph: {e}")
            return {"nodes": [], "relationships": []}

    def delete_document(self, doc_id: str):
        """Delete a document and its relationships"""
        if not self.available:
            return

        try:
            cursor = self.conn.cursor()

            # Get entities that will be orphaned
            cursor.execute(
                """
                SELECT entity_name FROM relationships WHERE doc_id = ?
                """,
                (doc_id,)
            )
            entities = [row[0] for row in cursor.fetchall()]

            # Delete document (relationships cascade)
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

            # Clean up orphaned entities (entities with no remaining relationships)
            for entity in entities:
                cursor.execute(
                    """
                    SELECT COUNT(*) as count FROM relationships WHERE entity_name = ?
                    """,
                    (entity,)
                )
                if cursor.fetchone()["count"] == 0:
                    cursor.execute("DELETE FROM entities WHERE name = ?", (entity,))

            self.conn.commit()

        except Exception as e:
            print(f"Error deleting document: {e}")
            self.conn.rollback()

    def clear_all(self):
        """Delete all nodes and relationships from graph database"""
        if not self.available:
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM entity_relationships")
            cursor.execute("DELETE FROM relationships")
            cursor.execute("DELETE FROM entities")
            cursor.execute("DELETE FROM documents")
            self.conn.commit()
            print("SQLite graph store cleared successfully")
        except Exception as e:
            print(f"Error clearing SQLite graph store: {e}")
            self.conn.rollback()
            raise

    def get_stats(self):
        """Get statistics about the graph database"""
        if not self.available:
            return {}

        try:
            cursor = self.conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM entities")
            entity_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM relationships")
            rel_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM entity_relationships")
            entity_rel_count = cursor.fetchone()["count"]

            # Get database file size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            db_size_mb = (page_count * page_size) / (1024 * 1024)

            return {
                "documents": doc_count,
                "entities": entity_count,
                "doc_entity_relationships": rel_count,
                "entity_entity_relationships": entity_rel_count,
                "database_size_mb": round(db_size_mb, 2)
            }

        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
