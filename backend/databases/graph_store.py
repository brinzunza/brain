from neo4j import GraphDatabase
from config import get_settings
from typing import List, Dict, Optional

class GraphStore:
    def __init__(self):
        self.settings = get_settings()
        try:
            self.driver = GraphDatabase.driver(
                self.settings.NEO4J_URI,
                auth=(self.settings.NEO4J_USERNAME, self.settings.NEO4J_PASSWORD)
            )
            # Test connection
            self.driver.verify_connectivity()
            self.available = True
        except Exception as e:
            print(f"Warning: Neo4j not available: {e}")
            print("Graph database features will be disabled. To enable:")
            print("1. Install Neo4j: docker run --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
            print("2. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env")
            self.driver = None
            self.available = False

    def close(self):
        if self.driver:
            self.driver.close()

    def create_document_node(self, doc_id: str, content: str, metadata: Dict):
        """Create a document node"""
        if not self.available:
            return
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (d:Document {id: $doc_id})
                    SET d.content = $content,
                        d.created_at = datetime(),
                        d.metadata = $metadata
                    """,
                    doc_id=doc_id,
                    content=content[:500],  # Store preview only
                    metadata=metadata
                )
        except Exception as e:
            print(f"Error creating document node: {e}")

    def create_entity_node(self, entity_name: str, entity_type: str):
        """Create an entity node"""
        if not self.available:
            return
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type
                    """,
                    name=entity_name,
                    type=entity_type
                )
        except Exception as e:
            print(f"Error creating entity node: {e}")

    def create_relationship(self, doc_id: str, entity_name: str, relationship_type: str = "MENTIONS"):
        """Create relationship between document and entity"""
        if not self.available:
            return
        try:
            with self.driver.session() as session:
                session.run(
                    f"""
                    MATCH (d:Document {{id: $doc_id}})
                    MATCH (e:Entity {{name: $entity_name}})
                    MERGE (d)-[r:{relationship_type}]->(e)
                    SET r.created_at = datetime()
                    """,
                    doc_id=doc_id,
                    entity_name=entity_name
                )
        except Exception as e:
            print(f"Error creating relationship: {e}")

    def find_related_entities(self, entity_name: str, max_depth: int = 2):
        """Find entities related to a given entity"""
        if not self.available:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH path = (e1:Entity {name: $name})-[*1..""" + str(max_depth) + """]->(e2:Entity)
                    RETURN e2.name as entity, e2.type as type, length(path) as distance
                    ORDER BY distance
                    LIMIT 20
                    """,
                    name=entity_name
                )
                return [{"entity": record["entity"], "type": record["type"], "distance": record["distance"]}
                        for record in result]
        except Exception as e:
            print(f"Error finding related entities: {e}")
            return []

    def find_documents_by_entity(self, entity_name: str):
        """Find all documents mentioning an entity"""
        if not self.available:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document)-[:MENTIONS]->(e:Entity {name: $name})
                    RETURN d.id as doc_id, d.content as preview
                    """,
                    name=entity_name
                )
                return [{"doc_id": record["doc_id"], "preview": record["preview"]}
                        for record in result]
        except Exception as e:
            print(f"Error finding documents by entity: {e}")
            return []

    def get_knowledge_graph(self, center_entity: Optional[str] = None, limit: int = 50):
        """Get knowledge graph for visualization"""
        if not self.available:
            return []
        try:
            with self.driver.session() as session:
                if center_entity:
                    query = """
                    MATCH path = (e1:Entity {name: $name})-[*1..2]-(e2:Entity)
                    RETURN e1, e2, relationships(path) as rels
                    LIMIT $limit
                    """
                    result = session.run(query, name=center_entity, limit=limit)
                else:
                    query = """
                    MATCH (e1:Entity)-[r]->(e2:Entity)
                    RETURN e1, e2, r
                    LIMIT $limit
                    """
                    result = session.run(query, limit=limit)

                return list(result)
        except Exception as e:
            print(f"Error getting knowledge graph: {e}")
            return []
