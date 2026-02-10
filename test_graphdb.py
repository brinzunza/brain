#!/usr/bin/env python3
"""
GraphDB Test Script
Tests both Neo4j and SQLite graph store implementations
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from databases.sqlite_graph_store import SQLiteGraphStore
from databases.graph_store import GraphStore
from datetime import datetime


def test_connection(store_type="sqlite"):
    """Test basic connection to graph database"""
    print(f"\n=== Testing {store_type.upper()} Graph Store Connection ===")

    if store_type == "sqlite":
        graph = SQLiteGraphStore(db_path="data/test_graph.db")
    else:
        graph = GraphStore()

    if not graph.available:
        print(f"❌ {store_type.upper()} graph store is not available.")
        if store_type == "neo4j":
            print("   Check:")
            print("   1. Neo4j is running (docker start neo4j)")
            print("   2. Connection settings in backend/.env")
        return None

    print(f"✓ Successfully connected to {store_type.upper()} graph store")
    return graph


def test_document_creation(graph):
    """Test creating document nodes"""
    print("\n=== Testing Document Creation ===")

    doc_id = f"test_doc_{datetime.now().timestamp()}"
    content = "This is a test document about artificial intelligence and machine learning."
    metadata = {
        "source": "test",
        "timestamp": datetime.now().isoformat()
    }

    try:
        graph.create_document_node(doc_id, content, metadata)
        print(f"✓ Created document node: {doc_id}")
        return doc_id
    except Exception as e:
        print(f"❌ Failed to create document: {e}")
        return None


def test_entity_creation(graph):
    """Test creating entity nodes"""
    print("\n=== Testing Entity Creation ===")

    entities = [
        ("Python", "technology"),
        ("OpenAI", "organization"),
        ("San Francisco", "location"),
        ("Neural Networks", "concept"),
        ("Machine Learning", "concept"),
    ]

    created_entities = []
    for entity_name, entity_type in entities:
        try:
            graph.create_entity_node(entity_name, entity_type)
            print(f"✓ Created entity: {entity_name} ({entity_type})")
            created_entities.append(entity_name)
        except Exception as e:
            print(f"❌ Failed to create entity {entity_name}: {e}")

    return created_entities


def test_relationships(graph, doc_id, entities):
    """Test creating relationships between documents and entities"""
    print("\n=== Testing Relationships ===")

    if not doc_id or not entities:
        print("⚠ Skipping relationship test (missing doc or entities)")
        return

    for entity in entities[:3]:  # Link first 3 entities to document
        try:
            graph.create_relationship(doc_id, entity, "MENTIONS")
            print(f"✓ Created relationship: {doc_id} -> MENTIONS -> {entity}")
        except Exception as e:
            print(f"❌ Failed to create relationship: {e}")


def test_queries(graph, entities):
    """Test querying the graph"""
    print("\n=== Testing Queries ===")

    if not entities:
        print("⚠ Skipping query test (no entities)")
        return

    # Test find_related_entities
    test_entity = entities[0] if entities else None
    if test_entity:
        try:
            related = graph.find_related_entities(test_entity, max_depth=2)
            print(f"✓ Found {len(related)} related entities to '{test_entity}':")
            for rel in related[:5]:  # Show first 5
                print(f"  - {rel.get('entity', '?')} (distance: {rel.get('distance', '?')})")
        except Exception as e:
            print(f"❌ Failed to find related entities: {e}")

    # Test find_documents_by_entity
    if test_entity:
        try:
            docs = graph.find_documents_by_entity(test_entity)
            print(f"✓ Found {len(docs)} documents mentioning '{test_entity}'")
            for doc in docs[:3]:  # Show first 3
                print(f"  - {doc.get('doc_id', 'unknown')}")
        except Exception as e:
            print(f"❌ Failed to find documents by entity: {e}")


def test_knowledge_graph(graph):
    """Test retrieving the full knowledge graph"""
    print("\n=== Testing Knowledge Graph Retrieval ===")

    try:
        kg = graph.get_knowledge_graph()
        nodes = kg.get('nodes', []) if isinstance(kg, dict) else []
        rels = kg.get('relationships', []) if isinstance(kg, dict) else []

        print(f"✓ Retrieved knowledge graph:")
        print(f"  - Nodes: {len(nodes)}")
        print(f"  - Relationships: {len(rels)}")

        # Show sample nodes
        if nodes:
            print(f"\n  Sample nodes (first 5):")
            for node in nodes[:5]:
                node_name = node.get('name', node.get('id', 'unknown'))
                node_type = node.get('type', 'unknown')
                print(f"    - {node_name} ({node_type})")

        # Show sample relationships
        if rels:
            print(f"\n  Sample relationships (first 5):")
            for rel in rels[:5]:
                source = rel.get('source', '?')
                target = rel.get('target', '?')
                rel_type = rel.get('type', '?')
                print(f"    - {source} -> {rel_type} -> {target}")

    except Exception as e:
        print(f"❌ Failed to retrieve knowledge graph: {e}")


def test_stats(graph):
    """Test getting database statistics"""
    print("\n=== Testing Database Statistics ===")

    if hasattr(graph, 'get_stats'):
        try:
            stats = graph.get_stats()
            print("✓ Database statistics:")
            for key, value in stats.items():
                print(f"  - {key}: {value}")
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
    else:
        print("⚠ Stats not available for this graph store type")


def test_cleanup(graph):
    """Test clearing all data"""
    print("\n=== Testing Cleanup ===")

    response = input("Do you want to clear ALL graph data? (yes/no): ")
    if response.lower() == 'yes':
        try:
            graph.clear_all()
            print("✓ Successfully cleared all graph data")
        except Exception as e:
            print(f"❌ Failed to clear graph data: {e}")
    else:
        print("⚠ Skipped cleanup (data remains in graph)")


def main():
    """Run all tests"""
    print("=" * 60)
    print("GraphDB Test Suite")
    print("=" * 60)

    # Ask which store to test
    print("\nWhich graph store do you want to test?")
    print("1. SQLite (default)")
    print("2. Neo4j")
    choice = input("Enter choice (1 or 2): ").strip()

    store_type = "neo4j" if choice == "2" else "sqlite"

    # Test connection
    graph = test_connection(store_type)
    if not graph:
        print(f"\n❌ Cannot proceed without {store_type.upper()} connection")
        sys.exit(1)

    # Run tests
    doc_id = test_document_creation(graph)
    entities = test_entity_creation(graph)
    test_relationships(graph, doc_id, entities)
    test_queries(graph, entities)
    test_knowledge_graph(graph)
    test_stats(graph)

    # Cleanup
    test_cleanup(graph)

    # Close connection
    if hasattr(graph, 'close'):
        graph.close()

    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
