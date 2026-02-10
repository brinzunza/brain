#!/usr/bin/env python3
"""
Diagnose why graph isn't working
"""

import sys
import os

# Change to project root
os.chdir('/Users/brunoinzunza/Documents/GitHub/brain/backend')
sys.path.insert(0, '.')

print("="*60)
print("GRAPH DATABASE DIAGNOSTIC")
print("="*60)

# 1. Check graph store
print("\n1. Checking graph store...")
try:
    from databases.sqlite_graph_store import SQLiteGraphStore
    graph = SQLiteGraphStore(db_path='./data/graph.db')
    if graph.available:
        stats = graph.get_stats()
        print(f"✓ Graph store available")
        print(f"  Entities: {stats.get('entities', 0)}")
        print(f"  Documents: {stats.get('documents', 0)}")
    else:
        print("❌ Graph store not available")
except Exception as e:
    print(f"❌ Error: {e}")

# 2. Check entity extractor
print("\n2. Checking entity extractor...")
try:
    from utils.entity_extraction import EntityExtractor
    extractor = EntityExtractor()
    print("✓ Entity extractor initialized")

    # Test extraction
    test_text = "OpenAI created GPT-4 using Machine Learning"
    entities = extractor.extract(test_text)
    print(f"✓ Extracted {len(entities)} entities from test text:")
    for e in entities:
        print(f"  - {e['name']} ({e['type']})")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# 3. Check config
print("\n3. Checking configuration...")
try:
    from config import get_settings
    settings = get_settings()
    print(f"✓ Graph store type: {settings.GRAPH_STORE_TYPE}")
    print(f"✓ Graph DB path: {settings.SQLITE_GRAPH_PATH}")
    print(f"✓ OpenAI API key: {'***' + settings.OPENAI_API_KEY[-4:] if settings.OPENAI_API_KEY else 'NOT SET'}")
except Exception as e:
    print(f"❌ Error: {e}")

# 4. Test full workflow
print("\n4. Testing full add-text workflow...")
try:
    from databases.vector_store import VectorStore
    from utils.text_processing import chunk_text
    import uuid

    # Initialize stores
    vector_store = VectorStore()
    graph_store = SQLiteGraphStore(db_path='./data/graph.db')
    extractor = EntityExtractor()

    # Test text
    text = "OpenAI created GPT-4 using Machine Learning in San Francisco"
    doc_id = f"test_doc_{uuid.uuid4()}"

    print(f"✓ Testing with text: '{text}'")
    print(f"✓ Document ID: {doc_id}")

    # Extract entities
    entities = extractor.extract(text)
    print(f"✓ Extracted {len(entities)} entities:")
    for e in entities:
        print(f"  - {e['name']} ({e['type']})")

    # Store in graph
    if graph_store.available and entities:
        graph_store.create_document_node(doc_id, text, {})
        for entity in entities:
            graph_store.create_entity_node(entity["name"], entity["type"])
            graph_store.create_relationship(doc_id, entity["name"])

        print(f"✓ Stored in graph database")

        # Check stats
        stats = graph_store.get_stats()
        print(f"✓ New stats:")
        print(f"  Entities: {stats.get('entities', 0)}")
        print(f"  Documents: {stats.get('documents', 0)}")
        print(f"  Relationships: {stats.get('doc_entity_relationships', 0)}")
    else:
        if not graph_store.available:
            print("❌ Graph store not available")
        if not entities:
            print("❌ No entities extracted")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
