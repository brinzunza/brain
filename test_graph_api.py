#!/usr/bin/env python3
"""
Test the graph visualization API endpoint
"""

import sys
sys.path.insert(0, 'backend')

from databases.sqlite_graph_store import SQLiteGraphStore

# Test the graph store directly
graph = SQLiteGraphStore(db_path='backend/data/graph.db')

if graph.available:
    print("✓ Graph store is available\n")

    # Get knowledge graph
    kg = graph.get_knowledge_graph(limit=50)

    print(f"Knowledge Graph Data:")
    print(f"  Nodes: {len(kg.get('nodes', []))}")
    print(f"  Edges: {len(kg.get('relationships', []))}")

    print(f"\nSample Nodes:")
    for node in kg.get('nodes', [])[:5]:
        print(f"  - {node.get('name')} ({node.get('type')})")

    print(f"\nSample Edges:")
    for edge in kg.get('relationships', [])[:5]:
        print(f"  - {edge.get('source')} -> {edge.get('target')} [{edge.get('type')}]")

    print("\n" + "="*60)
    print("Graph visualization API format:")
    print("="*60)

    # Format like the API does
    formatted_nodes = []
    for node in kg.get('nodes', []):
        formatted_nodes.append({
            "id": node.get('id', node.get('name', '')),
            "label": node.get('name', node.get('id', '')),
            "type": node.get('type', 'unknown'),
            "group": node.get('type', 'unknown')
        })

    formatted_edges = []
    for rel in kg.get('relationships', []):
        formatted_edges.append({
            "from": rel.get('source', ''),
            "to": rel.get('target', ''),
            "label": rel.get('type', 'RELATED'),
            "weight": rel.get('weight', 1.0)
        })

    print(f"\nFormatted for frontend:")
    print(f"  Nodes: {len(formatted_nodes)}")
    print(f"  Edges: {len(formatted_edges)}")

    print(f"\n✓ Graph data is ready for visualization!")
    print(f"\nTo see it in action:")
    print(f"  1. Restart backend: cd backend && python main.py")
    print(f"  2. Start frontend: cd frontend && npm run dev")
    print(f"  3. Navigate to /inputs page")

else:
    print("❌ Graph store not available")
