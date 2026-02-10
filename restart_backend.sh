#!/bin/bash

echo "======================================"
echo "Restarting Brain Backend"
echo "======================================"

# Find and kill existing backend process
echo ""
echo "1. Stopping existing backend..."
pkill -f "python.*main.py" && echo "✓ Stopped old backend" || echo "⚠ No backend was running"

# Wait a moment
sleep 1

# Start new backend
echo ""
echo "2. Starting new backend..."
cd backend
python3 main.py &

echo ""
echo "======================================"
echo "Backend restart complete!"
echo "======================================"
echo ""
echo "The backend is now running with:"
echo "  ✓ Fixed entity extraction"
echo "  ✓ Graph visualization endpoint"
echo "  ✓ SQLite graph database"
echo ""
echo "Next steps:"
echo "  1. Add some documents via UI or API"
echo "  2. Navigate to /inputs to see the graph"
echo ""
echo "To stop backend: pkill -f 'python.*main.py'"
echo "To view logs: tail -f backend/logs/* (if logging enabled)"
echo ""
