<div style="margin-bottom: 20px;">
  <img src="mnemosyne.jpg" alt="mnemosyne" width="100%" style="display: block;"/>
</div>

# brain

a minimalistic personal knowledge platform for storing and querying your thoughts, files, and memories.

## features

- **text input**: store any text data (ideas, stories, characteristics, health data)
- **file upload**: upload text files for storage
- **question answering**: ask questions about your stored data using rag (retrieval-augmented generation)
- **dual storage**: vector database (chromadb) + graph database (sqlite) for rich knowledge representation
- **entity extraction**: automatically extracts people, organizations, locations, concepts, and technologies
- **knowledge graph**: visualizes relationships between entities in your stored knowledge
- **modern chat interface**: chatgpt-style interface with fixed bottom input and smooth expansions
- **minimalistic ui**: black/white/green color scheme with jetbrains mono font

## prerequisites

before you begin, ensure you have the following installed:

- **python 3.11+** - [download python](https://www.python.org/downloads/)
  ```bash
  python --version  # should show 3.11 or higher
  ```

- **node.js 18+** - [download node.js](https://nodejs.org/)
  ```bash
  node --version  # should show 18.0.0 or higher
  npm --version   # should show 9.0.0 or higher
  ```

- **openai api key** - [get your key](https://platform.openai.com/api-keys)

## setup

### backend

1. navigate to backend directory:
```bash
cd backend
```

2. install python dependencies:
```bash
pip install -r requirements.txt
```

3. create `.env` file with your openai api key:
```bash
cp .env.example .env
# edit .env and add your OPENAI_API_KEY
```

4. run the backend server:
```bash
python main.py
```

backend will run on `http://localhost:8000`

### frontend

1. navigate to frontend directory:
```bash
cd frontend
```

2. install dependencies:
```bash
npm install
```

3. run the development server:
```bash
npm run dev
```

frontend will run on `http://localhost:5173`

## usage

### storing information

1. **add text**: type anything you want to remember and click "store"
2. **add file**: choose a text file and click "store"
3. **view stored inputs**: navigate to the "inputs" tab to see all your stored data

### asking questions

1. type your question in the chat input at the bottom
2. the system retrieves relevant information from both:
   - vector database (semantic similarity)
   - graph database (entity relationships)
3. get comprehensive answers based on your stored knowledge

### viewing your knowledge graph

1. navigate to the "inputs" tab
2. see the visual knowledge graph showing:
   - entities (people, organizations, locations, concepts, technologies)
   - connections between entities that appear together
   - color-coded by entity type

example:
- store: "i played soccer from 10 years old to 15 years old"
- store: "openai created gpt-4 using machine learning in san francisco"
- ask: "did i play soccer in my life?"
- answer: "yes, you played soccer from when you were 10 years old to 15 years old"
- view graph: see entities like "openai", "gpt-4", "machine learning", "san francisco" with their connections


## license

mit license - see license file for details
