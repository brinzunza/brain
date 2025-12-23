from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import QueryType
from config import get_settings
import json

class QueryClassifier:
    """Classifies queries to determine optimal retrieval strategy"""

    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classification expert. Classify queries into types:

            - SEMANTIC: Questions about meaning, concepts, "what is", "explain", similarity-based
              Examples: "What did I learn about Python?", "Explain machine learning concepts I stored"

            - RELATIONAL: Questions about relationships, connections, "how is X related to Y"
              Examples: "How are concepts A and B related?", "Show connections between topics"

            - STRUCTURED: Questions with filters, dates, counts, specific metadata
              Examples: "Show notes from last week", "How many documents about Python?", "Files tagged 'work'"

            - HYBRID: Complex queries needing multiple strategies
              Examples: "Related concepts from last month", "Count similar documents by topic"

            Return ONLY a valid JSON object: {{"query_type": "SEMANTIC|RELATIONAL|STRUCTURED|HYBRID", "reasoning": "brief explanation"}}"""),
            ("user", "Classify this query:\n\n{query}")
        ])

    def classify(self, query: str) -> QueryType:
        """Classify the query type"""
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"query": query})

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            query_type = result.get("query_type", "SEMANTIC")

            return QueryType(query_type.lower())
        except Exception as e:
            print(f"Classification error: {e}, defaulting to SEMANTIC")
            return QueryType.SEMANTIC
