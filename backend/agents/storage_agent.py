from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from models.schemas import StorageStrategy
from utils.entity_extraction import EntityExtractor
from config import get_settings
import json

class StorageRouter:
    """Determines optimal storage strategy for incoming data"""

    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.entity_extractor = EntityExtractor()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a storage strategy expert. Analyze text and decide storage approach:

            - VECTOR_ONLY: Simple text, no entities or relationships
            - GRAPH_ONLY: Primarily about relationships between entities
            - SQL_ONLY: Structured data with clear metadata
            - FULL_HYBRID: Rich content with entities, relationships, and semantic meaning (RECOMMENDED for most cases)

            Return JSON: {{"strategy": "VECTOR_ONLY|GRAPH_ONLY|SQL_ONLY|FULL_HYBRID", "has_entities": true/false, "reasoning": "brief explanation"}}"""),
            ("user", "Determine storage strategy for:\n\n{text}")
        ])

    def determine_strategy(self, text: str) -> tuple[StorageStrategy, list]:
        """Determine storage strategy and extract entities"""
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"text": text})

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            strategy = StorageStrategy(result.get("strategy", "FULL_HYBRID").lower())

            # Extract entities if strategy involves graph
            entities = []
            if strategy in [StorageStrategy.GRAPH_ONLY, StorageStrategy.FULL_HYBRID]:
                entities = self.entity_extractor.extract(text)

            return strategy, entities

        except Exception as e:
            print(f"Storage routing error: {e}, defaulting to FULL_HYBRID")
            return StorageStrategy.FULL_HYBRID, self.entity_extractor.extract(text)
