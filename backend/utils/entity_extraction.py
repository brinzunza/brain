from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import get_settings
from typing import List, Dict
import json

class EntityExtractor:
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting entities from text.
            Extract people, organizations, concepts, topics, and locations.
            Return ONLY a valid JSON array of objects with 'name' and 'type' fields.

            Example output:
            [
                {"name": "Python", "type": "technology"},
                {"name": "John Doe", "type": "person"},
                {"name": "machine learning", "type": "concept"}
            ]"""),
            ("user", "Extract entities from this text:\n\n{text}")
        ])

    def extract(self, text: str) -> List[Dict]:
        """Extract entities from text"""
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"text": text})

            # Parse JSON response
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            entities = json.loads(content)
            return entities
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []
