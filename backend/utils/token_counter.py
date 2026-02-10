"""Token counting utilities for tracking LLM API usage."""

import tiktoken
from typing import List, Dict, Any


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text: The input text to count tokens for
        model: The model name to use for tokenization (default: gpt-4)
               Supports: gpt-4, gpt-3.5-turbo, text-embedding-3-small, etc.

    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding (used by gpt-4, gpt-3.5-turbo)
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def count_tokens_for_chunks(chunks: List[str], model: str = "gpt-4") -> Dict[str, Any]:
    """
    Count tokens for a list of text chunks.

    Args:
        chunks: List of text chunks
        model: The model name to use for tokenization

    Returns:
        Dictionary containing:
            - total_tokens: Total tokens across all chunks
            - chunk_tokens: List of token counts per chunk
            - chunk_count: Number of chunks
            - avg_tokens_per_chunk: Average tokens per chunk
    """
    chunk_tokens = [count_tokens(chunk, model) for chunk in chunks]
    total_tokens = sum(chunk_tokens)
    avg_tokens = total_tokens / len(chunks) if chunks else 0

    return {
        "total_tokens": total_tokens,
        "chunk_tokens": chunk_tokens,
        "chunk_count": len(chunks),
        "avg_tokens_per_chunk": round(avg_tokens, 2)
    }


def count_messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
    """
    Count tokens for a list of chat messages (OpenAI format).

    Accounts for message formatting overhead (role, content, etc.)

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: The model name to use for tokenization

    Returns:
        Total number of tokens including formatting overhead
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Token overhead per message varies by model
    # For gpt-4 and gpt-3.5-turbo: every message follows <|start|>{role/name}\n{content}<|end|>\n
    tokens_per_message = 3  # Formatting overhead
    tokens_per_name = 1  # If name field is present

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>

    return num_tokens


def estimate_cost(token_count: int, model: str = "gpt-4", is_input: bool = True) -> float:
    """
    Estimate cost in USD based on token count and model.

    Args:
        token_count: Number of tokens
        model: Model name
        is_input: True for input tokens, False for output tokens

    Returns:
        Estimated cost in USD
    """
    # Pricing as of 2025 (per 1M tokens)
    pricing = {
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "text-embedding-3-small": {"input": 0.02, "output": 0.02},
        "text-embedding-3-large": {"input": 0.13, "output": 0.13},
    }

    # Default to gpt-4 pricing if model not found
    model_pricing = pricing.get(model, pricing["gpt-4"])
    price_per_million = model_pricing["input"] if is_input else model_pricing["output"]

    return (token_count / 1_000_000) * price_per_million
