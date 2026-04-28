import time
import os
import openai
from unittest.mock import MagicMock
import streamlit as st

# Mock st.cache_resource for testing
def mock_cache_resource(func):
    cache = {}
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

@mock_cache_resource
def get_ai_client_optimized(api_key):
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1"
    )

def benchmark_client_init_optimized():
    start = time.perf_counter()
    client = get_ai_client_optimized("fake-key")
    end = time.perf_counter()
    return (end - start) * 1000

def benchmark_context_construction_optimized():
    # Mock search results
    class MockHit:
        def __init__(self, section_title, section_url, chunk_text):
            self.payload = {
                'section_title': section_title,
                'section_url': section_url,
                'chunk_text': chunk_text
            }

    class MockResults:
        def __init__(self, points):
            self.points = points

    search_results = MockResults([
        MockHit(f"Section {i}", f"http://url{i}.com", "Some long chunk text... " * 100)
        for i in range(3)
    ])

    # Optimized implementation
    start = time.perf_counter()
    context_parts = []
    for i, hit in enumerate(search_results.points):
        context_parts.append(f"\n--- SOURCE {i+1}: {hit.payload['section_title']} ---\n")
        context_parts.append(f"URL: {hit.payload.get('section_url', 'N/A')}\n")
        context_parts.append(f"{hit.payload['chunk_text']}\n")

    context_text = "".join(context_parts)
    end = time.perf_counter()
    return (end - start) * 1000

if __name__ == "__main__":
    # Warm up cache
    get_ai_client_optimized("fake-key")

    init_times = [benchmark_client_init_optimized() for _ in range(100)]
    const_times = [benchmark_context_construction_optimized() for _ in range(100)]

    print(f"Optimized Client Init (cached): {sum(init_times)/len(init_times):.4f} ms")
    print(f"Optimized Context Const: {sum(const_times)/len(const_times):.4f} ms")
