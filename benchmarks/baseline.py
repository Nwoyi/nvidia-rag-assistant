import time
import os
from openai import OpenAI
from unittest.mock import MagicMock

def benchmark_client_init():
    start = time.perf_counter()
    # Simulate current streamlit_app.py behavior (initializing on every run)
    client = OpenAI(
        api_key="fake-key",
        base_url="https://api.cerebras.ai/v1"
    )
    end = time.perf_counter()
    return (end - start) * 1000

def benchmark_context_construction():
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

    # Current implementation in streamlit_app.py
    start = time.perf_counter()
    context_text = ""
    for i, hit in enumerate(search_results.points):
        context_text += f"\n--- SOURCE {i+1}: {hit.payload['section_title']} ---\n"
        context_text += f"URL: {hit.payload.get('section_url', 'N/A')}\n"
        context_text += f"{hit.payload['chunk_text']}\n"
    end = time.perf_counter()
    return (end - start) * 1000

if __name__ == "__main__":
    init_times = [benchmark_client_init() for _ in range(100)]
    const_times = [benchmark_context_construction() for _ in range(100)]

    print(f"Baseline Client Init: {sum(init_times)/len(init_times):.4f} ms")
    print(f"Baseline Context Const: {sum(const_times)/len(const_times):.4f} ms")
