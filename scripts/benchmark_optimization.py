import time
from streamlit.runtime.caching.hashing import update_hash, CacheType
from qdrant_client.models import ScoredPoint
from pydantic import BaseModel
from typing import List
import hashlib

class MockQueryResponse(BaseModel):
    points: List[ScoredPoint]

def benchmark_string_concatenation(n=50000):
    print(f"--- Benchmarking String Concatenation (n={n}) ---")

    # Method 1: += concatenation
    start = time.perf_counter()
    context_text = ""
    for i in range(n):
        context_text += f"\n--- SOURCE {i+1}: Section Title ---\n"
        context_text += f"URL: http://example.com/{i}\n"
        context_text += f"This is some chunk text for source {i}. " * 10
    end = time.perf_counter()
    duration_plus = (end - start) * 1000
    print(f"Method +=: {duration_plus:.4f} ms")

    # Method 2: List and join
    start = time.perf_counter()
    context_parts = []
    for i in range(n):
        context_parts.append(f"\n--- SOURCE {i+1}: Section Title ---\n")
        context_parts.append(f"URL: http://example.com/{i}\n")
        context_parts.append(f"This is some chunk text for source {i}. " * 10)
    context_text = "".join(context_parts)
    end = time.perf_counter()
    duration_join = (end - start) * 1000
    print(f"Method join(): {duration_join:.4f} ms")

    if duration_plus > 0:
        speedup = (duration_plus - duration_join) / duration_plus * 100
        print(f"Speedup: {speedup:.2f}%")

def benchmark_hashing_overhead():
    print("\n--- Benchmarking Hashing Overhead (Simulation) ---")

    mock_points = [
        ScoredPoint(
            id=i,
            version=1,
            score=0.9,
            payload={"section_title": "Title", "chunk_text": "Text" * 100},
            vector=None
        ) for i in range(100)
    ]
    mock_results = MockQueryResponse(points=mock_points)

    def streamlit_style_hash(obj):
        hasher = hashlib.md5()
        update_hash(obj, hasher=hasher, cache_type=CacheType.DATA)
        return hasher.hexdigest()

    # Warm up
    streamlit_style_hash(mock_results)

    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        streamlit_style_hash(mock_results)
    end = time.perf_counter()
    duration_hash = (end - start) * 1000 / iterations
    print(f"Avg time to hash MockQueryResponse: {duration_hash:.4f} ms")
    print("By using a leading underscore (e.g., _search_results), we skip this overhead entirely (0 ms).")

if __name__ == "__main__":
    benchmark_string_concatenation()
    benchmark_hashing_overhead()
