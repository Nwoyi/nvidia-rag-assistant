import time
import hashlib
import random
import string

# Simulating a large, complex Qdrant-like object
class MockPoint:
    def __init__(self, id, text, metadata):
        self.id = id
        self.payload = {"chunk_text": text, "section_title": metadata}

class MockResponse:
    def __init__(self, num_points):
        self.points = [
            MockPoint(i, "".join(random.choices(string.ascii_letters, k=500)), f"Section {i}")
            for i in range(num_points)
        ]

def simulate_streamlit_hashing(obj):
    """
    Streamlit's hashing is complex, but for a simple benchmark,
    we can simulate the overhead of traversing and hashing a large object.
    """
    hasher = hashlib.md5()
    for point in obj.points:
        hasher.update(str(point.id).encode())
        hasher.update(point.payload["chunk_text"].encode())
        hasher.update(point.payload["section_title"].encode())
    return hasher.hexdigest()

def benchmark_hashing():
    print("--- ⚡ Bolt: Benchmarking Hashing Optimization ---")
    mock_obj = MockResponse(100) # Simulating 100 search results

    # Measure with hashing (simulated)
    start = time.perf_counter()
    for _ in range(1000):
        _ = simulate_streamlit_hashing(mock_obj)
    end = time.perf_counter()
    with_hashing = (end - start) * 1000
    print(f"Time with hashing (1000 iterations): {with_hashing:.2f} ms")

    # Measure without hashing (skipping)
    start = time.perf_counter()
    for _ in range(1000):
        _ = id(mock_obj) # Simulating O(1) skip logic
    end = time.perf_counter()
    without_hashing = (end - start) * 1000
    print(f"Time without hashing (1000 iterations): {without_hashing:.2f} ms")

    improvement = ((with_hashing - without_hashing) / with_hashing) * 100
    print(f"⚡ Performance gain from skipping hashing: {improvement:.2f}%")

def benchmark_string_construction():
    print("\n--- ⚡ Bolt: Benchmarking String Construction Optimization ---")
    num_iterations = 10000
    num_points = 50
    mock_points = [
        {"section_title": f"Title {i}", "section_url": "http://test.com", "chunk_text": "Some text content " * 10}
        for i in range(num_points)
    ]

    # Measure concatenation
    start = time.perf_counter()
    for _ in range(num_iterations):
        context_text = ""
        for i, hit in enumerate(mock_points):
            context_text += f"\n--- SOURCE {i+1}: {hit['section_title']} ---\n"
            context_text += f"URL: {hit['section_url']}\n"
            context_text += f"{hit['chunk_text']}\n"
    end = time.perf_counter()
    concat_time = (end - start) * 1000
    print(f"Time with concatenation ({num_iterations} iterations): {concat_time:.2f} ms")

    # Measure join
    start = time.perf_counter()
    for _ in range(num_iterations):
        context_parts = []
        for i, hit in enumerate(mock_points):
            context_parts.append(f"\n--- SOURCE {i+1}: {hit['section_title']} ---\n")
            context_parts.append(f"URL: {hit['section_url']}\n")
            context_parts.append(f"{hit['chunk_text']}\n")
        context_text = "".join(context_parts)
    end = time.perf_counter()
    join_time = (end - start) * 1000
    print(f"Time with join ({num_iterations} iterations): {join_time:.2f} ms")

    improvement = ((concat_time - join_time) / concat_time) * 100
    print(f"⚡ Performance gain from using join: {improvement:.2f}%")

if __name__ == "__main__":
    benchmark_hashing()
    benchmark_string_construction()
