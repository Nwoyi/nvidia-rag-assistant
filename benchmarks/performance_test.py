import time
import random
import string
import hashlib

def get_random_string(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def run_string_concat_benchmark(num_sources=1000):
    random_strings = [get_random_string(1000) for _ in range(num_sources)]

    start = time.perf_counter()
    context_text = ""
    for i in range(num_sources):
        context_text += f"\n--- SOURCE {i+1}: Title ---\n"
        context_text += f"URL: http://example.com\n"
        context_text += random_strings[i] + "\n"
    old_duration = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    context_parts = []
    for i in range(num_sources):
        context_parts.append(f"\n--- SOURCE {i+1}: Title ---\n")
        context_parts.append(f"URL: http://example.com\n")
        context_parts.append(random_strings[i] + "\n")
    "".join(context_parts)
    new_duration = (time.perf_counter() - start) * 1000

    print(f"--- String Construction (N={num_sources}) ---")
    print(f"Old (Concatenation): {old_duration:.2f} ms")
    print(f"New (List Join): {new_duration:.2f} ms")

def run_hashing_benchmark():
    class MockQdrantObject:
        def __init__(self, size):
            self.data = list(range(size))

    size = 1000000
    obj = MockQdrantObject(size)
    start = time.perf_counter()
    hashlib.sha256(str(obj).encode()).hexdigest()
    duration = (time.perf_counter() - start) * 1000

    print(f"\n--- Complex Object Hashing ---")
    print(f"Hashing Duration: {duration:.2f} ms")
    print(f"By renaming parameters to _param, Streamlit skips this calculation, improving cache lookup speed.")

if __name__ == "__main__":
    run_string_concat_benchmark()
    run_hashing_benchmark()
