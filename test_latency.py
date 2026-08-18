import time

from rag.rag_pipeline import RAGPipeline


QUERY = """
What caused the increase in renewable energy
production between 2020 and 2025?
"""


print("\n")
print("=" * 70)
print("                 TOKENWISE LATENCY TEST")
print("=" * 70)


# ---------------------------------------------------------
# Initialize pipeline
# ---------------------------------------------------------

print("\nInitializing TokenWise...\n")

pipeline = RAGPipeline(
    documents_dir="documents",
    chunk_size=3,
    chunk_overlap=1,
    top_k=4,
    token_budget=80,
    coverage_threshold=0.80
)


# ---------------------------------------------------------
# Build index
# ---------------------------------------------------------

pipeline.build_index()


# ---------------------------------------------------------
# Warm-up
# ---------------------------------------------------------

print("\nRunning warm-up...")

pipeline.run(
    query=QUERY,
    top_k=4,
    token_budget=80
)


# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------

print("\nRunning benchmark...\n")

runs = 5

latencies = []

for run_number in range(
    1,
    runs + 1
):

    start_time = time.perf_counter()

    result = pipeline.run(
        query=QUERY,
        top_k=4,
        token_budget=80
    )

    end_time = time.perf_counter()

    latency_ms = (
        end_time
        -
        start_time
    ) * 1000

    latencies.append(
        latency_ms
    )

    print(
        f"Run {run_number}: "
        f"{latency_ms:.2f} ms"
    )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

average_latency = (
    sum(latencies)
    /
    len(latencies)
)

minimum_latency = min(
    latencies
)

maximum_latency = max(
    latencies
)


print("\n")
print("=" * 70)
print("                     LATENCY RESULTS")
print("=" * 70)

print(
    f"\nAverage latency: "
    f"{average_latency:.2f} ms"
)

print(
    f"Minimum latency: "
    f"{minimum_latency:.2f} ms"
)

print(
    f"Maximum latency: "
    f"{maximum_latency:.2f} ms"
)

print(
    f"\nOriginal tokens: "
    f"{result['original_tokens']}"
)

print(
    f"Compressed tokens: "
    f"{result['compressed_tokens']}"
)

print(
    f"Tokens saved: "
    f"{result['tokens_saved']}"
)

print(
    f"Compression ratio: "
    f"{result['compression_ratio']:.2f}%"
)

print(
    f"Query coverage: "
    f"{result['coverage']:.2%}"
)

print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()
