from rag.rag_pipeline import RAGPipeline
from llm.ollama_client import OllamaClient


QUERY = """
What caused the increase in renewable energy
production between 2020 and 2025?
"""


print("\n")
print("=" * 70)
print("             ORIGINAL vs COMPRESSED LLM TEST")
print("=" * 70)


# ---------------------------------------------------------
# Initialize TokenWise
# ---------------------------------------------------------

pipeline = RAGPipeline(
    documents_dir="documents",
    chunk_size=3,
    chunk_overlap=1,
    top_k=4,
    token_budget=80,
    coverage_threshold=0.80
)


# ---------------------------------------------------------
# Build retrieval index
# ---------------------------------------------------------

pipeline.build_index()


# ---------------------------------------------------------
# Run TokenWise
# ---------------------------------------------------------

print("\nRunning TokenWise compression...")

result = pipeline.run(
    query=QUERY,
    top_k=4,
    token_budget=80
)


original_context = result[
    "retrieved_context"
]

compressed_context = result[
    "compressed_context"
]


print("\n")
print("=" * 70)
print("                    CONTEXT SIZES")
print("=" * 70)

print(
    f"\nOriginal tokens:   "
    f"{result['original_tokens']}"
)

print(
    f"Compressed tokens: "
    f"{result['compressed_tokens']}"
)

print(
    f"Tokens saved:      "
    f"{result['tokens_saved']}"
)

print(
    f"Compression:       "
    f"{result['compression_ratio']:.2f}%"
)


# ---------------------------------------------------------
# Initialize Ollama
# ---------------------------------------------------------

llm = OllamaClient(
    model="llama3.2:latest"
)


# ---------------------------------------------------------
# Warm-up
# ---------------------------------------------------------

print("\nWarming up Llama 3.2...")

warmup = llm.generate(
    query=QUERY,
    context=compressed_context
)

print(
    f"Warm-up latency: "
    f"{warmup['latency_ms']:.2f} ms"
)


# ---------------------------------------------------------
# Original context
# ---------------------------------------------------------

print("\nTesting ORIGINAL context...")

original_result = llm.generate(
    query=QUERY,
    context=original_context
)


# ---------------------------------------------------------
# Compressed context
# ---------------------------------------------------------

print("Testing COMPRESSED context...")

compressed_result = llm.generate(
    query=QUERY,
    context=compressed_context
)


# ---------------------------------------------------------
# Latency comparison
# ---------------------------------------------------------

original_latency = (
    original_result["latency_ms"]
)

compressed_latency = (
    compressed_result["latency_ms"]
)

latency_saved = (
    original_latency
    -
    compressed_latency
)

if original_latency > 0:

    latency_reduction = (
        latency_saved
        /
        original_latency
    ) * 100

else:

    latency_reduction = 0.0


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                  LATENCY COMPARISON")
print("=" * 70)

print(
    f"\nOriginal context latency:   "
    f"{original_latency:.2f} ms"
)

print(
    f"Compressed context latency:"
    f" {compressed_latency:.2f} ms"
)

print(
    f"Latency difference:         "
    f"{latency_saved:.2f} ms"
)

print(
    f"Latency reduction:          "
    f"{latency_reduction:.2f}%"
)


# ---------------------------------------------------------
# Answers
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                  ORIGINAL ANSWER")
print("=" * 70)

print()

print(
    original_result["response"]
)


print("\n")
print("=" * 70)
print("                COMPRESSED ANSWER")
print("=" * 70)

print()

print(
    compressed_result["response"]
)


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                       SUMMARY")
print("=" * 70)

print(
    f"\nTokens saved:       "
    f"{result['tokens_saved']}"
)

print(
    f"Compression ratio:  "
    f"{result['compression_ratio']:.2f}%"
)

print(
    f"Original latency:   "
    f"{original_latency:.2f} ms"
)

print(
    f"Compressed latency: "
    f"{compressed_latency:.2f} ms"
)

print(
    f"Latency reduction:  "
    f"{latency_reduction:.2f}%"
)

print(
    f"Query coverage:     "
    f"{result['coverage']:.2%}"
)

print(
    f"Coverage guard:     "
    f"{'PASS' if result['coverage_guard_passed'] else 'FAIL'}"
)


print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()