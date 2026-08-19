from rag.rag_pipeline import RAGPipeline
from llm.ollama_client import OllamaClient


QUERY = """
What caused the increase in renewable energy
production between 2020 and 2025?
"""


print("\n")
print("=" * 70)
print("             TOKENWISE LLM BENCHMARK")
print("=" * 70)


# =========================================================
# 1. Initialize RAG pipeline
# =========================================================

print("\nInitializing TokenWise...\n")

pipeline = RAGPipeline(
    documents_dir="documents",
    chunk_size=3,
    chunk_overlap=1,
    top_k=4,
    token_budget=80,
    coverage_threshold=0.80
)


# =========================================================
# 2. Build retrieval index
# =========================================================

print("\nBuilding retrieval index...\n")

pipeline.build_index()


# =========================================================
# 3. Run retrieval + compression
# =========================================================

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


# =========================================================
# 4. Context metrics
# =========================================================

print("\n")
print("=" * 70)
print("                    CONTEXT METRICS")
print("=" * 70)

original_tokens = result[
    "original_tokens"
]

compressed_tokens = result[
    "compressed_tokens"
]

context_tokens_saved = result[
    "tokens_saved"
]

compression_ratio = result[
    "compression_ratio"
]

print(
    f"\nOriginal context tokens:   "
    f"{original_tokens}"
)

print(
    f"Compressed context tokens: "
    f"{compressed_tokens}"
)

print(
    f"Context tokens saved:      "
    f"{context_tokens_saved}"
)

print(
    f"Context compression:       "
    f"{compression_ratio:.2f}%"
)

print(
    f"Query coverage:             "
    f"{result['coverage']:.2%}"
)

print(
    f"Coverage guard:             "
    f"{'PASS' if result['coverage_guard_passed'] else 'FAIL'}"
)


# =========================================================
# 5. Initialize Ollama
# =========================================================

llm = OllamaClient(
    model="llama3.2:latest"
)


# =========================================================
# 6. Warm-up
# =========================================================

print("\n")
print("=" * 70)
print("                    MODEL WARM-UP")
print("=" * 70)

print("\nWarming up Llama 3.2...")

warmup = llm.generate(
    query=QUERY,
    context=compressed_context
)

print(
    f"Warm-up latency: "
    f"{warmup['latency_ms']:.2f} ms"
)


# =========================================================
# 7. Original context benchmark
# =========================================================

print("\n")
print("=" * 70)
print("                 ORIGINAL CONTEXT TEST")
print("=" * 70)

print("\nRunning Llama 3.2 with ORIGINAL context...")

original_result = llm.generate(
    query=QUERY,
    context=original_context
)


# =========================================================
# 8. Compressed context benchmark
# =========================================================

print("\n")
print("=" * 70)
print("               COMPRESSED CONTEXT TEST")
print("=" * 70)

print("\nRunning Llama 3.2 with COMPRESSED context...")

compressed_result = llm.generate(
    query=QUERY,
    context=compressed_context
)


# =========================================================
# 9. LLM latency metrics
# =========================================================

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


# =========================================================
# 10. LLM token metrics
# =========================================================

original_prompt_tokens = (
    original_result["prompt_tokens"]
)

compressed_prompt_tokens = (
    compressed_result["prompt_tokens"]
)

original_completion_tokens = (
    original_result["completion_tokens"]
)

compressed_completion_tokens = (
    compressed_result["completion_tokens"]
)

original_total_tokens = (
    original_result["total_tokens"]
)

compressed_total_tokens = (
    compressed_result["total_tokens"]
)

prompt_tokens_saved = (
    original_prompt_tokens
    -
    compressed_prompt_tokens
)

total_tokens_saved = (
    original_total_tokens
    -
    compressed_total_tokens
)

if original_prompt_tokens > 0:

    prompt_token_reduction = (
        prompt_tokens_saved
        /
        original_prompt_tokens
    ) * 100

else:

    prompt_token_reduction = 0.0


if original_total_tokens > 0:

    total_token_reduction = (
        total_tokens_saved
        /
        original_total_tokens
    ) * 100

else:

    total_token_reduction = 0.0


# =========================================================
# 11. Latency comparison
# =========================================================

print("\n")
print("=" * 70)
print("                  LATENCY COMPARISON")
print("=" * 70)

print(
    f"\nOriginal latency:     "
    f"{original_latency:.2f} ms"
)

print(
    f"Compressed latency:   "
    f"{compressed_latency:.2f} ms"
)

print(
    f"Latency saved:        "
    f"{latency_saved:.2f} ms"
)

print(
    f"Latency reduction:    "
    f"{latency_reduction:.2f}%"
)


# =========================================================
# 12. LLM token comparison
# =========================================================

print("\n")
print("=" * 70)
print("                   LLM TOKEN COMPARISON")
print("=" * 70)

print(
    f"\nOriginal prompt tokens:      "
    f"{original_prompt_tokens}"
)

print(
    f"Compressed prompt tokens:   "
    f"{compressed_prompt_tokens}"
)

print(
    f"Prompt tokens saved:         "
    f"{prompt_tokens_saved}"
)

print(
    f"Prompt token reduction:     "
    f"{prompt_token_reduction:.2f}%"
)

print(
    f"\nOriginal completion tokens:  "
    f"{original_completion_tokens}"
)

print(
    f"Compressed completion tokens:"
    f" {compressed_completion_tokens}"
)

print(
    f"\nOriginal total LLM tokens:   "
    f"{original_total_tokens}"
)

print(
    f"Compressed total LLM tokens: "
    f"{compressed_total_tokens}"
)

print(
    f"Total LLM tokens saved:      "
    f"{total_tokens_saved}"
)

print(
    f"Total token reduction:       "
    f"{total_token_reduction:.2f}%"
)


# =========================================================
# 13. Original answer
# =========================================================

print("\n")
print("=" * 70)
print("                  ORIGINAL ANSWER")
print("=" * 70)

print()

print(
    original_result["response"]
)


# =========================================================
# 14. Compressed answer
# =========================================================

print("\n")
print("=" * 70)
print("                COMPRESSED ANSWER")
print("=" * 70)

print()

print(
    compressed_result["response"]
)


# =========================================================
# 15. Final TokenWise summary
# =========================================================

print("\n")
print("=" * 70)
print("                  TOKENWISE SUMMARY")
print("=" * 70)

print(
    f"\nContext compression:       "
    f"{compression_ratio:.2f}%"
)

print(
    f"Context tokens saved:      "
    f"{context_tokens_saved}"
)

print(
    f"LLM prompt tokens saved:   "
    f"{prompt_tokens_saved}"
)

print(
    f"LLM prompt reduction:      "
    f"{prompt_token_reduction:.2f}%"
)

print(
    f"Latency saved:             "
    f"{latency_saved:.2f} ms"
)

print(
    f"Latency reduction:         "
    f"{latency_reduction:.2f}%"
)

print(
    f"Query coverage:            "
    f"{result['coverage']:.2%}"
)

print(
    f"Coverage guard:            "
    f"{'PASS' if result['coverage_guard_passed'] else 'FAIL'}"
)


# =========================================================
# 16. Final status
# =========================================================

print("\n")

if (
    compressed_tokens
    <
    original_tokens
):

    print(
        "PASS: TokenWise reduced context size."
    )

else:

    print(
        "FAIL: Context was not compressed."
    )


if (
    compressed_prompt_tokens
    <
    original_prompt_tokens
):

    print(
        "PASS: LLM prompt tokens reduced."
    )

else:

    print(
        "FAIL: LLM prompt tokens were not reduced."
    )


if latency_reduction > 0:

    print(
        "PASS: LLM latency reduced."
    )

else:

    print(
        "INFO: LLM latency did not decrease "
        "in this single run."
    )


if result["coverage_guard_passed"]:

    print(
        "PASS: Query coverage preserved."
    )

else:

    print(
        "FAIL: Query coverage threshold failed."
    )


print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()