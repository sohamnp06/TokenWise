import statistics
import time

from rag.rag_pipeline import RAGPipeline
from llm.ollama_client import OllamaClient


QUERY = """
What caused the increase in renewable energy
production between 2020 and 2025?
"""

BENCHMARK_RUNS = 5


def average(values):
    if not values:
        return 0.0

    return statistics.mean(values)


def reduction(original, compressed):
    if original <= 0:
        return 0.0

    return (
        (original - compressed)
        / original
    ) * 100


def print_separator(title):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


# =========================================================
# HEADER
# =========================================================

print_separator(
    "             TOKENWISE LLM BENCHMARK"
)

print("\nInitializing TokenWise...")


# =========================================================
# TOKENWISE
# =========================================================

pipeline = RAGPipeline(
    documents_dir="documents",
    chunk_size=3,
    chunk_overlap=1,
    top_k=4,
    token_budget=80,
    coverage_threshold=0.80
)


print("\nBuilding retrieval index...")

pipeline.build_index()


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
# CONTEXT METRICS
# =========================================================

print_separator(
    "                    CONTEXT METRICS"
)

print(
    f"\nOriginal context tokens:   "
    f"{result['original_tokens']}"
)

print(
    f"Compressed context tokens: "
    f"{result['compressed_tokens']}"
)

print(
    f"Context tokens saved:      "
    f"{result['tokens_saved']}"
)

print(
    f"Context compression:       "
    f"{result['compression_ratio']:.2f}%"
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
# OLLAMA
# =========================================================

llm = OllamaClient(
    model="llama3.2:latest"
)


# =========================================================
# WARM-UP
# =========================================================

print_separator(
    "                    MODEL WARM-UP"
)

print(
    "\nWarming up Llama 3.2..."
)

warmup = llm.generate(
    query=QUERY,
    context=compressed_context
)

print(
    f"Warm-up latency: "
    f"{warmup['latency_ms']:.2f} ms"
)


# =========================================================
# BENCHMARK STORAGE
# =========================================================

original_results = []
compressed_results = []


# =========================================================
# ORIGINAL CONTEXT BENCHMARK
# =========================================================

print_separator(
    "              ORIGINAL CONTEXT BENCHMARK"
)

print(
    f"\nRunning {BENCHMARK_RUNS} benchmark runs..."
)

for run_number in range(
    1,
    BENCHMARK_RUNS + 1
):

    print(
        f"Run {run_number}/{BENCHMARK_RUNS}..."
    )

    result_original = llm.generate(
        query=QUERY,
        context=original_context
    )

    original_results.append(
        result_original
    )

    print(
        f"  Latency: "
        f"{result_original['latency_ms']:.2f} ms"
    )

    print(
        f"  Prompt tokens: "
        f"{result_original['prompt_tokens']}"
    )

    print(
        f"  Completion tokens: "
        f"{result_original['completion_tokens']}"
    )

    print(
        f"  Total tokens: "
        f"{result_original['total_tokens']}"
    )


# =========================================================
# COMPRESSED CONTEXT BENCHMARK
# =========================================================

print_separator(
    "            COMPRESSED CONTEXT BENCHMARK"
)

print(
    f"\nRunning {BENCHMARK_RUNS} benchmark runs..."
)

for run_number in range(
    1,
    BENCHMARK_RUNS + 1
):

    print(
        f"Run {run_number}/{BENCHMARK_RUNS}..."
    )

    result_compressed = llm.generate(
        query=QUERY,
        context=compressed_context
    )

    compressed_results.append(
        result_compressed
    )

    print(
        f"  Latency: "
        f"{result_compressed['latency_ms']:.2f} ms"
    )

    print(
        f"  Prompt tokens: "
        f"{result_compressed['prompt_tokens']}"
    )

    print(
        f"  Completion tokens: "
        f"{result_compressed['completion_tokens']}"
    )

    print(
        f"  Total tokens: "
        f"{result_compressed['total_tokens']}"
    )


# =========================================================
# LATENCY ARRAYS
# =========================================================

original_latencies = [
    item["latency_ms"]
    for item in original_results
]

compressed_latencies = [
    item["latency_ms"]
    for item in compressed_results
]


# =========================================================
# INTERNAL OLLAMA TIMINGS
# =========================================================

original_prompt_eval = [
    item.get(
        "prompt_eval_duration_ms",
        0.0
    )
    for item in original_results
]

compressed_prompt_eval = [
    item.get(
        "prompt_eval_duration_ms",
        0.0
    )
    for item in compressed_results
]

original_generation = [
    item.get(
        "eval_duration_ms",
        0.0
    )
    for item in original_results
]

compressed_generation = [
    item.get(
        "eval_duration_ms",
        0.0
    )
    for item in compressed_results
]


# =========================================================
# AVERAGES
# =========================================================

original_average_latency = average(
    original_latencies
)

compressed_average_latency = average(
    compressed_latencies
)

original_average_prompt_eval = average(
    original_prompt_eval
)

compressed_average_prompt_eval = average(
    compressed_prompt_eval
)

original_average_generation = average(
    original_generation
)

compressed_average_generation = average(
    compressed_generation
)


# =========================================================
# LATENCY REDUCTION
# =========================================================

latency_saved = (
    original_average_latency
    -
    compressed_average_latency
)

latency_reduction = reduction(
    original_average_latency,
    compressed_average_latency
)


prompt_eval_saved = (
    original_average_prompt_eval
    -
    compressed_average_prompt_eval
)

prompt_eval_reduction = reduction(
    original_average_prompt_eval,
    compressed_average_prompt_eval
)


# =========================================================
# TOKEN AVERAGES
# =========================================================

original_prompt_tokens = average([
    item["prompt_tokens"]
    for item in original_results
])

compressed_prompt_tokens = average([
    item["prompt_tokens"]
    for item in compressed_results
])

original_completion_tokens = average([
    item["completion_tokens"]
    for item in original_results
])

compressed_completion_tokens = average([
    item["completion_tokens"]
    for item in compressed_results
])

original_total_tokens = average([
    item["total_tokens"]
    for item in original_results
])

compressed_total_tokens = average([
    item["total_tokens"]
    for item in compressed_results
])


prompt_tokens_saved = (
    original_prompt_tokens
    -
    compressed_prompt_tokens
)

prompt_token_reduction = reduction(
    original_prompt_tokens,
    compressed_prompt_tokens
)


total_tokens_saved = (
    original_total_tokens
    -
    compressed_total_tokens
)

total_token_reduction = reduction(
    original_total_tokens,
    compressed_total_tokens
)


# =========================================================
# LATENCY COMPARISON
# =========================================================

print_separator(
    "                  LATENCY COMPARISON"
)

print(
    f"\nOriginal average latency:   "
    f"{original_average_latency:.2f} ms"
)

print(
    f"Compressed average latency:"
    f" {compressed_average_latency:.2f} ms"
)

print(
    f"Latency saved:              "
    f"{latency_saved:.2f} ms"
)

print(
    f"Latency reduction:          "
    f"{latency_reduction:.2f}%"
)


# =========================================================
# LATENCY RANGE
# =========================================================

print(
    f"\nOriginal latency range:     "
    f"{min(original_latencies):.2f} - "
    f"{max(original_latencies):.2f} ms"
)

print(
    f"Compressed latency range:   "
    f"{min(compressed_latencies):.2f} - "
    f"{max(compressed_latencies):.2f} ms"
)


# =========================================================
# OLLAMA INTERNAL TIMING
# =========================================================

print_separator(
    "               OLLAMA INTERNAL TIMING"
)

print(
    f"\nOriginal prompt evaluation:   "
    f"{original_average_prompt_eval:.2f} ms"
)

print(
    f"Compressed prompt evaluation:"
    f" {compressed_average_prompt_eval:.2f} ms"
)

print(
    f"Prompt evaluation saved:      "
    f"{prompt_eval_saved:.2f} ms"
)

print(
    f"Prompt evaluation reduction:  "
    f"{prompt_eval_reduction:.2f}%"
)

print(
    f"\nOriginal generation time:     "
    f"{original_average_generation:.2f} ms"
)

print(
    f"Compressed generation time:   "
    f"{compressed_average_generation:.2f} ms"
)


# =========================================================
# TOKEN COMPARISON
# =========================================================

print_separator(
    "                  LLM TOKEN COMPARISON"
)

print(
    f"\nOriginal average prompt tokens:"
    f" {original_prompt_tokens:.0f}"
)

print(
    f"Compressed average prompt tokens:"
    f" {compressed_prompt_tokens:.0f}"
)

print(
    f"Prompt tokens saved:           "
    f"{prompt_tokens_saved:.0f}"
)

print(
    f"Prompt token reduction:        "
    f"{prompt_token_reduction:.2f}%"
)

print(
    f"\nOriginal average completion tokens:"
    f" {original_completion_tokens:.0f}"
)

print(
    f"Compressed average completion tokens:"
    f" {compressed_completion_tokens:.0f}"
)

print(
    f"\nOriginal average total LLM tokens:"
    f" {original_total_tokens:.0f}"
)

print(
    f"Compressed average total LLM tokens:"
    f" {compressed_total_tokens:.0f}"
)

print(
    f"Total LLM tokens saved:         "
    f"{total_tokens_saved:.0f}"
)

print(
    f"Total token reduction:          "
    f"{total_token_reduction:.2f}%"
)


# =========================================================
# ANSWERS
# =========================================================

print_separator(
    "                  ORIGINAL ANSWER"
)

print()

print(
    original_results[-1]["response"]
)


print_separator(
    "                COMPRESSED ANSWER"
)

print()

print(
    compressed_results[-1]["response"]
)


# =========================================================
# INDIVIDUAL RUNS
# =========================================================

print_separator(
    "                 INDIVIDUAL RUNS"
)

print(
    "\nRun        Original        Compressed"
)

print(
    "-" * 45
)

for index in range(
    BENCHMARK_RUNS
):

    print(
        f"{index + 1:<10}"
        f"{original_latencies[index]:>10.2f} ms   "
        f"{compressed_latencies[index]:>10.2f} ms"
    )


# =========================================================
# TOKENWISE SUMMARY
# =========================================================

print_separator(
    "                     TOKENWISE SUMMARY"
)

print(
    f"\nContext compression:       "
    f"{result['compression_ratio']:.2f}%"
)

print(
    f"Context tokens saved:      "
    f"{result['tokens_saved']}"
)

print(
    f"Average prompt tokens saved:"
    f" {prompt_tokens_saved:.0f}"
)

print(
    f"Prompt token reduction:    "
    f"{prompt_token_reduction:.2f}%"
)

print(
    f"Average latency saved:     "
    f"{latency_saved:.2f} ms"
)

print(
    f"Average latency reduction: "
    f"{latency_reduction:.2f}%"
)

print(
    f"Prompt evaluation reduction:"
    f" {prompt_eval_reduction:.2f}%"
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
# VALIDATION
# =========================================================

print_separator(
    "                     VALIDATION"
)

if (
    result["compressed_tokens"]
    <
    result["original_tokens"]
):

    print(
        "\nPASS: TokenWise reduced context size."
    )

else:

    print(
        "\nFAIL: Context was not reduced."
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


if (
    result["coverage_guard_passed"]
):

    print(
        "PASS: Query coverage preserved."
    )

else:

    print(
        "FAIL: Query coverage threshold violated."
    )


if latency_reduction > 0:

    print(
        "INFO: Average LLM latency decreased."
    )

else:

    print(
        "INFO: Average LLM latency did not decrease "
        "in this benchmark."
    )


# =========================================================
# END
# =========================================================

print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()