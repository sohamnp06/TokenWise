from rag.rag_pipeline import RAGPipeline


query = """
What caused the increase in renewable energy
production between 2020 and 2025?
"""


print("\n")
print("=" * 70)
print("              RETRIEVE -> COMPRESS TEST")
print("=" * 70)


# ---------------------------------------------------------
# Initialize pipeline
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
# Build index
# ---------------------------------------------------------

index_result = pipeline.build_index()


print("\n")
print("=" * 70)
print("                     INDEX SUMMARY")
print("=" * 70)

print(
    f"\nDocuments: "
    f"{index_result['documents']}"
)

print(
    f"Chunks:    "
    f"{index_result['chunks']}"
)

print(
    f"Indexed:   "
    f"{index_result['indexed']}"
)


# ---------------------------------------------------------
# Run complete pipeline
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                   RUNNING PIPELINE")
print("=" * 70)


result = pipeline.run(
    query=query,
    top_k=4,
    token_budget=80
)


# ---------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                    RETRIEVED CHUNKS")
print("=" * 70)


for chunk in result[
    "retrieved_chunks"
]:

    print(
        f"\nRank:       "
        f"{chunk['rank']}"
    )

    print(
        f"Score:      "
        f"{chunk['score']:.4f}"
    )

    print(
        f"Document:   "
        f"{chunk['document_name']}"
    )

    print(
        f"Chunk ID:   "
        f"{chunk['chunk_id']}"
    )

    print(
        f"\n{chunk['text']}"
    )

    print(
        "-" * 70
    )


# ---------------------------------------------------------
# Compression metrics
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                  COMPRESSION METRICS")
print("=" * 70)


print(
    f"\nRetrieved chunks:   "
    f"{result['retrieved_chunk_count']}"
)

print(
    f"Original tokens:    "
    f"{result['original_tokens']}"
)

print(
    f"Compressed tokens:  "
    f"{result['compressed_tokens']}"
)

print(
    f"Tokens saved:       "
    f"{result['tokens_saved']}"
)

print(
    f"Compression ratio:  "
    f"{result['compression_ratio']:.2f}%"
)

print(
    f"Query coverage:     "
    f"{result['coverage']:.2%}"
)

print(
    f"Coverage guard:     "
    f"{'PASS' if result['coverage_guard_passed'] else 'FAIL'}"
)

print(
    f"Guard triggered:    "
    f"{result['coverage_guard_triggered']}"
)


# ---------------------------------------------------------
# Compressed context
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                  COMPRESSED CONTEXT")
print("=" * 70)

print()

print(
    result["compressed_context"]
)


# ---------------------------------------------------------
# Final status
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                       STATUS")
print("=" * 70)

print()

if (
    result["compressed_tokens"]
    <=
    result["token_budget"]
):

    print(
        "PASS: Compressed context is within token budget."
    )

else:

    print(
        "FAIL: Compressed context exceeded token budget."
    )


if result[
    "coverage_guard_passed"
]:

    print(
        "PASS: Query coverage threshold satisfied."
    )

else:

    print(
        "WARNING: Query coverage threshold not satisfied."
    )


if (
    result["compressed_tokens"]
    <
    result["original_tokens"]
):

    print(
        "PASS: Context compression reduced token count."
    )

else:

    print(
        "WARNING: No token reduction achieved."
    )


print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()