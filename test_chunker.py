from rag.chunker import DocumentChunker


document = """
Renewable energy production increased significantly
between 2020 and 2025.

Solar generation increased by 42% during this period.

Government incentives and lower solar panel costs
were major contributors to this growth.

Wind energy production also increased during the
same period.

Several new wind projects were announced.

Solar panel manufacturing capacity also expanded.

The cost of renewable energy technologies declined.

This contributed to wider adoption of renewable
energy across multiple regions.
"""


print("\n")
print("=" * 70)
print("                    DOCUMENT CHUNKER TEST")
print("=" * 70)


chunker = DocumentChunker(
    chunk_size=3,
    overlap=1
)


chunks = chunker.chunk_document(
    document
)


print(
    f"\nTotal chunks: {len(chunks)}\n"
)


for index, chunk in enumerate(
    chunks,
    start=1
):

    print(
        f"CHUNK {index}"
    )

    print(
        "-" * 70
    )

    print(
        chunk
    )

    print(
        "-" * 70
    )


print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()