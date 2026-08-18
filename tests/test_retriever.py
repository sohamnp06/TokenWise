from rag.retriever import FAISSRetriever


documents = [
    """
    Renewable energy production increased significantly
    between 2020 and 2025. Solar generation increased
    by 42% during this period.
    """,

    """
    Government incentives and lower solar panel costs
    were major contributors to the growth of renewable
    energy production.
    """,

    """
    Wind energy production also increased during the
    same period. Several new wind projects were announced.
    """,

    """
    The global economy experienced several major changes
    during the period, including changes in inflation
    and interest rates.
    """,

    """
    Artificial intelligence systems have become more
    capable due to advances in transformer architectures,
    large language models, and training techniques.
    """,

    """
    Electric vehicle adoption increased as battery costs
    declined and charging infrastructure expanded.
    """,

    """
    Weather conditions remained relatively stable across
    many regions during the study period.
    """,
]


query = """
What caused the increase in renewable energy
production between 2020 and 2025?
"""


print("\n")
print("=" * 70)
print("                    FAISS RETRIEVER TEST")
print("=" * 70)


# ---------------------------------------------------------
# Initialize retriever
# ---------------------------------------------------------

retriever = FAISSRetriever()


# ---------------------------------------------------------
# Add documents
# ---------------------------------------------------------

print("\nAdding documents to FAISS...\n")

retriever.add_documents(
    documents
)

print(
    f"Documents indexed: "
    f"{retriever.document_count}"
)


# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

print("\nRunning retrieval...\n")

results = retriever.search(
    query=query,
    top_k=4
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("=" * 70)
print("                     RETRIEVAL RESULTS")
print("=" * 70)


for result in results:

    print(
        f"\nRank:  {result['rank']}"
    )

    print(
        f"Score: "
        f"{result['score']:.4f}"
    )

    print(
        f"Text:\n"
        f"{result['text'].strip()}"
    )

    print(
        "-" * 70
    )


print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()