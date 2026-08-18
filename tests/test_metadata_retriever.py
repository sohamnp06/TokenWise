from rag.ingestion import DocumentIngestor
from rag.chunker import DocumentChunker
from rag.retriever import FAISSRetriever


query = """
What caused the increase in renewable energy
production between 2020 and 2025?
"""


print("\n")
print("=" * 70)
print("              METADATA-AWARE RETRIEVER TEST")
print("=" * 70)


# ---------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------

ingestor = DocumentIngestor(
    documents_dir="documents"
)

documents = ingestor.load_documents()

print(
    f"\nDocuments loaded: "
    f"{len(documents)}"
)


# ---------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------

chunker = DocumentChunker(
    chunk_size=3,
    overlap=1
)

all_chunks = []

for document in documents:

    chunks = chunker.chunk_document(
        document["text"]
    )

    for chunk_index, chunk in enumerate(
        chunks,
        start=1
    ):

        all_chunks.append(
            {
                "chunk_id": (
                    f"{document['id']}_"
                    f"{chunk_index}"
                ),
                "document_id": document[
                    "id"
                ],
                "document_name": document[
                    "filename"
                ],
                "chunk_index": chunk_index,
                "text": chunk
            }
        )


print(
    f"Total chunks generated: "
    f"{len(all_chunks)}"
)


# ---------------------------------------------------------
# 3. FAISS indexing
# ---------------------------------------------------------

retriever = FAISSRetriever()

retriever.add_chunks(
    all_chunks
)

print(
    f"Chunks indexed in FAISS: "
    f"{retriever.document_count}"
)


# ---------------------------------------------------------
# 4. Search
# ---------------------------------------------------------

print(
    "\nRunning retrieval..."
)

results = retriever.search(
    query=query,
    top_k=4
)


# ---------------------------------------------------------
# 5. Display metadata-aware results
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                     RETRIEVAL RESULTS")
print("=" * 70)


for result in results:

    print(
        f"\nRank:          "
        f"{result['rank']}"
    )

    print(
        f"Score:         "
        f"{result['score']:.4f}"
    )

    print(
        f"Document:      "
        f"{result['document_name']}"
    )

    print(
        f"Document ID:   "
        f"{result['document_id']}"
    )

    print(
        f"Chunk ID:      "
        f"{result['chunk_id']}"
    )

    print(
        f"Chunk Index:   "
        f"{result['chunk_index']}"
    )

    print(
        "\nText:"
    )

    print(
        result["text"]
    )

    print(
        "-" * 70
    )


print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()