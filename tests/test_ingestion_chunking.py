from rag.ingestion import DocumentIngestor
from rag.chunker import DocumentChunker


print("\n")
print("=" * 70)
print("              INGESTION + CHUNKING TEST")
print("=" * 70)


# ---------------------------------------------------------
# Initialize components
# ---------------------------------------------------------

ingestor = DocumentIngestor(
    documents_dir="documents"
)

chunker = DocumentChunker(
    chunk_size=3,
    overlap=1
)


# ---------------------------------------------------------
# Load documents
# ---------------------------------------------------------

documents = ingestor.load_documents()


print(
    f"\nDocuments loaded: "
    f"{len(documents)}"
)


# ---------------------------------------------------------
# Chunk every document
# ---------------------------------------------------------

all_chunks = []

for document in documents:

    chunks = chunker.chunk_document(
        document["text"]
    )

    print(
        f"\nDocument: "
        f"{document['filename']}"
    )

    print(
        f"Chunks generated: "
        f"{len(chunks)}"
    )

    for chunk_index, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk_record = {
            "chunk_id": (
                f"{document['id']}_"
                f"{chunk_index}"
            ),
            "document_id": document["id"],
            "document_name": document[
                "filename"
            ],
            "chunk_index": chunk_index,
            "text": chunk
        }

        all_chunks.append(
            chunk_record
        )

        print(
            f"\nCHUNK "
            f"{chunk_record['chunk_id']}"
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


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                         SUMMARY")
print("=" * 70)

print(
    f"\nDocuments: "
    f"{len(documents)}"
)

print(
    f"Total chunks: "
    f"{len(all_chunks)}"
)

print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()