from rag.ingestion import DocumentIngestor


print("\n")
print("=" * 70)
print("                    DOCUMENT INGESTION TEST")
print("=" * 70)


# ---------------------------------------------------------
# Initialize ingestor
# ---------------------------------------------------------

ingestor = DocumentIngestor(
    documents_dir="documents"
)


# ---------------------------------------------------------
# Discover files
# ---------------------------------------------------------

files = ingestor.discover_files()


print(
    f"\nSupported files found: "
    f"{len(files)}"
)

for path in files:

    print(
        f"- {path.name}"
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
# Display documents
# ---------------------------------------------------------

for document in documents:

    print("\n")
    print(
        "=" * 70
    )

    print(
        f"ID:       {document['id']}"
    )

    print(
        f"Filename: {document['filename']}"
    )

    print(
        f"Path:     {document['path']}"
    )

    print(
        f"Characters: "
        f"{len(document['text'])}"
    )

    print(
        "\nPreview:"
    )

    print(
        document["text"][:300]
    )

    print(
        "=" * 70
    )


print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()