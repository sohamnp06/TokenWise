from rag.ingestion import DocumentIngestor
from rag.chunker import DocumentChunker
from rag.retriever import FAISSRetriever

from compressor.pipeline import TokenDiet


class RAGPipeline:
    """
    End-to-end Retrieve -> Compress pipeline.

    Pipeline:

        Documents
            ↓
        Ingestion
            ↓
        Sentence Chunking
            ↓
        FAISS Index
            ↓
        Query
            ↓
        Top-K Retrieved Chunks
            ↓
        TokenWise
            ↓
        Compressed Context
    """

    def __init__(
        self,
        documents_dir: str = "documents",
        chunk_size: int = 3,
        chunk_overlap: int = 1,
        top_k: int = 4,
        token_budget: int = 80,
        coverage_threshold: float = 0.80
    ):
        """
        Initialize the complete RAG pipeline.

        Parameters
        ----------
        documents_dir : str
            Directory containing source documents.

        chunk_size : int
            Number of sentences per retrieval chunk.

        chunk_overlap : int
            Number of overlapping sentences.

        top_k : int
            Number of chunks retrieved from FAISS.

        token_budget : int
            Maximum tokens allowed after compression.

        coverage_threshold : float
            Minimum acceptable query coverage.
        """

        self.documents_dir = documents_dir

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

        self.top_k = top_k

        self.token_budget = token_budget

        self.coverage_threshold = (
            coverage_threshold
        )

        # -----------------------------------------------------
        # Components
        # -----------------------------------------------------

        self.ingestor = DocumentIngestor(
            documents_dir=documents_dir
        )

        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )

        self.retriever = FAISSRetriever()

        self.token_diet = TokenDiet(
            coverage_threshold=coverage_threshold
        )

        # Track whether documents have been indexed.
        self.is_indexed = False

        # Keep metadata-aware chunk records.
        self.chunks = []

    def build_index(self) -> dict:
        """
        Load documents, chunk them, and build the FAISS index.

        Returns
        -------
        dict
            Indexing summary.
        """

        print("\n")
        print("=" * 70)
        print("                    BUILDING RAG INDEX")
        print("=" * 70)

        # -----------------------------------------------------
        # 1. Load documents
        # -----------------------------------------------------

        documents = (
            self.ingestor.load_documents()
        )

        if not documents:

            raise RuntimeError(
                "No documents were loaded."
            )

        print(
            f"\nDocuments loaded: "
            f"{len(documents)}"
        )

        # -----------------------------------------------------
        # 2. Generate chunks
        # -----------------------------------------------------

        all_chunks = []

        for document in documents:

            chunks = (
                self.chunker.chunk_document(
                    document["text"]
                )
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
                    "document_id": document[
                        "id"
                    ],
                    "document_name": document[
                        "filename"
                    ],
                    "chunk_index": chunk_index,
                    "text": chunk
                }

                all_chunks.append(
                    chunk_record
                )

        if not all_chunks:

            raise RuntimeError(
                "No chunks were generated."
            )

        print(
            f"Chunks generated: "
            f"{len(all_chunks)}"
        )

        # -----------------------------------------------------
        # 3. Build FAISS index
        # -----------------------------------------------------

        self.retriever.clear()

        self.retriever.add_chunks(
            all_chunks
        )

        self.chunks = all_chunks

        self.is_indexed = True

        print(
            f"Chunks indexed: "
            f"{self.retriever.document_count}"
        )

        print(
            "\nRAG index ready."
        )

        return {
            "documents": len(documents),
            "chunks": len(all_chunks),
            "indexed": self.retriever.document_count
        }

    def retrieve(
        self,
        query: str,
        top_k: int | None = None
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query.

        Parameters
        ----------
        query : str
            User query.

        top_k : int | None
            Number of chunks to retrieve.
            Uses configured top_k when omitted.

        Returns
        -------
        list[dict]
            Retrieved metadata-aware chunks.
        """

        if not self.is_indexed:

            raise RuntimeError(
                "RAG index has not been built. "
                "Call build_index() first."
            )

        if top_k is None:

            top_k = self.top_k

        return self.retriever.search(
            query=query,
            top_k=top_k
        )

    def compress_retrieved(
        self,
        query: str,
        retrieved_chunks: list[dict],
        token_budget: int | None = None
    ) -> dict:
        """
        Compress retrieved chunks using TokenWise.

        Parameters
        ----------
        query : str
            User query.

        retrieved_chunks : list[dict]
            Chunks returned by FAISS.

        token_budget : int | None
            Maximum compressed-context token budget.

        Returns
        -------
        dict
            TokenWise compression result plus retrieval data.
        """

        if not retrieved_chunks:

            raise ValueError(
                "No retrieved chunks were provided."
            )

        if token_budget is None:

            token_budget = self.token_budget

        # -----------------------------------------------------
        # Combine retrieved chunks
        # -----------------------------------------------------

        retrieved_context = "\n\n".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )

        # -----------------------------------------------------
        # Run TokenWise
        # -----------------------------------------------------

        compression_result = (
            self.token_diet.compress(
                query=query,
                context=retrieved_context,
                token_budget=token_budget
            )
        )

        # -----------------------------------------------------
        # Attach retrieval information
        # -----------------------------------------------------

        compression_result[
            "retrieved_chunks"
        ] = retrieved_chunks

        compression_result[
            "retrieved_chunk_count"
        ] = len(
            retrieved_chunks
        )

        compression_result[
            "retrieved_context"
        ] = retrieved_context

        compression_result[
            "token_budget"
        ] = token_budget

        return compression_result

    def run(
        self,
        query: str,
        top_k: int | None = None,
        token_budget: int | None = None
    ) -> dict:
        """
        Run the complete Retrieve -> Compress pipeline.

        Parameters
        ----------
        query : str
            User query.

        top_k : int | None
            Number of chunks to retrieve.

        token_budget : int | None
            Maximum compressed-context token budget.

        Returns
        -------
        dict
            Complete RAG + TokenWise result.
        """

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        # -----------------------------------------------------
        # Build index if necessary
        # -----------------------------------------------------

        if not self.is_indexed:

            self.build_index()

        # -----------------------------------------------------
        # Retrieval
        # -----------------------------------------------------

        retrieved_chunks = self.retrieve(
            query=query,
            top_k=top_k
        )

        # -----------------------------------------------------
        # Compression
        # -----------------------------------------------------

        result = self.compress_retrieved(
            query=query,
            retrieved_chunks=retrieved_chunks,
            token_budget=token_budget
        )

        # -----------------------------------------------------
        # Add query
        # -----------------------------------------------------

        result["query"] = query

        return result