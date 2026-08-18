from pathlib import Path


class DocumentIngestor:
    """
    Load text documents from a directory.

    Currently supports:
        .txt

    The ingestor is intentionally kept separate from
    chunking and retrieval.

    Pipeline:

        Files
          ↓
        Ingestor
          ↓
        Raw documents
          ↓
        Chunker
          ↓
        Retriever
    """

    SUPPORTED_EXTENSIONS = {
        ".txt"
    }

    def __init__(
        self,
        documents_dir: str = "documents"
    ):
        """
        Initialize the document ingestor.

        Parameters
        ----------
        documents_dir : str
            Directory containing source documents.
        """

        self.documents_dir = Path(
            documents_dir
        )

    def _validate_directory(self) -> None:
        """
        Validate that the document directory exists.
        """

        if not self.documents_dir.exists():

            raise FileNotFoundError(
                f"Document directory not found: "
                f"{self.documents_dir}"
            )

        if not self.documents_dir.is_dir():

            raise NotADirectoryError(
                f"Expected a directory: "
                f"{self.documents_dir}"
            )

    def discover_files(self) -> list[Path]:
        """
        Discover supported document files.

        Returns
        -------
        list[Path]
            Sorted list of supported document paths.
        """

        self._validate_directory()

        files = []

        for path in self.documents_dir.iterdir():

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                in self.SUPPORTED_EXTENSIONS
            ):
                files.append(
                    path
                )

        return sorted(
            files
        )

    def read_file(
        self,
        path: Path
    ) -> str:
        """
        Read a text file using UTF-8.

        Parameters
        ----------
        path : Path
            File to read.

        Returns
        -------
        str
            File contents.
        """

        try:

            return path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError as exc:

            raise ValueError(
                f"Could not decode file as UTF-8: "
                f"{path}"
            ) from exc

    def load_documents(self) -> list[dict]:
        """
        Load all supported documents.

        Returns
        -------
        list[dict]
            Each document contains:

                id
                filename
                path
                text
        """

        files = self.discover_files()

        documents = []

        for index, path in enumerate(
            files
        ):

            text = self.read_file(
                path
            )

            if not text.strip():
                continue

            documents.append({
                "id": index,
                "filename": path.name,
                "path": str(path),
                "text": text
            })

        return documents