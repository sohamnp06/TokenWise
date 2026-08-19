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
        ".txt",
        ".pdf",
        ".docx",
        ".md"
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

    def _read_pdf(self, path: Path) -> str:
        """Read text from a PDF file using pypdf or PyPDF2."""
        text_parts = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path), strict=False)
            for idx, page in enumerate(reader.pages):
                try:
                    extracted = page.extract_text()
                    if extracted and extracted.strip():
                        text_parts.append(extracted.strip())
                except Exception:
                    continue
        except Exception:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(str(path), strict=False)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted and extracted.strip():
                        text_parts.append(extracted.strip())
            except Exception as exc:
                raise ValueError(f"Could not parse PDF file '{path.name}': {exc}") from exc

        full_text = "\n\n".join(text_parts).strip()
        if not full_text:
            raise ValueError(
                f"PDF file '{path.name}' contains no extractable text. "
                "If it is a scanned document, please convert or provide a text-searchable PDF."
            )

        return full_text

    def _read_docx(self, path: Path) -> str:
        """Read text from a DOCX file using python-docx."""
        try:
            import docx
            doc = docx.Document(str(path))
            return "\n".join([p.text for p.text in doc.paragraphs if p.text.strip()])
        except Exception as exc:
            raise ValueError(f"Could not read DOCX file {path}: {exc}") from exc

    def read_file(
        self,
        path: Path
    ) -> str:
        """
        Read a document file (.txt, .pdf, .docx, .md).

        Parameters
        ----------
        path : Path
            File to read.

        Returns
        -------
        str
            File contents.
        """

        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._read_pdf(path)

        if ext == ".docx":
            return self._read_docx(path)

        try:
            return path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            return path.read_text(
                encoding="latin-1"
            )

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