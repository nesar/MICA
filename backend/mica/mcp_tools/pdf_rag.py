"""
MCP Tool: PDF RAG (Retrieval-Augmented Generation)
Description: Parses PDF documents and performs semantic search for relevant information
Inputs: pdf_path (str) or query (str) with collection_name
Outputs: Extracted text or relevant passages with sources

AGENT_INSTRUCTIONS:
You are a document analysis agent specialized in extracting and retrieving information
from PDF documents. Your task is to:

1. Parse PDF documents and extract text content with structure preservation
2. Index document content for efficient semantic search
3. Retrieve relevant passages based on user queries
4. Maintain document provenance (source, page numbers, sections)
5. Handle technical documents, reports, and assessments

When processing documents:
- Extract tables and figures metadata when possible
- Preserve section headings and document structure
- Handle multi-column layouts appropriately
- Note any parsing limitations or issues

When retrieving information:
- Return the most relevant passages with context
- Include page numbers and section references
- Rank results by relevance
- Provide source attribution for all retrieved content

Focus on accuracy and completeness when dealing with technical reports,
regulatory documents, and scientific publications.
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import config
from ..logging import SessionLogger
from .base import MCPTool, ToolResult, register_tool

logger = logging.getLogger(__name__)

# Agent instructions exposed at module level
AGENT_INSTRUCTIONS = __doc__.split("AGENT_INSTRUCTIONS:")[-1].strip()


@register_tool
class PDFRAGTool(MCPTool):
    """
    PDF Retrieval-Augmented Generation tool.

    Parses PDF documents and stores them in a vector database for
    semantic search and retrieval.
    """

    name = "pdf_rag"
    description = "Parse PDFs and perform semantic search for relevant information"
    version = "1.0.0"
    AGENT_INSTRUCTIONS = AGENT_INSTRUCTIONS

    def __init__(
        self,
        session_logger: Optional[SessionLogger] = None,
        chroma_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the PDF RAG tool.

        Args:
            session_logger: Optional session logger
            chroma_dir: Directory for ChromaDB persistence
            embedding_model: Sentence transformer model for embeddings
        """
        super().__init__(session_logger)
        self.chroma_dir = chroma_dir or config.rag.chroma_dir
        self.embedding_model = embedding_model or config.rag.embedding_model
        self._client = None
        self._embedding_function = None

    def _init_chroma(self):
        """Initialize ChromaDB client and embedding function."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            # Create persistent client
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))

            # Create embedding function
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )

            logger.info(f"ChromaDB initialized at {self.chroma_dir}")

        except ImportError as e:
            raise ImportError(
                "ChromaDB and sentence-transformers are required for PDF RAG. "
                "Install with: pip install chromadb sentence-transformers"
            ) from e

    def _parse_pdf(self, pdf_path: Path) -> list[dict]:
        """
        Parse a PDF file and extract text with metadata.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of document chunks with metadata
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")

        chunks = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""

                if text.strip():
                    # Split into paragraphs
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

                    for para_idx, paragraph in enumerate(paragraphs):
                        if len(paragraph) > 50:  # Skip very short fragments
                            chunks.append({
                                "text": paragraph,
                                "metadata": {
                                    "source": str(pdf_path),
                                    "filename": pdf_path.name,
                                    "page": page_num,
                                    "paragraph": para_idx,
                                    "total_pages": len(pdf.pages),
                                },
                            })

        return chunks

    def _chunk_text(
        self,
        chunks: list[dict],
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> list[dict]:
        """
        Re-chunk text into consistent sizes with overlap.

        Args:
            chunks: List of parsed chunks
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of re-chunked documents
        """
        result = []

        for chunk in chunks:
            text = chunk["text"]
            metadata = chunk["metadata"].copy()

            if len(text) <= chunk_size:
                result.append(chunk)
            else:
                # Split into smaller chunks
                start = 0
                chunk_idx = 0

                while start < len(text):
                    end = start + chunk_size

                    # Try to break at sentence boundary
                    if end < len(text):
                        for sep in [". ", ".\n", "\n\n", "\n", " "]:
                            last_sep = text[start:end].rfind(sep)
                            if last_sep > chunk_size // 2:
                                end = start + last_sep + 1
                                break

                    chunk_text = text[start:end].strip()

                    if chunk_text:
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_idx"] = chunk_idx
                        result.append({
                            "text": chunk_text,
                            "metadata": chunk_metadata,
                        })

                    start = end - overlap
                    chunk_idx += 1

        return result

    def execute(self, input_data: dict) -> ToolResult:
        """
        Execute PDF RAG operations.

        Args:
            input_data: Dictionary with one of:
                - pdf_path (str): Path to PDF to ingest
                - query (str): Query to search for
                - collection (str, optional): Collection name
                - num_results (int, optional): Number of results to return

        Returns:
            ToolResult with extracted text or search results
        """
        start_time = datetime.now()

        try:
            self._init_chroma()

            pdf_path = input_data.get("pdf_path")
            query = input_data.get("query")
            collection_name = input_data.get("collection", "default")
            num_results = input_data.get("num_results", 5)

            if pdf_path:
                # Ingest PDF
                result = self._ingest_pdf(Path(pdf_path), collection_name)
            elif query:
                # Search
                result = self._search(query, collection_name, num_results)
            else:
                return ToolResult.error("Either pdf_path or query is required")

            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            return result

        except Exception as e:
            logger.error(f"PDF RAG error: {e}")
            return ToolResult.error(str(e))

    def _ingest_pdf(self, pdf_path: Path, collection_name: str) -> ToolResult:
        """Ingest a PDF into the vector store."""
        if not pdf_path.exists():
            return ToolResult.error(f"PDF not found: {pdf_path}")

        # Parse PDF
        logger.info(f"Parsing PDF: {pdf_path}")
        chunks = self._parse_pdf(pdf_path)

        if not chunks:
            return ToolResult.error(f"No text extracted from {pdf_path}")

        # Re-chunk for better retrieval
        chunks = self._chunk_text(chunks)

        # Get or create collection
        collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
        )

        # Generate unique IDs
        doc_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]
        ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]

        # Add to collection
        collection.add(
            ids=ids,
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

        logger.info(f"Ingested {len(chunks)} chunks from {pdf_path}")

        # Log if session available
        if self.session_logger:
            self.session_logger.save_artifact(
                f"{doc_hash}_ingest.json",
                {
                    "pdf_path": str(pdf_path),
                    "collection": collection_name,
                    "num_chunks": len(chunks),
                },
                "data",
            )

        return ToolResult.success(
            data={
                "pdf_path": str(pdf_path),
                "collection": collection_name,
                "num_chunks": len(chunks),
            },
            message=f"Ingested {len(chunks)} chunks from {pdf_path.name}",
        )

    def _search(self, query: str, collection_name: str, num_results: int) -> ToolResult:
        """Search the vector store."""
        try:
            collection = self._client.get_collection(
                name=collection_name,
                embedding_function=self._embedding_function,
            )
        except Exception:
            return ToolResult.error(f"Collection not found: {collection_name}")

        # Query collection
        results = collection.query(
            query_texts=[query],
            n_results=num_results,
        )

        # Format results
        formatted_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            formatted_results.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

        return ToolResult.success(
            data=formatted_results,
            message=f"Found {len(formatted_results)} relevant passages",
            query=query,
            collection=collection_name,
        )

    def list_collections(self) -> list[str]:
        """List all available collections."""
        self._init_chroma()
        collections = self._client.list_collections()
        return [c.name for c in collections]

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        self._init_chroma()
        try:
            self._client.delete_collection(collection_name)
            return True
        except Exception:
            return False


@register_tool
class PDFParserTool(MCPTool):
    """
    Simple PDF parser tool (no vector store).

    Extracts text from PDF files without indexing.
    """

    name = "pdf_parser"
    description = "Extract text from PDF files"
    version = "1.0.0"

    AGENT_INSTRUCTIONS = """
    You are a PDF text extraction agent. Extract and return the text content
    from PDF documents, preserving structure where possible.
    """

    def execute(self, input_data: dict) -> ToolResult:
        """Extract text from a PDF."""
        pdf_path = input_data.get("pdf_path")
        if not pdf_path:
            return ToolResult.error("pdf_path is required")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return ToolResult.error(f"PDF not found: {pdf_path}")

        try:
            import pdfplumber

            text_content = []

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_content.append({
                            "page": page_num,
                            "text": page_text,
                        })

            return ToolResult.success(
                data={
                    "filename": pdf_path.name,
                    "pages": text_content,
                    "total_pages": len(text_content),
                },
                message=f"Extracted text from {len(text_content)} pages",
            )

        except Exception as e:
            return ToolResult.error(f"Failed to parse PDF: {e}")
