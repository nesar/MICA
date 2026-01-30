"""
MCP Tool: Local Document Search
Description: Searches and retrieves information from local PDF documents in the database
Inputs: query (str), num_results (int), file_filter (str)
Outputs: Relevant passages with sources

AGENT_INSTRUCTIONS:
You are a document retrieval agent specialized in searching the local document database
for relevant information. Your task is to:

1. Search the local PDF database for documents relevant to the user's query
2. Retrieve the most relevant passages with source attribution
3. Index new documents when they are added to the database
4. Provide context about what documents are available

When searching:
- Return passages ranked by relevance to the query
- Include page numbers and document names for citations
- Summarize key findings across multiple documents
- Note when no relevant documents are found

For supply chain analysis, focus on:
- USGS Mineral Commodity Summaries
- DOE Critical Materials reports
- Trade data publications
- Technical assessments and studies
- Policy documents and regulations

Always cite the source document and page number for each piece of information.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import config
from ..logging import SessionLogger
from .base import MCPTool, ToolResult, register_tool

logger = logging.getLogger(__name__)

# Agent instructions exposed at module level
AGENT_INSTRUCTIONS = __doc__.split("AGENT_INSTRUCTIONS:")[-1].strip()


@register_tool
class LocalDocumentSearchTool(MCPTool):
    """
    Local document search tool using RAG.

    Indexes and searches PDF documents from the local database directory.
    Uses ChromaDB for semantic search across all indexed documents.
    """

    name = "local_doc_search"
    description = "Search local PDF documents for relevant information"
    version = "1.0.0"
    AGENT_INSTRUCTIONS = AGENT_INSTRUCTIONS

    # Collection name for local documents
    COLLECTION_NAME = "local_documents"

    def __init__(
        self,
        session_logger: Optional[SessionLogger] = None,
        database_dir: Optional[Path] = None,
    ):
        """
        Initialize the local document search tool.

        Args:
            session_logger: Optional session logger
            database_dir: Override for the database directory
        """
        super().__init__(session_logger)
        self.database_dir = database_dir or config.database.database_dir
        self.pdf_dir = self.database_dir / config.database.pdf_subdir
        self.chroma_dir = config.rag.chroma_dir
        self.embedding_model = config.rag.embedding_model
        self._client = None
        self._embedding_function = None
        self._indexed_files: set = set()

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

            logger.info(f"ChromaDB initialized for local documents at {self.chroma_dir}")

        except ImportError as e:
            raise ImportError(
                "ChromaDB and sentence-transformers are required for local document search. "
                "Install with: pip install chromadb sentence-transformers"
            ) from e

    def _get_collection(self):
        """Get or create the local documents collection."""
        self._init_chroma()
        return self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embedding_function,
            metadata={"description": "Local PDF documents from MICA database"}
        )

    def execute(self, input_data: dict) -> ToolResult:
        """
        Execute local document search operations.

        Args:
            input_data: Dictionary with:
                - query (str): Search query
                - num_results (int, optional): Number of results to return (default 5)
                - file_filter (str, optional): Filter by filename pattern
                - operation (str, optional): 'search', 'index', 'list', 'reindex' (default 'search')

        Returns:
            ToolResult with search results or operation status
        """
        start_time = datetime.now()

        try:
            operation = input_data.get("operation", "search")

            if operation == "search":
                query = input_data.get("query")
                if not query:
                    return ToolResult.error("query is required for search operation")
                result = self._search(
                    query=query,
                    num_results=input_data.get("num_results", 5),
                    file_filter=input_data.get("file_filter"),
                )
            elif operation == "index":
                # Index a specific file or all files
                file_path = input_data.get("file_path")
                result = self._index_documents(file_path)
            elif operation == "reindex":
                # Force reindex all documents
                result = self._reindex_all()
            elif operation == "list":
                # List available documents
                result = self._list_documents()
            else:
                return ToolResult.error(f"Unknown operation: {operation}")

            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            return result

        except Exception as e:
            logger.error(f"Local document search error: {e}")
            return ToolResult.error(str(e))

    def _list_documents(self) -> ToolResult:
        """List all available PDF documents in the database."""
        if not self.pdf_dir.exists():
            return ToolResult.success(
                data={"documents": [], "count": 0},
                message=f"PDF directory not found: {self.pdf_dir}",
            )

        pdf_files = config.database.get_pdf_files()

        documents = []
        for pdf_path in pdf_files:
            documents.append({
                "filename": pdf_path.name,
                "path": str(pdf_path),
                "size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat(),
            })

        # Sort by name
        documents.sort(key=lambda x: x["filename"])

        return ToolResult.success(
            data={
                "documents": documents,
                "count": len(documents),
                "pdf_directory": str(self.pdf_dir),
            },
            message=f"Found {len(documents)} PDF documents in local database",
        )

    def _index_documents(self, file_path: Optional[str] = None) -> ToolResult:
        """Index PDF documents into ChromaDB."""
        collection = self._get_collection()

        if file_path:
            # Index specific file
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                return ToolResult.error(f"File not found: {file_path}")
            pdf_files = [pdf_path]
        else:
            # Index all files in database
            pdf_files = config.database.get_pdf_files()

        if not pdf_files:
            return ToolResult.success(
                data={"indexed": 0},
                message="No PDF files found to index",
            )

        total_chunks = 0
        indexed_files = []
        errors = []

        for pdf_path in pdf_files:
            try:
                chunks = self._parse_and_index_pdf(pdf_path, collection)
                total_chunks += chunks
                indexed_files.append(pdf_path.name)
                logger.info(f"Indexed {chunks} chunks from {pdf_path.name}")
            except Exception as e:
                logger.error(f"Failed to index {pdf_path.name}: {e}")
                errors.append({"file": pdf_path.name, "error": str(e)})

        return ToolResult.success(
            data={
                "indexed_files": indexed_files,
                "total_chunks": total_chunks,
                "errors": errors,
            },
            message=f"Indexed {len(indexed_files)} files with {total_chunks} chunks",
        )

    def _parse_and_index_pdf(self, pdf_path: Path, collection) -> int:
        """Parse a PDF and add chunks to the collection."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")

        chunks = []
        max_pages = config.database.max_pdf_pages

        with pdfplumber.open(pdf_path) as pdf:
            pages_to_process = min(len(pdf.pages), max_pages)

            for page_num, page in enumerate(pdf.pages[:pages_to_process], 1):
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
                                    "indexed_at": datetime.utcnow().isoformat(),
                                },
                            })

        if not chunks:
            return 0

        # Re-chunk for better retrieval
        chunks = self._chunk_text(chunks)

        # Generate unique IDs based on file and content
        doc_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]
        ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]

        # Check for existing documents and remove them first (to allow reindexing)
        try:
            existing = collection.get(where={"filename": pdf_path.name})
            if existing and existing["ids"]:
                collection.delete(ids=existing["ids"])
                logger.debug(f"Removed {len(existing['ids'])} existing chunks for {pdf_path.name}")
        except Exception:
            pass  # Collection might be empty

        # Add to collection
        collection.add(
            ids=ids,
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

        return len(chunks)

    def _chunk_text(
        self,
        chunks: list[dict],
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> list[dict]:
        """Re-chunk text into consistent sizes with overlap."""
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

    def _search(
        self,
        query: str,
        num_results: int = 5,
        file_filter: Optional[str] = None,
    ) -> ToolResult:
        """Search the local document collection."""
        collection = self._get_collection()

        # Check if collection has documents
        try:
            count = collection.count()
            if count == 0:
                # Auto-index if empty
                logger.info("Collection empty, auto-indexing local documents")
                index_result = self._index_documents()
                if index_result.data and index_result.data.get("total_chunks", 0) == 0:
                    return ToolResult.success(
                        data={"results": [], "query": query},
                        message="No documents indexed. Please add PDFs to the database directory.",
                    )
        except Exception:
            pass

        # Build query filter
        where_filter = None
        if file_filter:
            where_filter = {"filename": {"$contains": file_filter}}

        # Query collection
        try:
            results = collection.query(
                query_texts=[query],
                n_results=num_results,
                where=where_filter,
            )
        except Exception as e:
            logger.error(f"Search query failed: {e}")
            return ToolResult.error(f"Search failed: {e}")

        # Format results
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                result_item = {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                }
                if results.get("distances"):
                    result_item["relevance_score"] = 1 - (results["distances"][0][i] / 2)  # Normalize

                formatted_results.append(result_item)

        # Log search if session available
        if self.session_logger:
            self.session_logger.save_artifact(
                f"local_search_{datetime.now().strftime('%H%M%S')}.json",
                {
                    "query": query,
                    "num_results": len(formatted_results),
                    "file_filter": file_filter,
                },
                "data",
            )

        return ToolResult.success(
            data={
                "results": formatted_results,
                "query": query,
                "num_results": len(formatted_results),
            },
            message=f"Found {len(formatted_results)} relevant passages from local documents",
        )

    def _reindex_all(self) -> ToolResult:
        """Force reindex all documents by clearing the collection first."""
        try:
            self._init_chroma()
            # Delete and recreate collection
            try:
                self._client.delete_collection(self.COLLECTION_NAME)
                logger.info(f"Deleted collection {self.COLLECTION_NAME} for reindexing")
            except Exception:
                pass  # Collection might not exist

            return self._index_documents()
        except Exception as e:
            return ToolResult.error(f"Reindex failed: {e}")

    def get_document_count(self) -> int:
        """Get the number of indexed documents."""
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception:
            return 0
