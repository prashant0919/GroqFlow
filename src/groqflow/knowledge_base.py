"""Hybrid PDF ingestion and FAISS vector store manager.

Handles tenant-isolated document ingestion with a two-stage extraction pipeline:
1. Primary text extraction via ``pypdf`` (PyPDFLoader)
2. Automatic ``RapidOCR`` fallback for scanned pages with low character density

Manages FAISS indexes with deterministic source tracking for hot-reindexing
(replace documents by filename) and selective deletion.
"""

import logging
import os
from functools import lru_cache
from uuid import uuid4

import faiss
import numpy as np
import pymupdf
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from rapidocr_onnxruntime import RapidOCR

# Minimum character count for a page to skip OCR. Pages below this threshold
# are treated as scanned and processed through RapidOCR.
_OCR_CHAR_THRESHOLD = 40


@lru_cache(maxsize=1)
def _ocr_engine() -> RapidOCR:
    """Lazily initialize and cache the RapidOCR engine singleton."""
    return RapidOCR()


class KnowledgeBase:
    """Manages document ingestion, FAISS indexing, and retrieval for a tenant.

    Each tenant gets an isolated FAISS index directory. Documents are tracked
    by their ``source`` metadata field, enabling deterministic replacement
    when a file is re-uploaded with the same name.

    Args:
        folder_path: Directory containing the tenant's uploaded PDF documents.
        vector_db_path: Directory for persisting the FAISS index files.
    """

    @staticmethod
    def has_index(vector_db_path: str | os.PathLike[str]) -> bool:
        """Check whether a persisted FAISS index exists at the given path."""
        index_path = os.fspath(vector_db_path)
        return os.path.isfile(
            os.path.join(index_path, "index.faiss")
        ) and os.path.isfile(os.path.join(index_path, "index.pkl"))

    def __init__(
        self,
        folder_path: str | None = None,
        vector_db_path: str | None = None,
    ) -> None:
        if vector_db_path is None and folder_path is None:
            raise ValueError(
                "Either `folder_path` or `vector_db_path` must be provided."
            )

        self.folder_path = folder_path
        self.vector_db_path = vector_db_path
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store: FAISS | None = None

        if vector_db_path and self.has_index(vector_db_path):
            self.vector_store = FAISS.load_local(
                vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self._create_vector_store()

    def _save_vector_store(self) -> None:
        """Persist the current FAISS index and docstore to disk."""
        if not self.vector_db_path:
            raise RuntimeError("Cannot save vector store without `vector_db_path`.")
        os.makedirs(self.vector_db_path, exist_ok=True)
        self.vector_store.save_local(self.vector_db_path)

    def _create_vector_store(self) -> None:
        """Build an initial FAISS index from all PDFs in the folder path."""
        documents: list[Document] = []
        if (
            self.folder_path
            and os.path.isdir(self.folder_path)
            and os.listdir(self.folder_path)
        ):
            documents = self._load_documents()

        if documents:
            self.vector_store = FAISS.from_documents(
                documents=documents, embedding=self.embeddings
            )
        else:
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=faiss.IndexFlatL2(len(self.embeddings.embed_query("hello"))),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

        if self.vector_db_path:
            os.makedirs(self.vector_db_path, exist_ok=True)
            self._save_vector_store()

    def _load_document(self, filepath: str) -> list[Document]:
        """Extract text from a PDF using pypdf with RapidOCR fallback.

        For each page, if the extracted text is shorter than ``_OCR_CHAR_THRESHOLD``
        characters, the page is rendered to a pixmap and processed through RapidOCR.
        This avoids unnecessary OCR on digital pages while still handling scanned content.

        Args:
            filepath: Absolute path to the PDF file.

        Returns:
            A list of ``Document`` objects with non-empty page content.
        """
        if filepath.lower().endswith(".pdf"):
            try:
                documents = PyPDFLoader(filepath).load()
                if not documents:
                    return []

                with pymupdf.open(filepath) as pdf:
                    for document in documents:
                        if len(document.page_content.strip()) >= _OCR_CHAR_THRESHOLD:
                            continue

                        page_number = int(document.metadata.get("page", 0))
                        if page_number >= len(pdf):
                            continue
                        page = pdf[page_number]
                        pixmap = page.get_pixmap(
                            matrix=pymupdf.Matrix(2, 2), alpha=False
                        )
                        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                            pixmap.height, pixmap.width, pixmap.n
                        )
                        recognized, _ = _ocr_engine()(image)
                        recognized_text = "\n".join(
                            line[1] for line in (recognized or []) if line[1].strip()
                        )
                        if recognized_text:
                            document.page_content = "\n".join(
                                part
                                for part in (
                                    document.page_content.strip(),
                                    recognized_text,
                                )
                                if part
                            )

                return [
                    document for document in documents if document.page_content.strip()
                ]
            except Exception:
                logging.exception("Failed to load PDF %s", filepath)
        return []

    def _load_documents(self) -> list[Document]:
        """Load all visible PDFs from the tenant's document directory.

        Skips hidden files (prefixed with ``.``) and non-PDF files.
        """
        documents: list[Document] = []
        for filename in os.listdir(self.folder_path):
            if filename.startswith(".") or not filename.lower().endswith(".pdf"):
                continue
            filepath = os.path.join(self.folder_path, filename)
            if os.path.isfile(filepath):
                documents.extend(self._load_document(filepath))
        return documents

    def _get_deleted_chunk_ids(self, filename: str) -> list[str]:
        """Find all docstore IDs for chunks originating from a given source file.

        Args:
            filename: The ``source`` metadata value to match against.

        Returns:
            A list of docstore keys (chunk IDs) for the matching source.
        """
        try:
            return [
                key
                for key, value in self.vector_store.docstore._dict.items()
                if value.metadata.get("source") == filename
            ]
        except AttributeError:
            logging.warning("Unable to access docstore._dict directly.")
            return []

    def _retriever(self, k: int = 4, sources: list[str] | None = None) -> BaseRetriever:
        """Build a filtered MMR retriever over the vector store.

        When ``sources`` is provided, retrieval is restricted to chunks whose
        ``source`` metadata matches one of the given file paths.

        Args:
            k: Number of results to return.
            sources: Optional list of source file paths to restrict retrieval.

        Returns:
            A LangChain ``BaseRetriever`` configured for MMR search.
        """
        search_kwargs: dict = {"k": k}
        if sources is not None:
            source_set = set(sources)
            search_kwargs.update(
                filter=lambda metadata: metadata.get("source") in source_set,
                fetch_k=max(20, self.vector_store.index.ntotal),
            )
        return self.vector_store.as_retriever(
            search_type="mmr", search_kwargs=search_kwargs
        )

    def retrieve(
        self, user_query: str, k: int = 4, sources: list[str] | None = None
    ) -> list[str]:
        """Retrieve relevant document chunks for a user query.

        Args:
            user_query: The natural language query.
            k: Maximum number of chunks to return.
            sources: Optional list of source file paths to restrict retrieval.
                An empty list returns no results (prevents full-index search).

        Returns:
            A list of text chunks ranked by relevance.
        """
        if sources == []:
            return []
        documents = self._retriever(k=k, sources=sources).invoke(user_query)
        return [document.page_content for document in documents]

    def update(self, filename: str, source_path: str | None = None) -> bool:
        """Index or re-index a single document.

        Args:
            filename: Path to the PDF file to process.
            source_path: Optional canonical source path to store in metadata.
                Defaults to ``filename`` if not provided.

        Returns:
            ``True`` if the document was successfully indexed, ``False`` otherwise.
        """
        return self.update_many([(filename, source_path)])[0]

    def update_many(self, files: list[tuple[str, str | None]]) -> list[bool]:
        """Batch index or re-index multiple documents.

        For each file, existing chunks with the same source path are purged
        before new chunks are added. The entire batch is committed in a single
        save operation to minimize disk I/O.

        Args:
            files: List of ``(filepath, source_path)`` tuples. If ``source_path``
                is ``None``, the filepath is used as the source identifier.

        Returns:
            A list of booleans indicating success for each file in order.
        """
        documents: list[Document] = []
        deleted_chunk_ids: set[str] = set()
        results: list[bool] = []

        for filename, source_path in files:
            pages = self._load_document(filename)
            if not pages:
                results.append(False)
                continue

            indexed_source = source_path or filename
            deleted_chunk_ids.update(self._get_deleted_chunk_ids(indexed_source))
            for page in pages:
                page.metadata["source"] = indexed_source
            documents.extend(pages)
            results.append(True)

        if documents:
            if deleted_chunk_ids:
                self.vector_store.delete(list(deleted_chunk_ids))
            ids = [str(uuid4()) for _ in documents]
            self.vector_store.add_documents(documents, ids=ids)
            self._save_vector_store()

        return results

    def delete(self, filename: str) -> None:
        """Remove all indexed chunks for a single source file.

        Args:
            filename: The source path whose chunks should be deleted.
        """
        self.delete_many([filename])

    def delete_many(self, filenames: list[str]) -> None:
        """Batch-remove all indexed chunks for multiple source files.

        Args:
            filenames: List of source paths whose chunks should be deleted.
        """
        chunk_ids = {
            chunk_id
            for filename in filenames
            for chunk_id in self._get_deleted_chunk_ids(filename)
        }
        if chunk_ids:
            self.vector_store.delete(list(chunk_ids))
            self._save_vector_store()
