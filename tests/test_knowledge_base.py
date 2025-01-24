from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from groqflow.knowledge_base import KnowledgeBase


def test_update_many_batches_add_and_save(tmp_path: Path) -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.docstore = SimpleNamespace(
                _dict={
                    "old-a": Document(
                        page_content="old page a", metadata={"source": "a.pdf"}
                    ),
                    "old-b": Document(
                        page_content="old page b", metadata={"source": "a.pdf"}
                    ),
                }
            )
            self.deleted: list[str] = []
            self.add_calls: list[tuple[list[Document], list[str]]] = []

        def delete(self, ids: list[str]) -> None:
            self.deleted.extend(ids)
            for document_id in ids:
                self.docstore._dict.pop(document_id)

        def add_documents(self, documents: list[Document], ids: list[str]) -> None:
            self.add_calls.append((documents, ids))

    knowledge_base = object.__new__(KnowledgeBase)
    knowledge_base.vector_store = FakeVectorStore()
    knowledge_base._load_document = lambda filename: [
        Document(page_content=filename, metadata={})
    ]
    save_calls: list[bool] = []
    knowledge_base._save_vector_store = lambda: save_calls.append(True)

    results = knowledge_base.update_many([("a.pdf", "a.pdf"), ("b.pdf", "b.pdf")])

    assert results == [True, True]
    assert len(knowledge_base.vector_store.add_calls) == 1
    added_documents, ids = knowledge_base.vector_store.add_calls[0]
    assert [document.metadata["source"] for document in added_documents] == [
        "a.pdf",
        "b.pdf",
    ]
    assert len(ids) == 2
    assert set(knowledge_base.vector_store.deleted) == {"old-a", "old-b"}
    assert len(save_calls) == 1


def test_load_documents_skips_hidden_temporary_pdfs(tmp_path: Path) -> None:
    (tmp_path / "visible.pdf").touch()
    (tmp_path / ".interrupted.upload.pdf").touch()
    (tmp_path / "notes.txt").touch()

    knowledge_base = object.__new__(KnowledgeBase)
    knowledge_base.folder_path = str(tmp_path)
    loaded_files: list[str] = []
    knowledge_base._load_document = lambda filepath: (
        loaded_files.append(Path(filepath).name) or []
    )

    knowledge_base._load_documents()

    assert loaded_files == ["visible.pdf"]


def test_retrieve_filters_to_selected_sources() -> None:
    class FakeVectorStore:
        index = SimpleNamespace(ntotal=12)

        def as_retriever(self, search_type: str, search_kwargs: dict):
            assert search_type == "mmr"
            assert search_kwargs["k"] == 4
            source_filter = search_kwargs["filter"]
            assert source_filter({"source": "/docs/school.pdf"})
            assert not source_filter({"source": "/docs/hospital.pdf"})
            assert search_kwargs["fetch_k"] == 20
            return SimpleNamespace(invoke=lambda query: [])

    knowledge_base = object.__new__(KnowledgeBase)
    knowledge_base.vector_store = FakeVectorStore()

    assert knowledge_base.retrieve("school policy", sources=["/docs/school.pdf"]) == []


def test_retrieve_with_no_selected_sources_does_not_search() -> None:
    knowledge_base = object.__new__(KnowledgeBase)
    knowledge_base.vector_store = None

    assert knowledge_base.retrieve("question", sources=[]) == []


def test_delete_many_removes_sources_and_saves_once() -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def delete(self, ids: list[str]) -> None:
            self.deleted.extend(ids)

    knowledge_base = object.__new__(KnowledgeBase)
    knowledge_base.vector_store = FakeVectorStore()
    knowledge_base._get_deleted_chunk_ids = lambda filename: {
        "one.pdf": ["id-1", "id-2"],
        "two.pdf": ["id-3"],
    }[filename]
    save_calls: list[bool] = []
    knowledge_base._save_vector_store = lambda: save_calls.append(True)

    knowledge_base.delete_many(["one.pdf", "two.pdf"])

    assert set(knowledge_base.vector_store.deleted) == {"id-1", "id-2", "id-3"}
    assert len(save_calls) == 1


def test_load_document_ocr_fills_scanned_page_text(monkeypatch) -> None:
    page_document = Document(page_content="", metadata={"page": 0})

    class FakePixmap:
        width = 1
        height = 1
        n = 3
        samples = b"\x00\x00\x00"

    class FakePdfPage:
        def get_pixmap(self, matrix, alpha):
            assert alpha is False
            return FakePixmap()

    class FakePdf:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __len__(self):
            return 1

        def __getitem__(self, page_number):
            assert page_number == 0
            return FakePdfPage()

    class FakeOcr:
        def __call__(self, image):
            assert image.shape == (1, 1, 3)
            return [([0, 0, 1, 1], "Scanned certificate text", 0.99)], None

    monkeypatch.setattr(
        "groqflow.knowledge_base.PyPDFLoader",
        lambda filepath: SimpleNamespace(load=lambda: [page_document]),
    )
    monkeypatch.setattr(
        "groqflow.knowledge_base.pymupdf.open", lambda filepath: FakePdf()
    )
    monkeypatch.setattr("groqflow.knowledge_base._ocr_engine", lambda: FakeOcr())

    knowledge_base = object.__new__(KnowledgeBase)
    documents = knowledge_base._load_document("scan.pdf")

    assert len(documents) == 1
    assert documents[0].page_content == "Scanned certificate text"
