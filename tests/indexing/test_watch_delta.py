"""Watch-mode reindex reads only the paths that changed."""

from __future__ import annotations

from pathlib import Path

import duckdb

from smritikosh.adapters.file_source.local import LocalFileSource
from smritikosh.adapters.storage.duckdb import DuckDBAdapter
from smritikosh.adapters.vector_store.duckdb import DuckDBVectorStore
from smritikosh.indexing.pipeline import build_index
from smritikosh.ports.embedder import Embedder


class _Embedder(Embedder):
    dims = 4
    model_id = "stub:4"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _stores(tmp_path: Path) -> tuple[DuckDBAdapter, DuckDBVectorStore]:
    con = duckdb.connect(str(tmp_path / "index.duckdb"))
    return DuckDBAdapter(con=con), DuckDBVectorStore(con=con)


def test_one_changed_file_does_not_reread_the_rest(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 1\n")
    storage, vector_store = _stores(tmp_path)
    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vector_store,
    )

    reads: list[str] = []
    original = LocalFileSource.read_text

    def _spy(self: LocalFileSource, path: str) -> str:
        reads.append(path)
        return original(self, path)

    monkeypatch.setattr(LocalFileSource, "read_text", _spy)
    (tmp_path / "b.py").write_text("b = 2\n")
    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vector_store,
        only_paths=["b.py"],
    )

    assert reads == ["b.py"]
    assert "a.py" in storage.get_all_file_paths()
    assert "b.py" in storage.get_all_file_paths()


def test_a_partial_reindex_drops_only_the_removed_path(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 1\n")
    storage, vector_store = _stores(tmp_path)
    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vector_store,
    )

    (tmp_path / "a.py").unlink()
    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vector_store,
        only_paths=[],
        removed_paths=["a.py"],
    )

    assert set(storage.get_all_file_paths()) == {"b.py"}
