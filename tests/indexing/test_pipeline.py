"""Tests for ContextKeys, process_chunk, process_file, and build_index."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb

from smritikosh.adapters.retrieval.duckdb import DuckDBBm25Store
from smritikosh.adapters.storage.duckdb import DuckDBAdapter
from smritikosh.adapters.vector_store.duckdb import DuckDBVectorStore
from smritikosh.engine import PipelineContext
from smritikosh.indexing.pipeline._pipeline import (
    EMBEDDER,
    LEXICAL_STORE,
    STORAGE,
    VECTOR_STORE,
    _embed_one,
    build_index,
    process_chunk,
    process_file,
)
from smritikosh.models import Chunk, SearchOptions, SourceFile
from smritikosh.ports.embedder import Embedder

# ── Stubs ─────────────────────────────────────────────────────────────────────


class _Embedder(Embedder):
    """Fixed 4-dim embedder — no model loading."""

    dims = 4
    model_id = "stub:4"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _VectorStore:
    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    def setup(self, dims: int) -> None: ...  # noqa: D401
    def exists(self, chunk_id: str) -> bool:
        return chunk_id in self._store

    def upsert(self, chunk_id: str, vector: list[float]) -> None:
        self._store[chunk_id] = vector

    def delete(self, chunk_id: str) -> None:
        self._store.pop(chunk_id, None)

    def clear(self) -> None:
        self._store.clear()


class _Storage:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._files: dict[str, str] = {}

    def get_chunk_ids_for_file(self, path: str) -> set[str]:
        return {c.id for c in self._chunks.values() if c.path == path}

    def upsert_file_node(self, source: SourceFile) -> None: ...
    def upsert_chunk_nodes(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c

    def delete_chunk_node(self, chunk_id: str) -> None:
        self._chunks.pop(chunk_id, None)

    def get_all_file_paths(self) -> set[str]:
        return set(self._files.keys())

    def get_file_hash(self, path: str) -> str | None:
        return self._files.get(path)

    def set_file_hash(self, path: str, hash: str) -> None:  # noqa: A002
        self._files[path] = hash

    def delete_file(self, path: str) -> None:
        self._files.pop(path, None)

    def clear_nodes(self) -> None:
        self._chunks.clear()

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return []


class _LexicalStore:
    def __init__(self) -> None:
        self.chunk_ids: set[str] = set()

    def setup(self) -> bool:
        return False

    def upsert(self, chunks: list[Chunk]) -> None:
        self.chunk_ids.update(chunk.id for chunk in chunks)

    def delete(self, chunk_id: str) -> None:
        self.chunk_ids.discard(chunk_id)

    def delete_path(self, path: str) -> None: ...

    def clear(self) -> None:
        self.chunk_ids.clear()

    def search(self, query: str, *, options: Any) -> list[tuple[str, float]]:
        return []


def _make_chunk(text: str = "def foo(): pass", path: str = "a.py") -> Chunk:
    h = hashlib.sha256(text.encode()).hexdigest()
    return Chunk(
        id=h[:16],
        path=path,
        start_line=1,
        end_line=1,
        text=text,
        chunk_kind="function",
        content_hash=h,
    )


def _make_source(
    content: str = "def foo(): pass\n",
    path: str = "a.py",
) -> SourceFile:
    from smritikosh.indexing.strategies.section import SectionChunkingStrategy

    return SourceFile(
        path=path,
        language="python",
        content=content,
        has_tags_scm=False,
        strategy=SectionChunkingStrategy(),
    )


# ── _embed_one length-sorted sub-batching ─────────────────────────────────────


class _EchoEmbedder(Embedder):
    """Returns a vector encoding each text's length, so order is checkable."""

    dims = 1
    model_id = "echo:1"

    def __init__(self) -> None:
        self.groups: list[list[str]] = []

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.groups.append(list(texts))
        return [[float(len(t))] for t in texts]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self.encode_documents(texts)


async def test_embed_one_returns_vectors_in_caller_order() -> None:
    """Sorting by length must not permute results.

    The batcher matches results to futures positionally, so a reordering bug
    would silently attach every vector to the wrong chunk.
    """
    embedder = _EchoEmbedder()
    # Deliberately unsorted lengths spanning more than one sub-batch.
    texts = ["x" * n for n in (50, 3, 900, 12, 7, 400, 1, 65, 30, 8, 200, 2)]

    with PipelineContext() as ctx:
        ctx.provide(EMBEDDER, embedder)
        vectors = [await _embed_one(t) for t in texts]

    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]


async def test_embed_one_groups_similar_lengths_together() -> None:
    """Tiny and huge chunks must not share a forward pass.

    Padding is to the longest member of a batch, so mixing a 1-char chunk with
    a 900-char one is exactly the waste the sort exists to remove.  Needs more
    than _SUB_BATCH items, otherwise there is only one group to inspect.
    """
    import asyncio

    embedder = _EchoEmbedder()
    tiny = list(range(1, 9))  # 1..8 chars
    huge = list(range(901, 909))  # 901..908 chars
    texts = ["x" * n for n in (tiny + huge)]

    with PipelineContext() as ctx:
        ctx.provide(EMBEDDER, embedder)
        await asyncio.gather(*(_embed_one(t) for t in texts))

    multi = [g for g in embedder.groups if len(g) > 1]
    assert multi, "expected at least one multi-item forward pass"
    for group in multi:
        lengths = [len(t) for t in group]
        assert max(lengths) / min(lengths) < 100, (
            f"group mixes wildly different lengths: {sorted(lengths)}"
        )


# ── ContextKeys ───────────────────────────────────────────────────────────────


def test_embedder_key_has_detect_change() -> None:
    assert EMBEDDER.name == "embedder"
    assert EMBEDDER.detect_change is True


def test_storage_key_has_no_detect_change() -> None:
    assert STORAGE.name == "storage"
    assert STORAGE.detect_change is False


def test_vector_store_key_has_no_detect_change() -> None:
    assert VECTOR_STORE.name == "vector_store"
    assert VECTOR_STORE.detect_change is False


# ── process_chunk ─────────────────────────────────────────────────────────────


async def test_process_chunk_skips_when_vector_already_exists() -> None:
    vs = _VectorStore()
    chunk = _make_chunk()
    vs.upsert(chunk.id, [0.1, 0.2, 0.3, 0.4])  # pre-populate

    ctx = PipelineContext()
    ctx.provide(VECTOR_STORE, vs)
    ctx.provide(EMBEDDER, _Embedder())
    with ctx:
        await process_chunk(chunk)

    assert list(vs._store.values()) == [[0.1, 0.2, 0.3, 0.4]]  # unchanged


async def test_process_chunk_embeds_and_upserts_new_chunk() -> None:
    vs = _VectorStore()
    chunk = _make_chunk()

    ctx = PipelineContext()
    ctx.provide(VECTOR_STORE, vs)
    ctx.provide(EMBEDDER, _Embedder())
    with ctx:
        await process_chunk(chunk)

    assert chunk.id in vs._store
    assert len(vs._store[chunk.id]) == 4


# ── process_file ──────────────────────────────────────────────────────────────


async def test_process_file_writes_file_hash_after_success() -> None:
    storage = _Storage()
    vs = _VectorStore()
    source = _make_source()

    ctx = PipelineContext()
    ctx.provide(STORAGE, storage)
    ctx.provide(VECTOR_STORE, vs)
    ctx.provide(EMBEDDER, _Embedder())
    ctx.provide(LEXICAL_STORE, _LexicalStore())
    with ctx:
        await process_file(source)

    expected_hash = hashlib.sha256(source.content.encode()).hexdigest()
    assert storage.get_file_hash(source.path) == expected_hash


async def test_process_file_removes_stale_chunks() -> None:
    storage = _Storage()
    vs = _VectorStore()
    source = _make_source()

    # Pre-populate a stale chunk that will no longer appear after processing.
    stale = _make_chunk("stale", path=source.path)
    storage._chunks[stale.id] = stale
    vs.upsert(stale.id, [0.0, 0.0, 0.0, 0.0])

    ctx = PipelineContext()
    ctx.provide(STORAGE, storage)
    ctx.provide(VECTOR_STORE, vs)
    ctx.provide(EMBEDDER, _Embedder())
    ctx.provide(LEXICAL_STORE, _LexicalStore())
    with ctx:
        await process_file(source)

    assert stale.id not in storage._chunks
    assert stale.id not in vs._store


# ── build_index ───────────────────────────────────────────────────────────────


def test_build_index_runs_end_to_end(tmp_path: Path) -> None:
    """Smoke test: build_index over a one-file repo without errors."""
    (tmp_path / "hello.py").write_text("def hello(): pass\n")

    con = duckdb.connect(":memory:")
    storage = DuckDBAdapter(con=con)
    vs = DuckDBVectorStore(con=con)

    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vs,
    )

    paths = storage.get_all_file_paths()
    assert "hello.py" in paths


def test_build_index_second_run_is_idempotent(tmp_path: Path) -> None:
    """Second run on unchanged repo: file_hash hit, no new rows written."""
    (tmp_path / "hello.py").write_text("def hello(): pass\n")

    con = duckdb.connect(":memory:")
    storage = DuckDBAdapter(con=con)
    vs = DuckDBVectorStore(con=con)

    build_index(str(tmp_path), embedder=_Embedder(), storage=storage, vector_store=vs)
    chunk_ids_after_first = storage.get_chunk_ids_for_file("hello.py")

    build_index(str(tmp_path), embedder=_Embedder(), storage=storage, vector_store=vs)
    chunk_ids_after_second = storage.get_chunk_ids_for_file("hello.py")

    assert chunk_ids_after_first == chunk_ids_after_second


def _vector_count(con: duckdb.DuckDBPyConnection) -> int:
    return con.execute("SELECT count(*) FROM vectors").fetchone()[0]


def test_build_index_full_repopulates_what_it_cleared(tmp_path: Path) -> None:
    """full=True must rebuild, not just empty.

    Clearing the nodes and vectors while the memo cache still held every file
    left process_file a cache hit with nothing to write back, so the rebuild
    produced an empty index.
    """
    (tmp_path / "hello.py").write_text("def hello(): pass\n")

    con = duckdb.connect(":memory:")
    storage = DuckDBAdapter(con=con)
    vs = DuckDBVectorStore(con=con)

    build_index(str(tmp_path), embedder=_Embedder(), storage=storage, vector_store=vs)
    chunk_ids = storage.get_chunk_ids_for_file("hello.py")
    assert chunk_ids and _vector_count(con) == len(chunk_ids)

    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vs,
        full=True,
    )

    assert storage.get_chunk_ids_for_file("hello.py") == chunk_ids
    assert _vector_count(con) == len(chunk_ids)


def test_build_index_full_drops_rows_the_sources_no_longer_produce(
    tmp_path: Path,
) -> None:
    """A full rebuild starts from empty instead of layering over stale rows."""
    (tmp_path / "hello.py").write_text("def hello(): pass\n")

    con = duckdb.connect(":memory:")
    storage = DuckDBAdapter(con=con)
    vs = DuckDBVectorStore(con=con)

    build_index(str(tmp_path), embedder=_Embedder(), storage=storage, vector_store=vs)

    orphan = _make_chunk("left over from an older run", path="gone.py")
    storage.upsert_chunk_nodes([orphan])
    vs.upsert(orphan.id, [0.1, 0.2, 0.3, 0.4])

    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vs,
        full=True,
    )

    assert storage.get_chunk_ids_for_file("gone.py") == set()
    assert vs.exists(orphan.id) is False


def test_build_index_drops_the_vectors_of_a_deleted_source_file(
    tmp_path: Path,
) -> None:
    """Removing a file must take its vectors with it.

    delete_file dropped the nodes and the hash, and nothing removed the rows
    keyed by the chunk ids it had just deleted, so they were left behind with
    nothing pointing at them.
    """
    (tmp_path / "keep.py").write_text("def keep(): pass\n")
    (tmp_path / "gone.py").write_text("def gone(): pass\n")

    con = duckdb.connect(":memory:")
    storage = DuckDBAdapter(con=con)
    vs = DuckDBVectorStore(con=con)

    build_index(str(tmp_path), embedder=_Embedder(), storage=storage, vector_store=vs)
    gone_ids = storage.get_chunk_ids_for_file("gone.py")
    assert gone_ids and all(vs.exists(chunk_id) for chunk_id in gone_ids)

    (tmp_path / "gone.py").unlink()
    build_index(str(tmp_path), embedder=_Embedder(), storage=storage, vector_store=vs)

    assert storage.get_chunk_ids_for_file("gone.py") == set()
    assert not any(vs.exists(chunk_id) for chunk_id in gone_ids)
    assert storage.get_chunk_ids_for_file("keep.py"), "surviving file was collateral"


def test_build_index_reports_every_file_to_the_callback(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("def one(): pass\n")
    (tmp_path / "two.py").write_text("def two(): pass\n")

    con = duckdb.connect(":memory:")
    seen: list[str] = []

    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=DuckDBAdapter(con=con),
        vector_store=DuckDBVectorStore(con=con),
        on_file_indexed=seen.append,
    )

    assert sorted(seen) == ["one.py", "two.py"]


def test_build_index_refills_the_lexical_index_after_a_schema_change(
    tmp_path: Path,
) -> None:
    """A changed lexical schema must rebuild, not leave an empty index.

    The postings are written by the memoised process_file, so discarding them
    while the memo cache still claims every file is current left an ordinary
    run with nothing to write back and lexical search silently empty.
    """
    (tmp_path / "hello.py").write_text("def hello(): pass\n")

    con = duckdb.connect(":memory:")
    storage = DuckDBAdapter(con=con)
    vs = DuckDBVectorStore(con=con)
    lexical = DuckDBBm25Store(con=con)
    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vs,
        lexical_store=lexical,
    )
    con.execute(
        "UPDATE lexical_metadata SET value = 'older' WHERE key = 'schema_version'"
    )

    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vs,
        lexical_store=lexical,
    )

    assert lexical.search("hello", options=SearchOptions(top_k=5))


def test_build_index_removes_all_indexes_when_source_file_is_deleted(
    tmp_path: Path,
) -> None:
    source_path: Path = tmp_path / "hello.py"
    source_path.write_text("def hello(): pass\n")
    connection = duckdb.connect(":memory:")
    storage = DuckDBAdapter(con=connection)
    vector_store = DuckDBVectorStore(con=connection)
    lexical_store = DuckDBBm25Store(con=connection)
    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vector_store,
        lexical_store=lexical_store,
    )
    (chunk_id,) = storage.get_chunk_ids_for_file("hello.py")

    source_path.unlink()
    build_index(
        str(tmp_path),
        embedder=_Embedder(),
        storage=storage,
        vector_store=vector_store,
        lexical_store=lexical_store,
    )

    assert vector_store.exists(chunk_id) is False
    assert (
        lexical_store.search(
            "hello",
            options=SearchOptions(top_k=5),
        )
        == []
    )
