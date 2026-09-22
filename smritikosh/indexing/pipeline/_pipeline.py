"""Pipeline orchestration — ContextKeys, per-chunk/file processing, build_index."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable

from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder
from smritikosh.adapters.file_source.local import LocalFileSource
from smritikosh.adapters.storage.duckdb import DuckDBAdapter
from smritikosh.adapters.vector_store.duckdb import DuckDBVectorStore
from smritikosh.constants import DEFAULT_DB_PATH
from smritikosh.engine import (
    ContextKey,
    PipelineContext,
    get_memo_store,
    initialize_memo_store,
    sm,
    use_context,
)
from smritikosh.indexing.discovery import iter_source_files
from smritikosh.indexing.pipeline._router import _build_router
from smritikosh.indexing.pipeline._stages import _chunk, _extract, _parse
from smritikosh.indexing.strategies._helpers import budget_chars
from smritikosh.models import Chunk, SourceFile
from smritikosh.ports.embedder import Embedder
from smritikosh.ports.retrieval import LexicalStore
from smritikosh.ports.storage import StorageAdapter
from smritikosh.ports.vector_store import VectorStore

# ── ContextKeys ───────────────────────────────────────────────────────────────
# detect_change=True on EMBEDDER: swapping the model auto-invalidates all memos.

EMBEDDER: ContextKey[Embedder] = ContextKey("embedder", detect_change=True)
STORAGE: ContextKey[StorageAdapter] = ContextKey("storage")
VECTOR_STORE: ContextKey[VectorStore] = ContextKey("vector_store")
LEXICAL_STORE: ContextKey[LexicalStore] = ContextKey("lexical_store")


# ── Per-chunk embedding ───────────────────────────────────────────────────────


#: Chunks per ONNX forward pass, once a gathered batch has been length-sorted.
#: ONNX pads every sequence in a batch to the longest one, so a single large
#: chunk inflates the cost of everything beside it.  Measured on a 365-file
#: repo with CodeRankEmbed: 32 unsorted gives 2.34 chunks/s, 8 length-sorted
#: gives 5.62 — the same vectors in 2.4x less time.
_SUB_BATCH = 8


@sm.batched(max_size=32)
def _embed_one(texts: list[str]) -> list[list[float]]:
    """Batch single-item calls; raise RetryWithSmallerBatch on token errors.

    Sorts by length before splitting into forward passes so that chunks of
    similar size travel together, then restores the caller's order — the
    batcher matches results to futures positionally.
    """
    embedder = use_context(EMBEDDER)

    by_length = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors: list[list[float]] = []
    for start in range(0, len(by_length), _SUB_BATCH):
        group = [texts[i] for i in by_length[start : start + _SUB_BATCH]]
        vectors.extend(embedder.encode_documents(group))

    out: list[list[float]] = [[]] * len(texts)
    for original_index, vector in zip(by_length, vectors, strict=True):
        out[original_index] = vector
    return out


@sm.tracked
async def process_chunk(chunk: Chunk) -> tuple[str, list[float]] | None:
    """Embed one chunk; return None when its location and text are unchanged.

    Returns the vector rather than storing it so the caller can write a whole
    file's worth in one round trip — see :meth:`VectorStore.upsert_many`.
    """
    vector_store = use_context(VECTOR_STORE)
    if vector_store.exists(chunk.id):
        return None
    vector: list[float] = await _embed_one(chunk.text)
    return chunk.id, vector


# ── Per-file processing ───────────────────────────────────────────────────────


@sm.memoized
async def process_file(source: SourceFile) -> None:
    """Parse → extract → chunk → embed one source file incrementally."""
    storage = use_context(STORAGE)
    vector_store = use_context(VECTOR_STORE)
    lexical_store = use_context(LEXICAL_STORE)

    old_ids = storage.get_chunk_ids_for_file(source.path)
    parsed = await _parse(source)
    captures = await _extract(parsed, source.has_tags_scm)
    chunks = await _chunk(parsed, captures, source.strategy)
    new_ids = {c.id for c in chunks}

    for stale_id in old_ids - new_ids:
        storage.delete_chunk_node(stale_id)
        vector_store.delete(stale_id)
        lexical_store.delete(stale_id)

    storage.upsert_file_node(source)
    storage.upsert_chunk_nodes(chunks)
    lexical_store.upsert(chunks)

    embedded = await sm.gather(process_chunk, chunks)
    # One write per file rather than per chunk: the vectors are only committed
    # once the whole file has embedded, which matches the crash semantics
    # below — the file hash lands last, so an interrupted file is redone.
    vector_store.upsert_many([pair for pair in embedded if pair is not None])

    # Write file hash after success — a crash forces full re-process next run.
    storage.set_file_hash(
        source.path,
        hashlib.sha256(source.content.encode()).hexdigest(),
    )


# ── Root orchestrator ─────────────────────────────────────────────────────────


@sm.tracked
async def _run_pipeline(
    repo_path: str,
    *,
    file_concurrency: int | None = None,
    on_file_indexed: Callable[[str], None] | None = None,
) -> None:
    """Discover files, clean up deleted ones, fan out process_file."""
    storage = use_context(STORAGE)
    embedder = use_context(EMBEDDER)
    vector_store = use_context(VECTOR_STORE)
    lexical_store = use_context(LEXICAL_STORE)
    file_source = LocalFileSource(repo_path)
    # Chunks are capped at what this model actually encodes — an over-long
    # chunk would be stored whole but embedded from its prefix only.
    max_chars = budget_chars(embedder.max_tokens)
    files = list(
        iter_source_files(
            file_source,
            _build_router(
                max_chars,
                embedder.max_tokens,
                embedder.count_tokens,
            ),
        )
    )

    stored_paths = set(storage.get_all_file_paths())
    current_paths = {f.path for f in files}
    memo_store = get_memo_store()

    for deleted in stored_paths - current_paths:
        # Read the ids before delete_file drops the nodes that carry them.
        vector_store.delete_many(storage.get_chunk_ids_for_file(deleted))
        lexical_store.delete_path(deleted)
        storage.delete_file(deleted)
        memo_store.delete_component("process_file", deleted)

    def _item_done(source: SourceFile) -> None:
        if on_file_indexed is not None:
            on_file_indexed(source.path)

    await sm.fan_out(
        process_file,
        files,
        concurrency=file_concurrency,
        item_done=_item_done,
    )


# ── Public entry point ────────────────────────────────────────────────────────


def count_source_files(repo_path: str) -> int:
    """Return the number of source files that would be indexed in *repo_path*.

    Uses the same router and filters as :func:`build_index` so the count
    matches exactly what the pipeline will process.  Runs in milliseconds —
    no embedding or DB work is performed.
    """
    return sum(
        1 for _ in iter_source_files(LocalFileSource(repo_path), _build_router())
    )


def build_index(
    repo_path: str,
    embedder: Embedder | None = None,
    storage: StorageAdapter | None = None,
    vector_store: VectorStore | None = None,
    file_concurrency: int | None = None,
    on_file_indexed: Callable[[str], None] | None = None,
    *,
    lexical_store: LexicalStore | None = None,
    full: bool = False,
) -> None:
    """Build or incrementally update the vector index for *repo_path*.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to index.
    embedder:
        Defaults to FastEmbedEmbedder (local ONNX, no PyTorch, no API key).
    storage:
        Defaults to DuckDBAdapter writing to ``smritikosh.duckdb``.
    vector_store:
        Defaults to DuckDBVectorStore sharing the storage connection.
    file_concurrency:
        Max files processed concurrently.  ``None`` auto-selects
        ``min(cpu_count, 4)``.  Lower on memory-constrained machines.
    on_file_indexed:
        Optional callback called once per source file after it has been
        fully indexed (embedded + stored), including cache hits.  Receives
        the file path as a string.  Used by the CLI to drive progress bars.
    lexical_store:
        Defaults to DuckDBBm25Store sharing the storage connection.
    full:
        Clear the source, dense, and lexical indexes and the incremental
        caches before rebuilding, so the run starts from empty rather than
        layering new rows over stale ones.
    """
    embedder = embedder or FastEmbedEmbedder()
    storage = storage or DuckDBAdapter(DEFAULT_DB_PATH)

    # Share DuckDB connection across vector store and memo cache when available.
    # Adapters without .con skip memoization and always re-evaluate the pipeline.
    _con = getattr(storage, "con", None)
    vector_store = vector_store or DuckDBVectorStore(DEFAULT_DB_PATH, con=_con)
    if lexical_store is None:
        from smritikosh.adapters.retrieval.duckdb import DuckDBBm25Store

        lexical_store = DuckDBBm25Store(DEFAULT_DB_PATH, con=_con)

    vector_store.setup(embedder.dims)
    lexical_schema_changed: bool = lexical_store.setup()

    if _con is not None:
        initialize_memo_store(_con)

    if full or lexical_schema_changed:
        # Caches go with the indexes. Dropping the nodes and vectors alone
        # leaves process_file a memo hit for every unchanged file, so the
        # rebuild writes nothing back and the index comes out empty.  A
        # changed lexical schema lands here for the same reason: its postings
        # can only be refilled by running every file through process_file
        # again, and the empty file hashes that follow suspend deletion
        # detection, so the other indexes have to start from empty too.
        storage.clear_caches()
        storage.clear_nodes()
        vector_store.clear()
        lexical_store.clear()

    ctx = PipelineContext()
    ctx.provide(EMBEDDER, embedder)
    ctx.provide(STORAGE, storage)
    ctx.provide(VECTOR_STORE, vector_store)
    ctx.provide(LEXICAL_STORE, lexical_store)

    with ctx:
        asyncio.run(
            _run_pipeline(
                repo_path,
                file_concurrency=file_concurrency,
                on_file_indexed=on_file_indexed,
            )
        )
