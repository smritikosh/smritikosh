"""Tests for MpsEmbedder wiring — no torch, no GPU, no model load.

Everything here exercises the parts that must be right *before* the
checkpoint is touched: identity, the query instruction, the chunker's token
budget, the pickle round trip the engine uses for change detection, and the
thread safety two real crashes taught us to guard.
"""

from __future__ import annotations

import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from smritikosh.adapters.embedder.mps import MpsEmbedder
from smritikosh.constants import CODE_QUERY_INSTRUCTION, DEFAULT_MODEL


class _StubSession:
    """Stands in for SentenceTransformer — records what it was asked to encode."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.batch_sizes: list[int] = []

    def encode(self, texts, *, batch_size, normalize_embeddings, show_progress_bar):
        self.calls.append(list(texts))
        self.batch_sizes.append(batch_size)
        return [_StubVector() for _ in texts]


class _StubVector:
    def tolist(self) -> list[float]:
        return [0.5] * 768


class _StubEncoding:
    def __init__(self, n: int) -> None:
        self.ids = list(range(n))


class _StubCounter:
    """Stands in for the raw Rust tokenizer used only for counting."""

    def encode(self, text: str) -> _StubEncoding:
        return _StubEncoding(len(text.split()))


class _OverlapDetectingSession:
    """Fails the test if two encodes are ever in flight at once."""

    def __init__(self) -> None:
        self._active = 0
        self.peak_concurrent = 0
        self._guard = threading.Lock()

    def encode(self, texts, *, batch_size, normalize_embeddings, show_progress_bar):
        with self._guard:
            self._active += 1
            self.peak_concurrent = max(self.peak_concurrent, self._active)
        time.sleep(0.01)
        with self._guard:
            self._active -= 1
        return [_StubVector() for _ in texts]


@pytest.fixture()
def stubbed() -> tuple[MpsEmbedder, _StubSession]:
    embedder = MpsEmbedder()
    session = _StubSession()
    embedder._model = session  # type: ignore[assignment]
    embedder._counter = _StubCounter()  # type: ignore[assignment]
    return embedder, session


# ── Identity and budget, without loading ──────────────────────────────────────


def test_dims_resolve_without_loading_the_model() -> None:
    e = MpsEmbedder()

    assert e.dims == 768
    assert e._model is None


def test_model_id_marks_the_backend() -> None:
    """fp32 on MPS is a different vector space from the INT8 ONNX export."""
    from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder

    mps_id = MpsEmbedder().model_id

    assert "mps" in mps_id
    assert DEFAULT_MODEL in mps_id
    assert mps_id != FastEmbedEmbedder().model_id


def test_max_tokens_defaults_to_the_cpu_window() -> None:
    """A wider window would mean longer chunks and coarser line citations."""
    assert MpsEmbedder().max_tokens == 512


def test_max_seq_length_is_configurable() -> None:
    assert MpsEmbedder(max_seq_length=1024).max_tokens == 1024


def test_rejects_a_nonsense_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        MpsEmbedder(batch_size=0)


# ── Query instruction ─────────────────────────────────────────────────────────


def test_query_instruction_is_shared_with_the_cpu_adapter() -> None:
    """Two copies of this string could drift; both adapters read one constant."""
    from smritikosh.adapters.embedder.fastembed import _CODE_INSTRUCTION

    assert MpsEmbedder()._query_prefix == _CODE_INSTRUCTION == CODE_QUERY_INSTRUCTION


def test_queries_are_prefixed_with_the_instruction(stubbed) -> None:
    embedder, session = stubbed

    embedder.encode_queries(["where do we retry uploads"])

    assert session.calls[0] == [CODE_QUERY_INSTRUCTION + "where do we retry uploads"]


def test_documents_are_not_prefixed(stubbed) -> None:
    """CodeRankEmbed is asymmetric — the instruction belongs to queries only."""
    embedder, session = stubbed

    embedder.encode_documents(["def foo(): pass"])

    assert session.calls[0] == ["def foo(): pass"]


# ── Encoding shape ────────────────────────────────────────────────────────────


def test_encode_documents_returns_plain_float_lists(stubbed) -> None:
    embedder, _ = stubbed

    result = embedder.encode_documents(["a", "b"])

    assert len(result) == 2
    assert all(isinstance(v, list) for v in result)
    assert all(isinstance(x, float) for x in result[0])


def test_configured_batch_size_reaches_the_session(stubbed) -> None:
    embedder, session = stubbed
    embedder._batch_size = 64

    embedder.encode_documents(["a"])

    assert session.batch_sizes == [64]


# ── Pickle support ────────────────────────────────────────────────────────────


def test_pickle_round_trip_drops_the_session(stubbed) -> None:
    """A live torch module is not picklable; _load rebuilds it afterwards."""
    embedder, _ = stubbed

    restored = pickle.loads(pickle.dumps(embedder))

    assert restored._model is None
    assert restored.max_tokens == embedder.max_tokens
    assert restored._query_prefix == CODE_QUERY_INSTRUCTION


def test_pickle_round_trip_restores_a_usable_lock() -> None:
    """A threading.Lock cannot be pickled, so __setstate__ has to remake it."""
    restored = pickle.loads(pickle.dumps(MpsEmbedder()))
    restored._model = _StubSession()  # type: ignore[assignment]

    assert restored.encode_documents(["x"])  # would raise without a lock


def test_query_prefix_is_part_of_the_change_fingerprint() -> None:
    """Editing the instruction must not share a memo cache key."""
    e = MpsEmbedder()
    other = MpsEmbedder()
    other._query_prefix = "a different instruction: "

    assert pickle.dumps(e) != pickle.dumps(other)


def test_window_is_part_of_the_change_fingerprint() -> None:
    """A different window rechunks the corpus, so caches must not carry over."""
    assert pickle.dumps(MpsEmbedder()) != pickle.dumps(MpsEmbedder(max_seq_length=1024))


# ── count_tokens ──────────────────────────────────────────────────────────────


def test_count_tokens_uses_its_own_tokenizer_not_the_models(stubbed) -> None:
    """Regression: sharing the model's tokenizer raised "Already borrowed".

    SentenceTransformer.encode reconfigures truncation on its own tokenizer
    for every batch, which is a Rust mutable borrow. Counting from chunker
    threads at the same time aborted the run.
    """
    embedder, _ = stubbed

    assert embedder.count_tokens("one two three") == 3


def test_count_tokens_is_safe_under_concurrency(stubbed) -> None:
    embedder, _ = stubbed

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(lambda _: embedder.count_tokens("a b"), range(64)))

    assert set(counts) == {2}


# ── Thread safety ─────────────────────────────────────────────────────────────


def test_concurrent_load_builds_exactly_one_model() -> None:
    """Regression: two threads each built a model, and the second initialising
    Metal while the first encoded aborted the process with "A command encoder
    is already encoding to this command buffer".

    count_tokens reaches _load from the chunker, which the pipeline runs on a
    thread pool, so this path is genuinely concurrent.
    """
    builds: list[int] = []
    embedder = MpsEmbedder()

    def slow_build():
        time.sleep(0.05)  # widen the window two threads used to race through
        builds.append(1)
        return _StubSession(), _StubCounter()

    embedder._build = slow_build  # type: ignore[assignment]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: embedder._load(), range(8)))

    assert len(builds) == 1


def test_concurrent_encodes_never_overlap() -> None:
    """One Metal device — overlapping forward passes abort the process."""
    embedder = MpsEmbedder()
    session = _OverlapDetectingSession()
    embedder._model = session  # type: ignore[assignment]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: embedder.encode_documents([str(i)]), range(16)))

    assert session.peak_concurrent == 1
