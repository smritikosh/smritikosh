"""Tests for BatchGatherer, AsyncWrapper, RetryWithSmallerBatch."""

import asyncio
import time

import pytest

from smritikosh.engine import RetryWithSmallerBatch, sm
from smritikosh.engine._batch import BatchGatherer

# ---------------------------------------------------------------------------
# RetryWithSmallerBatch
# ---------------------------------------------------------------------------


def test_retry_with_smaller_batch_is_exception():
    with pytest.raises(RetryWithSmallerBatch):
        raise RetryWithSmallerBatch()


# ---------------------------------------------------------------------------
# BatchGatherer — basic batching
# ---------------------------------------------------------------------------


async def test_gatherer_fires_full_batch_immediately():
    """When max_size is reached, all items fire in a single call."""
    batch_calls: list[list] = []

    async def batch_fn(items: list) -> list:
        batch_calls.append(list(items))
        return [x.upper() for x in items]

    gatherer = BatchGatherer(batch_fn, max_size=3)
    results = await asyncio.gather(
        gatherer.submit("a"),
        gatherer.submit("b"),
        gatherer.submit("c"),
    )
    assert sorted(results) == ["A", "B", "C"]
    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 3


async def test_gatherer_fires_on_window_timeout():
    """Items below max_size fire after the 1 ms collection window."""
    fired: list[int] = []

    async def batch_fn(items: list) -> list:
        fired.append(len(items))
        return [i * 2 for i in items]

    gatherer = BatchGatherer(batch_fn, max_size=100)
    results = await asyncio.gather(
        gatherer.submit(1),
        gatherer.submit(2),
    )
    assert sorted(results) == [2, 4]
    assert fired == [2]


async def test_gatherer_flushes_an_item_submitted_while_a_batch_is_firing():
    """Nothing else would flush it if no later submit arrives."""
    release = asyncio.Event()
    fired: list[list[str]] = []

    async def batch_fn(items: list[str]) -> list[str]:
        fired.append(list(items))
        if len(fired) == 1:
            await release.wait()
        return [item.upper() for item in items]

    gatherer = BatchGatherer(batch_fn, max_size=100)
    first = asyncio.create_task(gatherer.submit("a"))
    while not fired:
        await asyncio.sleep(0.001)
    second = asyncio.create_task(gatherer.submit("b"))
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.wait_for(asyncio.gather(first, second), 2) == ["A", "B"]
    assert fired == [["a"], ["b"]]


async def test_gatherer_sync_batch_fn():
    """BatchGatherer wraps a sync function via asyncio.to_thread."""

    def sync_batch(items: list) -> list:
        return [x + "!" for x in items]

    gatherer = BatchGatherer(sync_batch, max_size=4)
    results = await asyncio.gather(
        gatherer.submit("x"),
        gatherer.submit("y"),
    )
    assert "x!" in results
    assert "y!" in results


# ---------------------------------------------------------------------------
# BatchGatherer — RetryWithSmallerBatch halving
# ---------------------------------------------------------------------------


async def test_retry_with_smaller_batch_halves_recursively():
    attempts: list[int] = []

    async def batch_fn(items: list) -> list:
        attempts.append(len(items))
        if len(items) > 1:
            raise RetryWithSmallerBatch()
        return [items[0] + "!"]

    gatherer = BatchGatherer(batch_fn, max_size=4)
    results = await asyncio.gather(
        gatherer.submit("a"),
        gatherer.submit("b"),
    )
    assert set(results) == {"a!", "b!"}
    assert 2 in attempts  # tried with 2 first
    assert attempts.count(1) == 2  # then two single-item calls


async def test_retry_batch_of_one_raises_runtime_error():
    async def batch_fn(items: list) -> list:
        raise RetryWithSmallerBatch()

    gatherer = BatchGatherer(batch_fn, max_size=1)
    with pytest.raises(RuntimeError, match="cannot halve further"):
        await gatherer.submit("x")


# ---------------------------------------------------------------------------
# AsyncWrapper — @sm.batched
# ---------------------------------------------------------------------------


async def test_sm_batched_groups_concurrent_calls():
    batch_calls: list[int] = []

    @sm.batched(max_size=4)
    async def embed(texts: list) -> list:
        batch_calls.append(len(texts))
        return [t.upper() for t in texts]

    results = await asyncio.gather(embed("a"), embed("b"), embed("c"), embed("d"))
    assert list(results) == ["A", "B", "C", "D"]
    assert batch_calls == [4]  # single batch call


async def test_sm_batched_returns_correct_result_per_caller():
    @sm.batched(max_size=10)
    async def double(items: list) -> list:
        return [x * 2 for x in items]

    r1, r2, r3 = await asyncio.gather(double(1), double(2), double(3))
    assert r1 == 2
    assert r2 == 4
    assert r3 == 6


# ---------------------------------------------------------------------------
# BatchGatherer — inference semaphore
# ---------------------------------------------------------------------------


async def test_inference_sem_serialises_concurrent_fires():
    """At most one _fire runs at a time regardless of concurrent batches."""
    active = 0
    peak = 0

    async def slow_batch(items: list) -> list:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)  # simulate ONNX inference time
        active -= 1
        return items

    gatherer = BatchGatherer(slow_batch, max_size=2)
    # Submit 6 items — should produce 3 batches of 2
    await asyncio.gather(*[gatherer.submit(i) for i in range(6)])

    # Despite concurrent submits, inference semaphore caps active batches at 1
    assert peak == 1


# ---------------------------------------------------------------------------
# AsyncWrapper — @sm.threaded
# ---------------------------------------------------------------------------


async def test_sm_threaded_runs_sync_fn():
    @sm.threaded
    def cpu_work(n: int) -> int:
        return n * n

    result = await cpu_work(7)
    assert result == 49


async def test_sm_threaded_true_concurrency():
    """Multiple @sm.threaded calls should overlap in time."""

    @sm.threaded
    def slow(n: int) -> int:
        time.sleep(0.02)
        return n

    start = time.monotonic()
    results = await asyncio.gather(slow(1), slow(2), slow(3))
    elapsed = time.monotonic() - start

    assert sorted(results) == [1, 2, 3]
    # If truly concurrent, 3 × 20 ms tasks should finish in well under 50 ms
    assert elapsed < 0.05
