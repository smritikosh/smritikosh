"""Batching primitives — RetryWithSmallerBatch, BatchGatherer, AsyncWrapper."""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class RetryWithSmallerBatch(Exception):
    """Raise inside an ``@sm.batched`` function to halve the batch and retry.

    :class:`BatchGatherer` catches this, splits the current batch in two, and
    recursively retries each half.  If the batch is already size 1, a plain
    ``RuntimeError`` is propagated to the caller.
    """


# ---------------------------------------------------------------------------
# BatchGatherer
# ---------------------------------------------------------------------------


class BatchGatherer:
    """Collects concurrent single-item calls and fires them as a single batch.

    Fires when either:

    * ``max_size`` items have accumulated, **or**
    * ~1 ms has elapsed since the first item arrived (collection window).

    On :class:`RetryWithSmallerBatch` the batch is split in half and each half
    is retried recursively.

    Parameters
    ----------
    batch_fn:
        Batch callable — ``(items: list[T]) -> list[U]`` (sync or async).
        Must return exactly one result per input item, in the same order.
    max_size:
        Maximum items accumulated before the batch fires immediately.

    Inference serialisation
    -----------------------
    A process-wide semaphore (``_INFERENCE_SEM``) ensures **at most one**
    ``_fire`` call — and therefore at most one ONNX / ML inference session —
    is active at any moment.  This prevents multiple concurrent ONNX threads
    from OOM-killing the process on memory-constrained machines, regardless
    of how many files are being processed in parallel above.
    """

    _WINDOW_SECS: float = 0.001  # 1 ms collection window

    # Process-wide inference semaphore.  Lazily created so it binds to the
    # running event loop on first use.  Class-level so it is shared across
    # all BatchGatherer instances (all embedding calls share one device).
    _inference_sem: asyncio.Semaphore | None = None

    @classmethod
    def _get_inference_sem(cls) -> asyncio.Semaphore:
        if cls._inference_sem is None:
            cls._inference_sem = asyncio.Semaphore(1)
        return cls._inference_sem

    def __init__(self, batch_fn: Callable[..., Any], max_size: int) -> None:
        self._batch_fn = batch_fn
        self._max_size = max_size
        self._pending: list[tuple[Any, asyncio.Future[Any]]] = []
        self._lock: asyncio.Lock | None = None
        self._window_task: asyncio.Task[None] | None = None

    # -- internal helpers ------------------------------------------------------

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create the lock so it binds to the running event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # -- public API ------------------------------------------------------------

    async def submit(self, item: Any) -> Any:
        """Add *item* to the queue and await its result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        fire_items: list[Any] | None = None
        fire_futures: list[asyncio.Future[Any]] | None = None

        async with self._get_lock():
            self._pending.append((item, future))
            if len(self._pending) >= self._max_size:
                fire_items = [it for it, _ in self._pending]
                fire_futures = [f for _, f in self._pending]
                self._pending.clear()
            elif self._window_task is None or self._window_task.done():
                self._window_task = asyncio.create_task(self._delayed_flush())

        if fire_items is not None and fire_futures is not None:
            asyncio.create_task(self._fire(fire_items, fire_futures))

        return await future

    # -- private ---------------------------------------------------------------

    async def _delayed_flush(self) -> None:
        # Runs until the queue is empty: an item submitted while a batch is
        # firing sees this task still alive and starts no flush of its own.
        while True:
            await asyncio.sleep(self._WINDOW_SECS)
            async with self._get_lock():
                if not self._pending:
                    return
                fire_items = [it for it, _ in self._pending]
                fire_futures = [f for _, f in self._pending]
                self._pending.clear()
            await self._fire(fire_items, fire_futures)

    async def _fire(
        self,
        items: list[Any],
        futures: list[asyncio.Future[Any]],
    ) -> None:
        """Invoke the batch function and resolve waiting futures.

        Handles :class:`RetryWithSmallerBatch` by splitting and recursing.
        """
        try:
            async with self._get_inference_sem():
                if inspect.iscoroutinefunction(self._batch_fn):
                    results: list[Any] = await self._batch_fn(items)
                else:
                    results = await asyncio.to_thread(self._batch_fn, items)

            for fut, res in zip(futures, results, strict=True):
                if not fut.done():
                    fut.set_result(res)

        except RetryWithSmallerBatch:
            if len(items) == 1:
                exc = RuntimeError(
                    "RetryWithSmallerBatch raised with a batch of size 1 — "
                    "cannot halve further."
                )
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)
                return
            mid = len(items) // 2
            await self._fire(items[:mid], futures[:mid])
            await self._fire(items[mid:], futures[mid:])

        except Exception as exc:  # noqa: BLE001
            for fut in futures:
                if not fut.done():
                    fut.set_exception(exc)


# ---------------------------------------------------------------------------
# AsyncWrapper
# ---------------------------------------------------------------------------


class AsyncWrapper:
    """Implements ``@sm.threaded`` and ``@sm.batched(max_size=N)``."""

    @staticmethod
    def threaded(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a sync function so it runs in a thread pool via ``asyncio.to_thread``.

        tree-sitter releases the GIL, so ``@sm.threaded`` parse calls achieve
        true parallelism while keeping the event loop free.
        """

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.to_thread(fn, *args, **kwargs)

        return wrapper

    @staticmethod
    def batched(
        max_size: int,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator routing single-item calls through :class:`BatchGatherer`.

        The decorated function must have the *batch* signature::

            (items: list[T]) -> list[U]   # sync or async

        After decoration it is callable with a **single** item::

            result: U = await fn(item)

        Parameters
        ----------
        max_size:
            Maximum items accumulated before the batch fires immediately.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            gatherer = BatchGatherer(fn, max_size=max_size)

            @functools.wraps(fn)
            async def wrapper(item: Any) -> Any:
                return await gatherer.submit(item)

            return wrapper

        return decorator
