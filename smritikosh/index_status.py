"""Plain-text and TOON views of one local index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from smritikosh.adapters.embedder.fastembed import _VECTOR_SPACE as _ONNX_SPACE
from smritikosh.adapters.embedder.mps.embedder import _VECTOR_SPACE as _MPS_SPACE
from smritikosh.adapters.embedder.resolver import _mps_available
from smritikosh.adapters.retrieval.source import DuckDBSourceReader
from smritikosh.ports.embedder import EmbeddingIdentity

__all__ = ["IndexStatus", "load_index_status"]


@dataclass(frozen=True)
class IndexStatus:
    """The fields ``smritikosh status`` prints."""

    index_path: str
    size_bytes: int
    files: int
    chunks: int
    vectors: int
    dimensions: int | None
    coverage: float | None
    model: str | None
    vector_space: str | None
    warm_server: str
    languages: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]

    def lines(self) -> list[str]:
        coverage = "n/a" if self.coverage is None else f"{self.coverage:.2f}"
        dimensions = "none" if self.dimensions is None else str(self.dimensions)
        languages = (
            ", ".join(f"{suffix} {count}" for suffix, count in self.languages) or "none"
        )
        rows = [
            f"index {self.index_path}",
            f"size {_format_size(self.size_bytes)}",
            f"files {self.files}",
            f"chunks {self.chunks}",
            f"vectors {self.vectors}",
            f"dimensions {dimensions}",
            f"coverage {coverage}",
            f"model {self.model or 'none'}",
            f"vector space {self.vector_space or 'none'}",
            f"warm server {self.warm_server}",
            f"languages {languages}",
        ]
        rows.extend(f"warning {warning}" for warning in self.warnings)
        return rows

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index_path,
            "size_bytes": self.size_bytes,
            "files": self.files,
            "chunks": self.chunks,
            "vectors": self.vectors,
            "dimensions": self.dimensions,
            "coverage": self.coverage,
            "model": self.model,
            "vector_space": self.vector_space,
            "warm_server": self.warm_server,
            "languages": [
                {"suffix": suffix, "chunks": count} for suffix, count in self.languages
            ],
            "warnings": list(self.warnings),
        }


def load_index_status(db_path: str, *, warm_server: str) -> IndexStatus:
    """Read coverage and embedding identity.

    *warm_server* is the label for this machine's query runtime.
    """
    path = Path(db_path)
    reader = DuckDBSourceReader(str(path))
    try:
        info = reader.index_info()
        identity = reader.embedding_identity()
        languages = tuple(reader.chunk_path_suffixes().items())
    finally:
        reader.close()
    coverage = None if info.chunks == 0 else info.vectors / info.chunks
    warnings = _warnings(info.chunks, info.vectors, identity)
    return IndexStatus(
        index_path=str(path),
        size_bytes=path.stat().st_size,
        files=info.files,
        chunks=info.chunks,
        vectors=info.vectors,
        dimensions=info.dimensions,
        coverage=coverage,
        model=None if identity is None else identity.model,
        vector_space=None if identity is None else identity.vector_space,
        warm_server=warm_server,
        languages=languages,
        warnings=tuple(warnings),
    )


def _warnings(
    chunks: int,
    vectors: int,
    identity: EmbeddingIdentity | None,
) -> list[str]:
    warnings: list[str] = []
    if chunks > vectors:
        warnings.append("Some chunks were stored and not embedded.")
    if identity is None:
        warnings.append(
            "No embedding identity is stored. Rebuild with `smritikosh index`."
        )
        return warnings
    if identity.vector_space == _ONNX_SPACE:
        return warnings
    if identity.vector_space == _MPS_SPACE:
        if not _mps_available():
            warnings.append(
                "This index uses the Apple GPU vector space, which this "
                "machine cannot read. Rebuild it here so search can use "
                "these vectors."
            )
        return warnings
    warnings.append(
        f"No installed embedder supports vector space {identity.vector_space!r}."
    )
    return warnings


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def open_error(exc: duckdb.Error) -> str:
    """Turn a missing-table failure into a sentence the CLI can print."""
    return f"This file is not a Smritikosh index ({exc})."
