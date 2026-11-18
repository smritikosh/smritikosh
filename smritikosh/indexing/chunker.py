"""Stage 3: chunk orchestration — delegates to a ChunkingStrategy."""

from dataclasses import replace

from smritikosh.engine import sm
from smritikosh.models import Capture, Chunk, ChunkingStrategy, ParsedFile


@sm.tracked
def chunk_file(
    parsed: ParsedFile,
    captures: list[Capture],
    strategy: ChunkingStrategy,
) -> list[Chunk]:
    """Delegate chunking to *strategy*, keeping the source of each chunk.

    A strategy may shape a chunk's text for retrieval — a Markdown heading
    breadcrumb, a JSON key path — so that text is not always the file's own
    lines. Those lines are then kept beside it, so a reader can still rebuild
    the file from the index.
    """
    source_lines: list[str] = parsed.content.split("\n")
    return [
        _with_source_text(chunk, source_lines)
        for chunk in strategy.chunk(parsed, captures)
    ]


def _with_source_text(chunk: Chunk, source_lines: list[str]) -> Chunk:
    """Return *chunk* with its source lines when its text differs from them."""
    source: str = "\n".join(source_lines[chunk.start_line - 1 : chunk.end_line])
    if chunk.text == source:
        return chunk
    return replace(chunk, source_text=source)
