"""File routing configuration — maps extensions to languages and strategies.

Adding a new language is one register_extension() call here.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from smritikosh.indexing.file_router import FileRouter, JsonExcludeFilter
from smritikosh.indexing.strategies import (
    AstChunkingStrategy,
    RegexChunkingStrategy,
    SectionChunkingStrategy,
)


@functools.cache
def _build_router(
    max_chars: int | None = None,
    max_tokens: int | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> FileRouter:
    """Return a FileRouter pre-configured for all supported file types.

    Cached on *max_chars*: strategies and router are constructed once per
    process per chunk budget and reused across every _run_pipeline call (e.g.
    in --watch mode).  *max_chars* caps a chunk at what the embedder will
    actually encode; ``None`` leaves chunks uncapped.
    """
    ast = AstChunkingStrategy(max_chars)
    sections = SectionChunkingStrategy(max_chars, max_tokens, token_counter)
    toml = RegexChunkingStrategy(r"^\[+[^\]]+\]", max_chars)

    router = FileRouter(json_exclude_filter=JsonExcludeFilter())
    router.register_extension(".py", "python", ast)
    router.register_extension(".ts", "typescript", ast)
    router.register_extension(".tsx", "typescript", ast)
    router.register_extension(".js", "javascript", ast)
    router.register_extension(".jsx", "javascript", ast)
    router.register_extension(".java", "java", ast)
    router.register_extension(".kt", "kotlin", ast)
    router.register_extension(".kts", "kotlin", ast)
    router.register_extension(".md", "markdown", sections)
    router.register_extension(".mdx", "markdown", sections)
    router.register_extension(".toml", "toml", toml, has_tags_scm=False)
    router.register_extension(".json", "json", sections)
    router.register_extension(".yaml", "yaml", sections)
    router.register_extension(".yml", "yaml", sections)
    return router
