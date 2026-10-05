"""The whole loop: embed real snippets, store them, and retrieve by meaning."""

import hashlib
from pathlib import Path

import pytest

from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder
from smritikosh.adapters.file_source.local import LocalFileSource
from smritikosh.adapters.vector_store.duckdb import DuckDBVectorStore
from smritikosh.indexing.discovery import iter_source_files
from smritikosh.indexing.file_router import FileRouter, JsonExcludeFilter
from smritikosh.indexing.strategies import (
    AstChunkingStrategy,
    RegexChunkingStrategy,
    SectionChunkingStrategy,
)
from smritikosh.models import SourceFile

pytestmark = pytest.mark.slow

CSV_SNIPPET = '''
def load_orders(path: str) -> list[dict]:
    """Read a CSV file and return its rows."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))
'''

RETRY_SNIPPET = '''
def fetch_with_retry(url: str, attempts: int = 3) -> Response:
    """Issue an HTTP GET, retrying with exponential backoff."""
    for attempt in range(attempts):
        try:
            return httpx.get(url)
        except httpx.TransportError:
            time.sleep(2**attempt)
    raise RuntimeError("all attempts failed")
'''

SNIPPETS = (CSV_SNIPPET, RETRY_SNIPPET)


def chunk_id_for(text: str) -> str:
    """Mirror the content-hash chunk ID the indexing pipeline will use."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# Module-scoped: model download (~640 MB) is a one-time cost, so do it once.
@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder()


@pytest.fixture(scope="module")
def store(
    tmp_path_factory: pytest.TempPathFactory,
    embedder: FastEmbedEmbedder,
) -> DuckDBVectorStore:
    db_path: Path = tmp_path_factory.mktemp("index") / "smritikosh.duckdb"
    store = DuckDBVectorStore(str(db_path))
    store.setup(embedder.identity)

    vectors = embedder.encode_documents(list(SNIPPETS))
    for text, vector in zip(SNIPPETS, vectors, strict=True):
        store.upsert(chunk_id_for(text), vector)

    return store


def test_should_retrieve_the_csv_snippet_when_asked_about_csv_files(
    store: DuckDBVectorStore,
    embedder: FastEmbedEmbedder,
) -> None:
    query = embedder.encode_queries(["how do I read rows out of a CSV file"])[0]

    hits = store.search(query, top_k=2)

    assert hits[0][0] == chunk_id_for(CSV_SNIPPET)


def test_should_not_duplicate_rows_when_a_chunk_is_reindexed(
    store: DuckDBVectorStore,
    embedder: FastEmbedEmbedder,
) -> None:
    query = embedder.encode_queries(["anything at all"])[0]
    store.upsert(chunk_id_for(CSV_SNIPPET), embedder.encode_documents([CSV_SNIPPET])[0])

    hits = store.search(query, top_k=10)

    assert len(hits) == len(SNIPPETS)


# A repo with one file per outcome: four worth indexing, four that must be
# dropped -- by excluded directory, by .gitignore, and by the router twice.
REPO = {
    "src/billing/invoice.py": "class Invoice:\n    pass\n",
    "README.md": "# Title\n\nBody.\n",
    "pyproject.toml": "[tool.ruff]\nline-length = 88\n",
    "config/eval_config.json": '{"model": "jina"}\n',
    "node_modules/lodash/index.js": "module.exports = {};\n",
    "secrets.env": "TOKEN=abc\n",
    "package-lock.json": '{"lockfileVersion": 3}\n',
    "assets/logo.png": "binary-ish\n",
    ".gitignore": "secrets.env\n",
}


def test_should_select_and_route_a_real_repo(tmp_path: Path) -> None:
    """LocalFileSource + FileRouter + discovery over files that exist on disk."""
    for rel, text in REPO.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)

    ast = AstChunkingStrategy()
    sections = SectionChunkingStrategy()
    toml = RegexChunkingStrategy(r"^\[+[^\]]+\]")

    router = FileRouter(JsonExcludeFilter())
    router.register_extension(".py", "python", ast)
    router.register_extension(".js", "javascript", ast)
    router.register_extension(".md", "markdown", sections)
    router.register_extension(".json", "json", sections)
    router.register_extension(".toml", "toml", toml, has_tags_scm=False)

    selected = list(iter_source_files(LocalFileSource(str(tmp_path)), router))

    assert selected == [
        SourceFile(
            path="README.md",
            language="markdown",
            content=REPO["README.md"],
            has_tags_scm=True,
            strategy=sections,
        ),
        SourceFile(
            path="config/eval_config.json",
            language="json",
            content=REPO["config/eval_config.json"],
            has_tags_scm=True,
            strategy=sections,
        ),
        SourceFile(
            path="pyproject.toml",
            language="toml",
            content=REPO["pyproject.toml"],
            has_tags_scm=False,
            strategy=toml,
        ),
        SourceFile(
            path="src/billing/invoice.py",
            language="python",
            content=REPO["src/billing/invoice.py"],
            has_tags_scm=True,
            strategy=ast,
        ),
    ]
