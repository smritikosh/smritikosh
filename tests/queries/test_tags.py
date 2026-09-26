"""Tests for the bundled tree-sitter tag queries.

Compiling a query proves only that it parses; a pattern naming a node type the
grammar does not emit compiles happily and captures nothing. So each language
also gets a snippet exercising the constructs it is meant to recognise.
"""

import importlib.resources

import pytest
from tree_sitter import Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language

LANGUAGES = [
    "python",
    "java",
    "kotlin",
    "typescript",
    "javascript",
    "markdown",
    "json",
    "yaml",
]


def _queries_root():
    return importlib.resources.files("smritikosh.queries")


def _query_text(language: str) -> str:
    return (_queries_root() / language / "tags.scm").read_text(encoding="utf-8")


def _capture_names(language: str, source: str) -> set[str]:
    lang = get_language(language)
    # Parser(get_language(...)) rather than get_parser(...): only the former
    # yields nodes the standalone tree_sitter.QueryCursor accepts.
    root = Parser(lang).parse(source.encode("utf-8")).root_node
    captures = QueryCursor(Query(lang, _query_text(language))).captures(root)
    return {name for name in captures if name.startswith("definition.")}


@pytest.mark.parametrize("language", LANGUAGES)
def test_should_compile_the_query_for_each_supported_language(language: str) -> None:
    Query(get_language(language), _query_text(language))


def test_should_ship_no_toml_query_because_toml_chunks_by_regex() -> None:
    assert not (_queries_root() / "toml").is_dir()


PYTHON = """
MAX_RETRIES: Final[int] = 3
SerializableId = Annotated[str, PlainSerializer(str)]


class Tenant(str, Enum):
    QBO = "qbo"


class Mode(StrEnum):
    FAST = "fast"


class Client:
    TIMEOUT = 30

    def __init__(self) -> None: ...

    async def fetch(self) -> str: ...


async def top_level() -> None: ...
"""


def test_should_capture_python_enums_constants_and_async_definitions() -> None:
    assert _capture_names("python", PYTHON) == {
        "definition.constant",  # annotated module constant
        "definition.type",  # PascalCase alias, missed by the constant pattern
        "definition.enum",  # str+Enum mixin and StrEnum
        "definition.class",
        "definition.class_constant",
        "definition.class_init",  # __init__, absorbed into the class chunk
        "definition.method",  # async def in a class body
        "definition.function",  # async def at module level
    }


TYPESCRIPT = """
export const MAX_RETRIES = 3;
export type UserId = string;

export interface Repo {
  find(id: UserId): Promise<string>;
}

export enum Status {
  Active = "active",
}

export class Client implements Repo {
  static TIMEOUT = 30;
  constructor(private base: string) {}
  async find(id: UserId): Promise<string> {
    return this.base + id;
  }
}

export function plain(a: number): number {
  return a;
}

export const arrow = (a: number): number => a;
"""


def test_should_capture_typescript_classes_and_functions_missing_upstream() -> None:
    assert _capture_names("typescript", TYPESCRIPT) == {
        "definition.class",  # class_declaration, absent from the official file
        "definition.function",  # function_declaration, likewise absent
        "definition.interface",
        "definition.enum",
        "definition.type",
        "definition.class_init",
        "definition.class_constant",
        "definition.method",
        "definition.constant",
    }


JAVASCRIPT = """
const MAX_RETRIES = 3;

class Client {
  static TIMEOUT = 30;
  constructor(base) {
    this.base = base;
  }
  async find(id) {
    return this.base + id;
  }
}

function plain(a) {
  return a;
}

const arrow = (a) => a;
"""


def test_should_capture_javascript_constructors_and_constants() -> None:
    assert _capture_names("javascript", JAVASCRIPT) == {
        "definition.class",
        "definition.class_init",
        "definition.class_constant",
        "definition.method",
        "definition.function",
        "definition.constant",
    }


JAVA = """
public interface Repo {
    String find(String id);
}

public enum Status {
    ACTIVE
}

public class Client implements Repo {
    public static final int TIMEOUT = 30;

    public Client(String base) {}

    @Override
    public String find(String id) {
        return id;
    }
}
"""


def test_should_capture_java_constructors_enums_and_static_finals() -> None:
    assert _capture_names("java", JAVA) == {
        "definition.interface",
        "definition.enum",
        "definition.class",
        "definition.class_init",  # constructor_declaration
        "definition.class_constant",  # enum_constant and static final field
        "definition.method",
    }


KOTLIN = """
package demo

const val MAX_RETRIES = 3

typealias UserId = String

interface Repo {
    fun find(id: UserId): String
}

enum class Status {
    ACTIVE
}

class Client(private val base: String) : Repo {

    constructor(base: String, port: Int) : this(base)

    companion object {
        const val TIMEOUT = 30
    }

    override fun find(id: UserId): String {
        return base + id
    }
}

fun topLevel(a: Int): Int = a
"""


def test_should_capture_kotlin_enum_classes_and_secondary_constructors() -> None:
    assert _capture_names("kotlin", KOTLIN) == {
        "definition.constant",
        "definition.type",
        "definition.interface",
        "definition.enum",  # enum class, via its enum_class_body
        "definition.class",
        "definition.class_init",  # primary and secondary constructors
        "definition.class_constant",  # enum entry and companion-object const
        "definition.method",
        "definition.function",
    }


MARKDOWN = """# Title

Intro prose.

## Section Two

More prose.

```python
print(1)
```
"""


def test_should_capture_markdown_headings_and_code_blocks() -> None:
    assert _capture_names("markdown", MARKDOWN) == {
        "definition.section",
        "definition.code_block",
    }


def test_should_capture_markdown_heading_nodes_without_recursive_body_text() -> None:
    lang = get_language("markdown")
    root = Parser(lang).parse(MARKDOWN.encode("utf-8")).root_node
    captures = QueryCursor(Query(lang, _query_text("markdown"))).captures(root)

    sections = captures["definition.section"]

    assert [node.text.rstrip() for node in sections] == [
        b"# Title",
        b"## Section Two",
    ]


JSON = """
{
  "name": "demo",
  "nested": {"inner": 1}
}
"""


def test_should_capture_json_keys_at_three_depths() -> None:
    """The query offers depth; the strategy decides how much of it to use.

    Nested keys are captured under their own capture names so
    SectionChunkingStrategy can descend into an oversized section.  Keeping a
    small config flat is the *strategy's* job (it stops at the first depth that
    fits the budget), not the query's — see test_section_strategy.
    """
    lang = get_language("json")
    root = Parser(lang).parse(JSON.encode("utf-8")).root_node
    captures = QueryCursor(Query(lang, _query_text("json"))).captures(root)

    names = {node.text.decode() for node in captures["name"]}

    assert names == {"name", "nested", "inner"}
    assert {n.text.decode() for n in captures["definition.section"]} == {
        '"name": "demo"',
        '"nested": {"inner": 1}',
    }
    assert {n.text.decode() for n in captures["definition.subsection"]} == {
        '"inner": 1'
    }


YAML = """\
apiVersion: batch/v1
kind: CronJob
metadata:
  name: trig-stmt-gen
spec:
  jobTemplate:
    spec:
      backoffLimit: 2
"""


def test_should_capture_yaml_keys_at_three_depths() -> None:
    """Top-level keys, the keys under them, and one level further.

    A short manifest still emits the deeper names. The section strategy
    decides which depth becomes a chunk.
    """
    lang = get_language("yaml")
    root = Parser(lang).parse(YAML.encode("utf-8")).root_node
    captures = QueryCursor(Query(lang, _query_text("yaml"))).captures(root)

    sections = {
        node.text.decode().rstrip("\n") for node in captures["definition.section"]
    }
    assert sections == {
        "apiVersion: batch/v1",
        "kind: CronJob",
        "metadata:\n  name: trig-stmt-gen",
        "spec:\n  jobTemplate:\n    spec:\n      backoffLimit: 2",
    }
    assert "name: trig-stmt-gen" in {
        node.text.decode() for node in captures["definition.subsection"]
    }
    assert any(
        b"backoffLimit" in node.text for node in captures["definition.subsubsection"]
    )


def test_should_anchor_the_json_capture_on_the_pair_not_the_key() -> None:
    """The value has to travel with the key or the chunk carries no content.

    Documented in the .scm comment but untested until now.
    """
    lang = get_language("json")
    root = Parser(lang).parse(JSON.encode("utf-8")).root_node
    captures = QueryCursor(Query(lang, _query_text("json"))).captures(root)

    texts = [node.text.decode() for node in captures["definition.section"]]

    assert all(":" in text for text in texts)
