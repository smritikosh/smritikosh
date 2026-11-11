from textwrap import dedent

from tree_sitter import Parser
from tree_sitter_language_pack import get_language

from smritikosh.indexing.extractor import extract_file
from smritikosh.indexing.graph import graph_file, resolve
from smritikosh.models import ParsedFile


def _graph(path: str, source: str, language: str = "python"):
    source = dedent(source)
    tree = Parser(get_language(language)).parse(source.encode())
    parsed = ParsedFile(path=path, language=language, content=source, tree=tree)
    return graph_file(parsed, extract_file(parsed, has_tags_scm=True))


def _edges(files: dict[str, str]) -> set[tuple[str, str]]:
    symbols, calls, imports = [], [], []
    for path, source in files.items():
        s, c, i = _graph(path, source)
        symbols += s
        calls += c
        imports += i
    return {(e.src, e.dst) for e in resolve(set(files), symbols, calls, imports)}


def test_should_name_methods_after_their_class() -> None:
    symbols, _, _ = _graph(
        "a.py",
        """
        class Repo:
            def __init__(self): ...
            def save(self): ...

        def save(): ...
        """,
    )

    assert {(s.id, s.kind, s.parent) for s in symbols} == {
        ("a.py:Repo", "class", None),
        ("a.py:Repo.__init__", "method", "a.py:Repo"),
        ("a.py:Repo.save", "method", "a.py:Repo"),
        ("a.py:save", "function", None),
    }


def test_should_skip_builtins_and_module_level_calls() -> None:
    _, calls, _ = _graph(
        "a.py",
        """
        def run():
            print(len([]))
            helper()

        run()
        """,
    )

    assert [(c.scope, c.name, c.line) for c in calls] == [("a.py:run", "helper", 4)]


def test_should_return_nothing_for_languages_without_refs_query() -> None:
    assert _graph("A.java", "class A { void f() { g(); } }", "java") == ([], [], [])


def test_should_resolve_calls_within_a_file() -> None:
    edges = _edges(
        {
            "a.py": """
            def helper(): ...

            def run():
                helper()
            """
        }
    )

    assert edges == {("a.py:run", "a.py:helper")}


def test_should_resolve_self_calls_to_the_enclosing_class() -> None:
    edges = _edges(
        {
            "a.py": """
            class Cache:
                def update(self): ...

                def refresh(self):
                    self.update()

            class Other:
                def update(self): ...
            """
        }
    )

    assert edges == {("a.py:Cache.refresh", "a.py:Cache.update")}


def test_should_resolve_methods_called_on_a_new_instance() -> None:
    edges = _edges(
        {
            "repo.py": """
            class Repo:
                def save(self): ...
            """,
            "app.py": """
            from repo import Repo

            def run():
                Repo().save()
            """,
        }
    )

    assert edges == {
        ("app.py:run", "repo.py:Repo"),
        ("app.py:run", "repo.py:Repo.save"),
    }


def test_should_not_guess_methods_on_a_returned_value() -> None:
    edges = _edges(
        {
            "a.py": """
            class Repo:
                def save(self): ...

            def make_repo(): ...

            def run():
                make_repo().save()
            """
        }
    )

    assert edges == {("a.py:run", "a.py:make_repo")}


def test_should_not_guess_the_target_of_an_untyped_receiver() -> None:
    edges = _edges(
        {
            "a.py": """
            class Client:
                def send(self): ...

            def run(client):
                client.send()
            """
        }
    )

    assert edges == set()


def test_should_follow_from_imports() -> None:
    edges = _edges(
        {
            "pkg/__init__.py": "",
            "pkg/models.py": """
            class User:
                def load(self): ...

            def make(): ...
            """,
            "app.py": """
            from pkg.models import User, make as build

            def run():
                User()
                User.load()
                build()
            """,
        }
    )

    assert edges == {
        ("app.py:run", "pkg/models.py:User"),
        ("app.py:run", "pkg/models.py:User.load"),
        ("app.py:run", "pkg/models.py:make"),
    }


def test_should_follow_module_aliases() -> None:
    edges = _edges(
        {
            "pkg/__init__.py": "",
            "pkg/util.py": "def slugify(): ...",
            "a.py": """
            import pkg.util as u

            def run():
                u.slugify()
            """,
            "b.py": """
            from pkg import util

            def run():
                util.slugify()
            """,
        }
    )

    assert edges == {
        ("a.py:run", "pkg/util.py:slugify"),
        ("b.py:run", "pkg/util.py:slugify"),
    }


def test_should_follow_relative_imports_under_a_source_root() -> None:
    edges = _edges(
        {
            "src/pkg/__init__.py": "",
            "src/pkg/util.py": "def slugify(): ...",
            "src/pkg/api/__init__.py": "",
            "src/pkg/api/views.py": """
            from ..util import slugify
            from .. import util

            def index():
                slugify()

            def detail():
                util.slugify()
            """,
        }
    )

    assert edges == {
        ("src/pkg/api/views.py:index", "src/pkg/util.py:slugify"),
        ("src/pkg/api/views.py:detail", "src/pkg/util.py:slugify"),
    }


def test_should_not_resolve_a_name_that_is_not_imported() -> None:
    edges = _edges(
        {
            "a.py": """
            def run(callback):
                callback()
            """,
            "b.py": "def callback(): ...",
        }
    )

    assert edges == set()


def test_should_not_fall_back_when_the_import_is_external() -> None:
    edges = _edges(
        {
            "client.py": "def get(): ...",
            "app.py": """
            from requests import get

            def run():
                get()
            """,
        }
    )

    assert edges == set()
