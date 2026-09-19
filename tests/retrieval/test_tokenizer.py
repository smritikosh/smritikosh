"""Tests for code-aware lexical tokenization."""

from smritikosh.retrieval.tokenizer import tokenize_code


def test_should_retain_and_split_code_identifiers() -> None:
    tokens = tokenize_code("src/report/UnbilledInterest.py idempotent_handler")

    assert "unbilledinterest" in tokens
    assert {"unbilled", "interest"} <= set(tokens)
    assert "idempotent_handler" in tokens
    assert {"idempotent", "handler"} <= set(tokens)


def test_should_drop_single_characters_and_bare_numbers() -> None:
    tokens = tokenize_code("x = sha256(v2) + 1")

    assert "sha256" in tokens, "the identifier itself stays findable"
    assert "sha" in tokens
    assert "v2" in tokens, "a short identifier is not a single character"
    assert {"x", "1", "2", "256", "v"}.isdisjoint(tokens)
