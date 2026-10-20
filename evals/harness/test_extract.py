"""Tests for the parts of extraction that a wrong answer looks plausible in.

The tool classifier earned a test the hard way. Its first version matched
the substring "smritikosh" anywhere in a bash command, so the very first
pilot run scored `find smritikosh -name '*.py'` in the grep-only arm as two
retrieval calls. Nothing failed, nothing looked wrong, and the headline
comparison was quietly meaningless. Everything below is a case where a bug
produces a number rather than an error.

Run from evals/harness:  pytest test_extract.py
"""

from __future__ import annotations

import json

import pytest
from extract import _programs, classify_tool, extract_ranges
from grade import citation_hit, score

GREP = "gr" + "ep"  # written this way so the repo's own tooling stays out of it


class TestProgramDetection:
    @pytest.mark.parametrize(
        "command,expected",
        [
            ("find smritikosh -name '*.py'", ["find"]),
            (f"cd /repo && {GREP} -rn foo .", [GREP]),
            ("smritikosh explore search 'a' 'b'", ["smritikosh"]),
            ("FOO=1 graphify query x", ["graphify"]),
            ("cat a.py", ["cat"]),
            (f"{GREP} foo . | head -20", [GREP, "head"]),
            ("/usr/local/bin/smritikosh index .", ["smritikosh"]),
        ],
    )
    def test_programs(self, command, expected):
        assert _programs(command) == expected

    def test_unbalanced_quotes_do_not_raise(self):
        # Agents write commands with stray quotes; shlex raises on those and
        # a crashed extractor would cost the whole run's metrics.
        assert _programs(f'{GREP} "unterminated') == [GREP]


class TestClassifier:
    def test_repo_name_in_a_path_is_not_retrieval(self):
        """The regression. This repository is named after the tool."""
        args = {"command": "find smritikosh -name '*chunk*'"}
        assert classify_tool("bash", args) == "grep_find"

    def test_grep_over_the_index_file_is_still_grep(self):
        args = {"command": f"{GREP} -c . smritikosh.duckdb"}
        assert classify_tool("bash", args) == "grep_find"

    def test_cli_through_bash_is_retrieval(self):
        args = {"command": "smritikosh explore chunks --range a.py 1 20"}
        assert classify_tool("bash", args) == "smritikosh"

    def test_retrieval_piped_into_grep_is_retrieval(self):
        args = {"command": f"smritikosh explore search 'x' | {GREP} rrf"}
        assert classify_tool("bash", args) == "smritikosh"

    def test_extension_tools_classify_by_name(self):
        assert classify_tool("smritikosh_search", {"queries": ["a"]}) == "smritikosh"
        assert classify_tool("graphify_query", {}) == "graphify"
        assert classify_tool("read", {"path": "a.py"}) == "read"


class TestRangeParsing:
    def test_extracts_path_and_range(self):
        text = "see smritikosh/retrieval/fusion.py:76-107 for the merge"
        assert extract_ranges(text) == ["smritikosh/retrieval/fusion.py:76-107"]

    def test_single_line_citation_becomes_a_degenerate_range(self):
        assert extract_ranges("models/retrieval.py:131") == [
            "models/retrieval.py:131-131"
        ]

    def test_bare_path_is_not_a_citation(self):
        # "the answer is somewhere in fusion.py" earns no credit.
        assert extract_ranges("look in smritikosh/retrieval/fusion.py") == []

    def test_duplicates_collapse(self):
        text = "a.py:1-5 and again a.py:1-5"
        assert extract_ranges(text) == ["a.py:1-5"]


class TestCitationHit:
    gold = [{"path": "smritikosh/retrieval/fusion.py", "start": 76, "end": 107}]

    def test_overlap_counts(self):
        hit, matched = citation_hit("see fusion.py:70-120", self.gold)
        assert hit and matched == ["fusion.py:70-120"]

    def test_disjoint_range_does_not_count(self):
        hit, _ = citation_hit("see smritikosh/retrieval/fusion.py:10-20", self.gold)
        assert not hit

    def test_right_lines_in_the_wrong_file_does_not_count(self):
        hit, _ = citation_hit("see smritikosh/retrieval/priors.py:76-107", self.gold)
        assert not hit

    def test_no_citation_at_all(self):
        assert citation_hit("Go is not supported.", self.gold) == (False, [])


class TestCoverageScore:
    def test_partial_is_worth_half(self):
        verdicts = [
            {"fact_index": 0, "verdict": "covered"},
            {"fact_index": 1, "verdict": "partial"},
        ]
        coverage, tally = score(verdicts, total=2)
        assert coverage == 0.75
        assert tally == {"covered": 1, "partial": 1, "missing": 0}

    def test_facts_the_judge_skipped_count_as_missing(self):
        """Omission must never be cheaper for the judge than saying 'missing'."""
        verdicts = [{"fact_index": 0, "verdict": "covered"}]
        coverage, tally = score(verdicts, total=4)
        assert coverage == 0.25
        assert tally["missing"] == 3

    def test_duplicate_verdicts_for_one_fact_are_ignored(self):
        verdicts = [
            {"fact_index": 0, "verdict": "covered"},
            {"fact_index": 0, "verdict": "covered"},
        ]
        coverage, _ = score(verdicts, total=2)
        assert coverage == 0.5


class TestQueryExtraction:
    """Only search text belongs in the dossier the gap analyst reads."""

    def test_grep_pattern_is_the_query(self):
        from extract import extract_queries

        assert extract_queries("bash", {"command": f"{GREP} -rn 'rrf_k' ."}) == [
            "rrf_k"
        ]

    def test_find_directory_is_not_a_query(self):
        from extract import extract_queries

        # The first positional of find is where to look, not what to find.
        assert extract_queries("bash", {"command": "find smritikosh/ -type f"}) == []

    def test_find_name_pattern_is_a_query(self):
        from extract import extract_queries

        out = extract_queries("bash", {"command": "find . -name '*router*'"})
        assert out == ["*router*"]

    def test_facets_come_through_verbatim(self):
        from extract import extract_queries

        args = {"queries": ["rank fusion constant", "merge dense and lexical"]}
        assert extract_queries("smritikosh_search", args) == args["queries"]

    def test_pipeline_noise_is_dropped(self):
        from extract import extract_queries

        out = extract_queries("bash", {"command": f"{GREP} -rn foo . | head -20"})
        assert out == ["foo"]


class TestProviderErrors:
    """A billing problem must never be reported as a retrieval result."""

    def test_error_stop_reason_is_captured(self, tmp_path):
        from extract import extract_run

        run_dir = tmp_path / "q01__bare__r1"
        (run_dir / "session").mkdir(parents=True)
        (run_dir / "meta.json").write_text(
            json.dumps({"question_id": "q01", "arm": "bare", "repeat": 1})
        )
        (run_dir / "session" / "s.jsonl").write_text(
            json.dumps(
                {
                    "type": "message",
                    "timestamp": "2026-10-20T00:00:00Z",
                    "message": {
                        "role": "assistant",
                        "content": [],
                        "stopReason": "error",
                        "errorMessage": "400 invalid_request_error: out of credit",
                    },
                }
            )
            + "\n"
        )
        (run_dir / "stream.jsonl").write_text("")

        record = extract_run(run_dir)
        assert "out of credit" in record["provider_error"]
        assert record["answer"] == ""

    def test_clean_run_has_no_provider_error(self, tmp_path):
        from extract import extract_run

        run_dir = tmp_path / "q01__bare__r1"
        (run_dir / "session").mkdir(parents=True)
        (run_dir / "meta.json").write_text(json.dumps({"question_id": "q01"}))
        (run_dir / "session" / "s.jsonl").write_text(
            json.dumps(
                {
                    "type": "message",
                    "timestamp": "2026-10-20T00:00:00Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Go is not supported."}],
                        "stopReason": "stop",
                    },
                }
            )
            + "\n"
        )
        (run_dir / "stream.jsonl").write_text("")

        record = extract_run(run_dir)
        assert record["provider_error"] is None
        assert record["answer"] == "Go is not supported."


class TestQuoteVerification:
    """The grader's own anti-hallucination check."""

    answer = "Go is not supported.\nNo `.go` extension is registered anywhere."

    def test_exact_quote_passes(self):
        from grade import quote_is_present

        assert quote_is_present("Go is not supported.", self.answer)

    def test_invented_quote_fails(self):
        from grade import quote_is_present

        assert not quote_is_present("Go support was removed in v0.3.", self.answer)

    def test_elided_quote_passes_when_both_halves_are_real(self):
        from grade import quote_is_present

        quote = "Go is not supported.\n\n...\n\nNo `.go` extension is registered"
        assert quote_is_present(quote, self.answer)

    def test_elided_quote_fails_when_a_half_is_invented(self):
        from grade import quote_is_present

        quote = "Go is not supported. ... Rust is also unsupported here"
        assert not quote_is_present(quote, self.answer)

    def test_rewrapped_whitespace_still_passes(self):
        from grade import quote_is_present

        assert quote_is_present("Go is\n   not supported.", self.answer)

    def test_empty_quote_is_allowed_for_missing_verdicts(self):
        from grade import quote_is_present

        assert quote_is_present("", self.answer)

    def test_markdown_emphasis_is_not_invention(self):
        from grade import quote_is_present

        answer = "**smritikosh does not support Go source files**."
        assert quote_is_present("smritikosh does not support Go source files", answer)

    def test_stripping_emphasis_does_not_let_inventions_through(self):
        from grade import quote_is_present

        answer = "**smritikosh does not support Go source files**."
        assert not quote_is_present("smritikosh fully supports Go", answer)


class TestContextComposition:
    """The context breakdown is the eval's central measurement."""

    def _run(self, tmp_path, tool_result_chars: int):
        run_dir = tmp_path / "q01__bare__r1"
        (run_dir / "session").mkdir(parents=True)
        (run_dir / "meta.json").write_text(json.dumps({"question_id": "q01"}))
        entries = [
            {
                "type": "message",
                "timestamp": "2026-10-20T00:00:00Z",
                "message": {
                    "role": "system",
                    "sections": {"preamble": "x" * 400},
                    "toolsAdded": [{"name": "read"}],
                },
            },
            {
                "type": "message",
                "timestamp": "2026-10-20T00:00:01Z",
                "message": {
                    "role": "toolResult",
                    "content": [{"type": "text", "text": "y" * tool_result_chars}],
                },
            },
            {
                "type": "message",
                "timestamp": "2026-10-20T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "the answer"}],
                    "stopReason": "stop",
                },
            },
        ]
        (run_dir / "session" / "s.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )
        (run_dir / "stream.jsonl").write_text("")
        return run_dir

    def test_peak_is_the_sum_of_the_parts(self, tmp_path):
        from extract import extract_run

        ctx = extract_run(self._run(tmp_path, 4000))["context"]
        parts = ctx["system_tok"] + ctx["tool_result_tok"]
        parts += ctx["assistant_tok"] + ctx["user_tok"]
        assert ctx["peak_tok"] == parts

    def test_tool_results_dominate_a_grep_style_run(self, tmp_path):
        from extract import extract_run

        ctx = extract_run(self._run(tmp_path, 40000))["context"]
        # The whole claim of the eval: unbounded tool output is the context.
        assert ctx["tool_result_tok"] > 5 * ctx["system_tok"]

    def test_cost_is_broken_down_not_just_totalled(self, tmp_path):
        from extract import extract_run

        run_dir = self._run(tmp_path, 100)
        path = run_dir / "session" / "s.jsonl"
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        entries[-1]["message"]["usage"] = {
            "input": 10,
            "output": 20,
            "cacheRead": 1000,
            "cacheWrite": 500,
            "cost": {
                "input": 0.001,
                "output": 0.002,
                "cacheRead": 0.003,
                "cacheWrite": 0.004,
                "total": 0.01,
            },
        }
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        cost = extract_run(run_dir)["cost"]
        assert cost["total"] == 0.01
        assert cost["cacheWrite"] == 0.004

    def test_max_result_chars_catches_the_one_big_dump(self, tmp_path):
        from extract import extract_run

        run_dir = self._run(tmp_path, 100)
        events = [
            {
                "type": "tool_execution_start",
                "toolCallId": str(i),
                "toolName": "read",
                "args": {"path": f"f{i}.py"},
            }
            for i in range(3)
        ]
        sizes = [50, 27157, 80]  # one enormous file read among small ones
        for i, size in enumerate(sizes):
            events.append(
                {
                    "type": "tool_execution_end",
                    "toolCallId": str(i),
                    "toolName": "read",
                    "result": {"content": [{"type": "text", "text": "z" * size}]},
                }
            )
        (run_dir / "stream.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        assert extract_run(run_dir)["max_result_chars"] == 27157
