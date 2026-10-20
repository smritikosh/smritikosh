"""Shared paths and loaders for the eval harness.

Every script reads its configuration through here so that the arm matrix,
the golden set, and the directory layout have exactly one definition.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re

import yaml

EVALS_DIR = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = EVALS_DIR.parent
ARMS_DIR = EVALS_DIR / "arms"
DOCS_DIR = ARMS_DIR / "docs"
GOLDEN_DIR = EVALS_DIR / "golden"
RUNS_DIR = EVALS_DIR / "runs"
RESULTS_DIR = EVALS_DIR / "results"
WORKTREES_DIR = EVALS_DIR / ".worktrees"

# A citation the agent can earn credit for: a path followed by a line range.
# Both `path.py:10-20` and `path.py:10` are accepted; a bare path is not,
# because "the answer is somewhere in fusion.py" is not a line citation.
CITATION_RE = re.compile(
    r"(?P<path>[\w./\-]+\.[A-Za-z]{1,6}):(?P<start>\d+)(?:\s*[-–]\s*(?P<end>\d+))?"
)


@dataclasses.dataclass(frozen=True)
class Arm:
    id: str
    script: str
    docs: list[str] | str | None
    profile: str
    enabled: bool
    note: str = ""

    @property
    def script_path(self) -> pathlib.Path:
        return ARMS_DIR / self.script

    @property
    def docs_files(self) -> list[str]:
        """Docs as a list, whether the YAML gave one name or several."""
        if not self.docs:
            return []
        return [self.docs] if isinstance(self.docs, str) else list(self.docs)

    @property
    def docs_paths(self) -> list[pathlib.Path]:
        return [DOCS_DIR / name for name in self.docs_files]

    @property
    def docs_label(self) -> str:
        return "+".join(self.docs_files) or "-"


@dataclasses.dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    key_facts: list[str]
    gold_citations: list[dict]


@dataclasses.dataclass(frozen=True)
class Golden:
    commit: str
    questions: list[Question]


def load_arms(path: pathlib.Path | None = None) -> tuple[list[Arm], dict]:
    """Return (arms, defaults). Disabled arms are included; callers filter.

    Model choices in `defaults` are overridable by environment variable, so
    a whole sweep can be pointed at a different provider without editing a
    tracked file. Precedence, loosest last:

        arms.yaml  <  EVAL_MODEL / EVAL_JUDGE_MODEL  <  --model / --judge-model

    The env layer earns its place because provider access changes for
    reasons unrelated to the experiment \u2014 an exhausted plan, a key that only
    works for one vendor \u2014 and none of those are a reason to commit a diff.
    """
    data = yaml.safe_load((path or ARMS_DIR / "arms.yaml").read_text())
    arms = [Arm(**row) for row in data["arms"]]
    defaults = dict(data.get("defaults", {}))
    for key, variable in (
        ("model", "EVAL_MODEL"),
        ("judge_model", "EVAL_JUDGE_MODEL"),
        ("analyst_model", "EVAL_ANALYST_MODEL"),
        ("summary_model", "EVAL_SUMMARY_MODEL"),
        ("thinking", "EVAL_THINKING"),
    ):
        value = os.environ.get(variable)
        if value:
            defaults[key] = value
    return arms, defaults


def load_golden(name: str = "golden") -> Golden:
    """Load `golden/<name>.yaml`.

    The default set is `golden.yaml`. The parameter exists so a second set
    can be added later \u2014 a different repository, or a harder tier \u2014 without
    any script learning a new filename.
    """
    data = yaml.safe_load((GOLDEN_DIR / f"{name}.yaml").read_text())
    # A key fact containing an unquoted colon parses as a mapping, and a
    # mapping stringifies into the judge prompt as `{'Supported': '.py ...'}`,
    # braces and all. That grades the answer against YAML punctuation. It
    # happened once; failing loudly here is cheaper than noticing it in a
    # verdict weeks later.
    for q in data["questions"]:
        for index, fact in enumerate(q["key_facts"]):
            if not isinstance(fact, str):
                raise SystemExit(
                    f"{name}.yaml: {q['id']} key_facts[{index}] parsed as "
                    f"{type(fact).__name__}, not a string \u2014 quote it:\n"
                    f"  {fact!r}"
                )

    questions = [
        Question(
            id=q["id"],
            category=q["category"],
            # YAML folded scalars keep a trailing newline and soft line
            # breaks; the model should see one clean sentence.
            question=" ".join(q["question"].split()),
            key_facts=list(q["key_facts"]),
            gold_citations=list(q.get("gold_citations", [])),
        )
        for q in data["questions"]
    ]
    return Golden(commit=str(data["commit"]), questions=questions)


def run_id(question_id: str, arm_id: str, repeat: int) -> str:
    """Stable identity for one cell of the matrix.

    Used as a directory name and as the resume key, so re-running the
    harness skips completed work instead of paying for it twice.
    """
    return f"{question_id}__{arm_id}__r{repeat}"


def parse_run_id(value: str) -> tuple[str, str, int]:
    question_id, arm_id, repeat = value.split("__")
    return question_id, arm_id, int(repeat.lstrip("r"))
