"""Create one worktree per index profile and build the indexes in it.

Why worktrees: the golden set is pinned to a commit that predates `evals/`,
so an arm checked out there cannot grep its way to the answer sheet. Running
the arms against the live tree would leak the golden file, the harness, and
any uncommitted work into the thing being measured.

Why per *profile* rather than per arm: three arms share the smritikosh
index. Building it once instead of three times saves time and, more
importantly, removes a source of variance — all three arms then search
byte-identical indexes.

Build time and index size are recorded here, as setup, and deliberately kept
out of per-question numbers. Index construction is a once-per-repo cost;
amortising it into every question would flatter whichever arm answers the
most questions.

Usage:
    python prepare.py            # build what the enabled arms need
    python prepare.py --force    # rebuild from scratch
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time

from config import REPO_ROOT, WORKTREES_DIR, load_arms, load_golden


def run(cmd: list[str], cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def ensure_worktree(profile: str, commit: str, force: bool) -> tuple[bool, str]:
    path = WORKTREES_DIR / profile
    if path.exists() and force:
        run(["git", "worktree", "remove", "--force", str(path)], cwd=REPO_ROOT)
        shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        head = run(["git", "rev-parse", "HEAD"], cwd=path).stdout.strip()
        if head.startswith(commit) or commit.startswith(head[:7]):
            return False, "exists"
        return (
            False,
            f"WRONG COMMIT {head[:7]} (expected {commit}) — rerun with --force",
        )

    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    # --detach: these are read-only checkouts, not branches anyone works on.
    result = run(
        ["git", "worktree", "add", "--detach", str(path), commit], cwd=REPO_ROOT
    )
    if result.returncode != 0:
        return False, f"FAILED: {result.stderr.strip()}"
    return True, "created"


def build_smritikosh(path) -> dict:
    binary = shutil.which("smritikosh") or str(REPO_ROOT / ".venv/bin/smritikosh")
    started = time.time()
    result = run([binary, "index", "."], cwd=path)
    elapsed = time.time() - started
    db = path / "smritikosh.duckdb"
    return {
        "tool": "smritikosh",
        "ok": result.returncode == 0,
        "seconds": round(elapsed, 1),
        "bytes": db.stat().st_size if db.exists() else 0,
        # Local embeddings, so the index costs CPU time and zero API credit.
        # Recorded explicitly because "free to build" is a real advantage and
        # a reader should not have to infer it from a missing field.
        "api_cost": 0.0,
        "stderr": result.stderr.strip()[-500:],
    }


def build_graphify(path) -> dict:
    binary = shutil.which("graphify")
    if not binary:
        return {"tool": "graphify", "ok": False, "stderr": "graphify not on PATH"}
    started = time.time()
    result = run([binary, "update", "."], cwd=path)
    elapsed = time.time() - started
    out = path / "graphify-out"
    size = (
        sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        if out.exists()
        else 0
    )
    return {
        "tool": "graphify",
        "ok": result.returncode == 0,
        "seconds": round(elapsed, 1),
        "bytes": size,
        "api_cost": 0.0,  # AST-only build; no LLM credits spent
        "stderr": result.stderr.strip()[-500:],
    }


BUILDERS = {
    "none": [],
    "smritikosh": [build_smritikosh],
    "graphify": [build_graphify],
    "both": [build_smritikosh, build_graphify],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="golden")
    parser.add_argument(
        "--force", action="store_true", help="remove and rebuild worktrees"
    )
    parser.add_argument("--all-arms", action="store_true", help="include disabled arms")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    arms, _ = load_arms()
    selected = [a for a in arms if a.enabled or args.all_arms]
    profiles = sorted({a.profile for a in selected})

    print(f"golden commit : {golden.commit}")
    print(f"arms          : {', '.join(a.id for a in selected)}")
    print(f"profiles      : {', '.join(profiles)}\n")

    failed = False
    for profile in profiles:
        created, status = ensure_worktree(profile, golden.commit, args.force)
        print(f"[{profile}] worktree: {status}")
        if "FAILED" in status or "WRONG COMMIT" in status:
            failed = True
            continue

        path = WORKTREES_DIR / profile
        setup_file = path.parent / f"{profile}.setup.json"
        if not created and setup_file.exists() and not args.force:
            print(f"[{profile}] indexes: already built (--force to rebuild)")
            continue

        builds = []
        for builder in BUILDERS.get(profile, []):
            info = builder(path)
            builds.append(info)
            flag = "ok" if info.get("ok") else "FAILED"
            print(
                f"[{profile}] {info['tool']}: {flag} "
                f"{info.get('seconds', 0)}s {info.get('bytes', 0) / 1e6:.1f}MB "
                f"{info.get('stderr', '')}"
            )
            failed = failed or not info.get("ok")

        setup_file.write_text(
            json.dumps(
                {"profile": profile, "commit": golden.commit, "builds": builds},
                indent=2,
            )
        )
        if not builds:
            print(f"[{profile}] indexes: none by design")

    print("\nready" if not failed else "\ncompleted with failures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
