#!/usr/bin/env python3
"""Phase C finish: root cleanup, path fixes, approach README stubs."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "Frozen-BN-Narrative-Evidence-2026-07-20"
LLM = REPO / "LLM-Narrative-Parser-Experiments-2026-07-15"
STRUCT = REPO / "Structural-Mapping-Retrieval-2026-07-25"
EMBED = REPO / "Embedding-Similarity-Counting-Path-2026-03-25"
ZHANG = REPO / "Zhang-Replication-Foundation-2026-06-04"
BNCPT = REPO / "BN-CPT-Upgrade-Experiments-2026-06-19"
ARCHIVE = REPO / "archive"
SHARED = REPO / "shared"

FROZEN_DOCS = FROZEN / "docs_FrozenBN"
FROZEN_OUT = FROZEN / "outputs"
DATA = SHARED / "data" / "processed"
ZHANG_REF = ZHANG / "reference"


def move_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    print(f"  move {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")
    shutil.move(str(src), str(dst))


def cleanup_root() -> None:
    print("=== Root cleanup ===")
    docs = REPO / "docs"
    if docs.is_dir():
        move_if_exists(docs / "AGENT_CURSOR_DOCS.md", ARCHIVE / "AGENT_CURSOR_DOCS.md")
        move_if_exists(docs / "catchall_decomposition_results.json", LLM / "outputs" / "catchall_decomposition_results.json")
        move_if_exists(docs / "catchall_llm_cache.json", LLM / "outputs" / "catchall_llm_cache.json")
        move_if_exists(docs / "week9_lr_pyspark_with_notes.ipynb", ARCHIVE / "week9_lr_pyspark_with_notes.ipynb")
        for name in (
            "qc_embed_keys.json", "qc_embed_vecs.npy",
            "qc_robust_embed_keys.json", "qc_robust_embed_vecs.npy",
            "qc_robust_paraphrase.json",
        ):
            move_if_exists(docs / name, EMBED / "outputs" / name)
        if (docs / "presentations").is_dir():
            move_if_exists(docs / "presentations", ARCHIVE / "docs_presentations_legacy")
        if (docs / "image").is_dir():
            move_if_exists(docs / "image", ARCHIVE / "docs_image_legacy")
        for f in docs.iterdir():
            if f.name.startswith("~$"):
                f.unlink(missing_ok=True)
        if docs.is_dir() and not any(docs.iterdir()):
            docs.rmdir()
            print("  removed empty docs/")

    move_if_exists(REPO / "methodology_audit_june_22_2026", ARCHIVE / "methodology_audit_june_22_2026")
    move_if_exists(REPO / "evaluation", SHARED / "evaluation_legacy")

    ckpt = REPO / "checkpoints" / "model"
    if ckpt.is_dir():
        for f in ckpt.iterdir():
            move_if_exists(f, DATA / f.name)
    if (REPO / "checkpoints").is_dir() and not any((REPO / "checkpoints").rglob("*")):
        shutil.rmtree(REPO / "checkpoints", ignore_errors=True)

    for empty in ("tests", "data", "outputs", "experiments", "models"):
        p = REPO / empty
        if p.is_dir():
            leftovers = list(p.rglob("*"))
            if not leftovers or all(x.name in (".DS_Store",) or ".mplcache" in str(x) for x in leftovers if x.is_file()):
                shutil.rmtree(p, ignore_errors=True)
                print(f"  removed stale root {empty}/")


def fix_test_paths() -> None:
    print("=== Path fixes in Frozen-BN/tests ===")
    tests_dir = FROZEN / "tests"
    header_old = re.compile(
        r"^ROOT = Path\(__file__\)\.resolve\(\)\.parents\[1\]\s*\n"
        r"(?:sys\.path\.insert\(0, str\(ROOT\)\)\s*\n)?",
        re.MULTILINE,
    )
    header_new = (
        "REPO_ROOT = Path(__file__).resolve().parents[2]\n"
        "FROZEN_DIR = Path(__file__).resolve().parents[1]\n"
        "_SHARED = REPO_ROOT / \"shared\" / \"code\"\n"
        "_FROZEN_CODE = FROZEN_DIR / \"code\"\n"
        "for _p in (_SHARED, _FROZEN_CODE):\n"
        "    if str(_p) not in sys.path:\n"
        "        sys.path.insert(0, str(_p))\n"
        "ROOT = REPO_ROOT\n"
    )
    subs = [
        ('ROOT / "data" / "processed"', 'ROOT / "shared" / "data" / "processed"'),
        ('ROOT / "docs"', 'ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "docs_FrozenBN"'),
        ('ROOT / "outputs"', 'ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs"'),
        (
            'ROOT / "Zhang\'s Approach 2026"',
            'ROOT / "Zhang-Replication-Foundation-2026-06-04" / "reference"',
        ),
        ("pathlib.Path(__file__).resolve().parents[1]", "pathlib.Path(__file__).resolve().parents[2]"),
    ]
    count = 0
    for py in tests_dir.glob("*.py"):
        if py.name == "conftest.py":
            continue
        text = py.read_text(encoding="utf-8")
        orig = text
        text = header_old.sub(header_new, text, count=1)
        for old, new in subs:
            text = text.replace(old, new)
        if text != orig:
            py.write_text(text, encoding="utf-8")
            count += 1
            print(f"  fixed {py.name}")
    print(f"  updated {count} test files")


def write_shared_data_readme() -> None:
    p = SHARED / "data" / "README.md"
    if p.exists():
        return
    p.write_text(
        """# Shared processed data

Canonical location for NTSB incident JSON, embeddings, and train/window splits.

## Required for Frozen-BN reproduction

| File | Purpose |
|------|---------|
| `processed/refined_dataset.json` | Full corpus 1982–2019 |
| `processed/refined_dataset_1982_2006.json` | Zhang training window |
| `processed/embeddings*.npy` + `embeddings_map*.json` | Retrieval index |
| `processed/cause_statistics.json` | Cause counts |

Zhang BTS departures table and `NTSB.xdsl` live in
`Zhang-Replication-Foundation-2026-06-04/reference/`.

Set `PYTHONPATH=shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code` from repo root.
""",
        encoding="utf-8",
    )
    print("  wrote shared/data/README.md")


def write_approach_readmes() -> None:
    print("=== Approach READMEs ===")
    specs = {
        LLM: (
            "LLM Narrative Parser Experiments",
            "REJECTED for main paper (tier-2 fallback only in Frozen-BN).",
            "tests/",
            "docs_LLM/",
            "outputs/",
        ),
        STRUCT: (
            "Structural Mapping + Retrieval",
            "EXPLORATORY — section 8/9 generators; not main eval path.",
            "worked_examples/, tests/",
            "docs_StructMap/, presentations_StructMap/",
            "outputs/",
        ),
        EMBED: (
            "Embedding Similarity Counting Path",
            "SUPERSEDED by Frozen-BN for paper; kept for ablations and Zhang counting parity.",
            "tests/, code/",
            "docs_Embedding/",
            "outputs/",
        ),
        ZHANG: (
            "Zhang Replication Foundation",
            "REFERENCE — vendored paper code, BTS xlsx, NTSB.xdsl, replication scripts.",
            "scripts/, reference/",
            "docs_Zhang/",
            "outputs/",
        ),
        BNCPT: (
            "BN CPT Upgrade Experiments",
            "REJECTED branch — sparse CPT / envelope experiments from June 2026.",
            "code/, tests/",
            "docs_BNUpgrade/",
            "outputs/",
        ),
    }
    for folder, (title, status, key_dirs, docs_dir, out_dir) in specs.items():
        readme = folder / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# {title}\n\n**Status:** {status}\n\n"
                f"## Layout\n\n- `{key_dirs}` — code and runners\n"
                f"- `{docs_dir}` — notes and reports\n"
                f"- `{out_dir}` — run artifacts\n\n"
                f"See repo root `PROJECT_MAP.md` for how this approach relates to "
                f"`Frozen-BN-Narrative-Evidence-2026-07-20/` (current paper path).\n",
                encoding="utf-8",
            )
            print(f"  wrote {readme.relative_to(REPO)}")
        idx = folder / "FILE_INDEX.md"
        if not idx.exists():
            lines = [f"# File index — {folder.name}\n"]
            for sub in sorted(folder.rglob("*")):
                if sub.is_file() and sub.name not in ("README.md", "FILE_INDEX.md"):
                    if sub.suffix in (".py", ".md", ".json", ".sh", ".ipynb"):
                        lines.append(f"- `{sub.relative_to(folder)}`")
            idx.write_text("\n".join(lines[:200]) + ("\n" if len(lines) > 200 else ""), encoding="utf-8")
            print(f"  wrote {idx.relative_to(REPO)}")


def update_gitignore() -> None:
    print("=== .gitignore ===")
    gi = REPO / ".gitignore"
    extra = """

# Repo-specific
.env
.venv/
.pytest_cache/
__pycache__/
*.pyc
.DS_Store
.cursor/state/
archive/docs_*_legacy/

# Large artifacts (keep small JSON results in git if needed)
shared/data/processed/*.npy
shared/data/processed/**/*.npy
shared/data/processed/modernbert/
shared/data/processed/openai/
Frozen-BN-Narrative-Evidence-2026-07-20/outputs/*.npy
"""
    text = gi.read_text(encoding="utf-8")
    if "Repo-specific" not in text:
        gi.write_text(text.rstrip() + extra, encoding="utf-8")
        print("  updated .gitignore")


def patch_config_zhang_paths() -> None:
    cfg = SHARED / "code" / "config.py"
    text = cfg.read_text(encoding="utf-8")
    block = '''
# Zhang replication reference (vendored paper assets)
ZHANG_REF_DIR = REPO_ROOT / "Zhang-Replication-Foundation-2026-06-04" / "reference"
ZHANG_BTS_XLSX = ZHANG_REF_DIR / "data" / "table_01_37_061019.xlsx"
ZHANG_METADATA_XLSX = ZHANG_REF_DIR / "data" / "metaData.xlsx"
ZHANG_XDSL = ZHANG_REF_DIR / "NTSB.xdsl"
DOCS_FROZEN_DIR = FROZEN_BN_DIR / "docs_FrozenBN"
'''
    if "ZHANG_REF_DIR" not in text:
        text = text.replace(
            "FROZEN_BN_DIR = REPO_ROOT / \"Frozen-BN-Narrative-Evidence-2026-07-20\"\n",
            "FROZEN_BN_DIR = REPO_ROOT / \"Frozen-BN-Narrative-Evidence-2026-07-20\"\n" + block,
        )
        cfg.write_text(text, encoding="utf-8")
        print("  patched config.py with Zhang reference paths")


def main() -> None:
    cleanup_root()
    fix_test_paths()
    patch_config_zhang_paths()
    write_shared_data_readme()
    write_approach_readmes()
    update_gitignore()
    print("\nPhase C finish script complete.")


if __name__ == "__main__":
    main()
