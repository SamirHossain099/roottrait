"""Only the package and its analysis are public; the working directory is not.

This directory also holds the unpublished manuscript, working notes and submission material. Two
routes could publish them: the git repository (and the Zenodo archive made from it) and the sdist.
Both are checked here, and the guard was seen to fail on a planted file before it was trusted.
"""
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Never public, whatever their extension.
FORBIDDEN = ("CLAUDE.md", "FINDINGS.md", "CORRECTIONS.md", "LITERATURE.md", "MANUSCRIPT",
             "HIGHLIGHTS", "COVER", "submission/", "prose_scan.py", "register_scan.py",
             "verify_refs.py", "refs_verified.json", "test_manuscript_numbers.py", "_pred.npy",
             "data/", "logs/", ".venv")


def _git_published():
    """Files git would publish: tracked plus untracked-but-not-ignored."""
    if not os.path.isdir(os.path.join(ROOT, ".git")) or not shutil.which("git"):
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln]


def test_git_publishes_nothing_private():
    bad = [f for f in _git_published() if any(x in f for x in FORBIDDEN)]
    assert not bad, f"would be published: {bad}"


def test_git_publishes_the_package_and_the_analysis():
    files = set(_git_published())
    for need in ("src/roottrait/__init__.py", "src/roottrait/traits.py", "pyproject.toml",
                 "LICENSE", "README.md", "src/prmi_analysis.py", "results/synthetic_summary.json"):
        assert need in files, need


def test_published_text_has_no_em_dashes():
    """Standing rule: no em dashes in anything that leaves the machine."""
    bad = []
    for f in _git_published():
        if f.endswith((".py", ".md", ".toml", ".cff", ".yml", ".txt")):
            with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                if chr(0x2014) in fh.read():
                    bad.append(f)
    assert not bad, bad


def test_built_distributions_contain_only_the_library(tmp_path):
    pytest.importorskip("build")
    subprocess.run([sys.executable, "-m", "build", "--outdir", str(tmp_path), ROOT],
                   check=True, capture_output=True)
    for p in tmp_path.iterdir():
        if p.suffix == ".whl":
            names = zipfile.ZipFile(p).namelist()
        elif p.name.endswith(".tar.gz"):
            names = [n.split("/", 1)[1] for n in tarfile.open(p).getnames() if "/" in n]
        else:
            continue
        stray = [n for n in names if n and not n.startswith(("roottrait", "src/roottrait",
                                                             "roottrait-", "PKG-INFO",
                                                             "pyproject.toml", ".gitignore",
                                                             "LICENSE", "README.md"))]
        assert not stray, (p.name, stray[:10])
