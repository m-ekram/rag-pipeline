"""Corpus reproducibility: the pinned ref and the lock file.

An unpinned corpus drifts with upstream, and every measured retrieval number
quietly stops describing the corpus it was measured on. The pin prevents that;
the lock is how drift gets *detected* rather than silently absorbed.
"""

import json
import subprocess

import pytest

import prepare_fastapi_docs as prep


@pytest.fixture
def corpus(tmp_path):
    """A tiny corpus directory with a matching lock file."""
    out = tmp_path / "fastapi"
    out.mkdir()
    (out / "alpha.md").write_text("# Alpha\n\nAlpha content.\n", encoding="utf-8")
    (out / "beta.md").write_text("# Beta\n\nBeta content.\n", encoding="utf-8")
    prep.write_lock(out, prep.build_lock(out, "0.141.1", "a" * 40))
    return out


# -- the pin -------------------------------------------------------------------


def test_the_default_ref_is_a_version_not_a_branch():
    """Pinning to main is what made the previous corpus unreproducible."""
    assert prep.DEFAULT_REF not in {"main", "master", "HEAD"}
    assert prep.DEFAULT_REF[0].isdigit()


def test_clone_passes_the_ref_to_git(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    prep.clone(tmp_path / "repo", "0.140.0")
    clone_cmd = calls[0]
    assert "--branch" in clone_cmd
    assert clone_cmd[clone_cmd.index("--branch") + 1] == "0.140.0"


def test_clone_stays_shallow_and_sparse(tmp_path, monkeypatch):
    """A full clone of FastAPI is orders of magnitude more than the docs need."""
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    prep.clone(tmp_path / "repo", "0.141.1")
    assert "--depth" in calls[0] and "--filter=blob:none" in calls[0]
    assert calls[1][:3] == ["git", "sparse-checkout", "set"]


def test_an_existing_checkout_at_the_right_ref_is_reused(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / prep.DOCS_SUBDIR).mkdir(parents=True)
    monkeypatch.setattr(prep, "checked_out_ref", lambda d: "0.141.1")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("should not re-clone"))
    prep.clone(repo, "0.141.1")


def test_an_existing_checkout_at_the_wrong_ref_is_replaced(tmp_path, monkeypatch):
    """Otherwise --ref would silently do nothing on a machine that already cloned."""
    repo = tmp_path / "repo"
    (repo / prep.DOCS_SUBDIR).mkdir(parents=True)
    monkeypatch.setattr(prep, "checked_out_ref", lambda d: "0.100.0")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    prep.clone(repo, "0.141.1")
    assert not repo.exists() or calls, "expected a re-clone"
    assert "--branch" in calls[0]


def test_checked_out_ref_is_none_when_git_fails(tmp_path):
    assert prep.checked_out_ref(tmp_path) is None


# -- the lock ------------------------------------------------------------------


def test_the_lock_records_provenance(corpus):
    lock = prep.read_lock(corpus)
    assert lock["ref"] == "0.141.1"
    assert lock["commit"] == "a" * 40
    assert lock["source"] == prep.REPO_URL
    assert lock["file_count"] == 2


def test_the_lock_digests_every_file(corpus):
    lock = prep.read_lock(corpus)
    assert set(lock["files"]) == {"alpha.md", "beta.md"}
    assert all(len(d) == 64 for d in lock["files"].values())


def test_the_lock_is_stable_for_unchanged_content(corpus):
    first = json.dumps(prep.build_lock(corpus, "0.141.1", "a" * 40), sort_keys=True)
    second = json.dumps(prep.build_lock(corpus, "0.141.1", "a" * 40), sort_keys=True)
    assert first == second


def test_reading_a_missing_lock_returns_none(tmp_path):
    assert prep.read_lock(tmp_path) is None


# -- verify --------------------------------------------------------------------


def test_verify_passes_an_untouched_corpus(corpus, capsys):
    assert prep.verify(corpus) == 0
    assert "matches" in capsys.readouterr().out


def test_verify_detects_edited_content(corpus, capsys):
    (corpus / "alpha.md").write_text("# Alpha\n\nSomething else entirely.\n", encoding="utf-8")
    assert prep.verify(corpus) == 1
    assert "CHANGED  alpha.md" in capsys.readouterr().out


def test_verify_detects_a_deleted_file(corpus, capsys):
    (corpus / "beta.md").unlink()
    assert prep.verify(corpus) == 1
    assert "MISSING  beta.md" in capsys.readouterr().out


def test_verify_detects_an_unexpected_file(corpus, capsys):
    (corpus / "gamma.md").write_text("# Gamma\n", encoding="utf-8")
    assert prep.verify(corpus) == 1
    assert "EXTRA    gamma.md" in capsys.readouterr().out


def test_verify_counts_every_difference(corpus):
    (corpus / "beta.md").unlink()
    (corpus / "gamma.md").write_text("# Gamma\n", encoding="utf-8")
    (corpus / "alpha.md").write_text("changed\n", encoding="utf-8")
    assert prep.verify(corpus) == 3


def test_verify_without_a_lock_is_a_problem(tmp_path, capsys):
    assert prep.verify(tmp_path) == 1
    assert "No CORPUS.lock.json" in capsys.readouterr().out


# -- the content transforms the corpus quality depends on ----------------------


def test_include_directives_are_inlined_as_code(tmp_path):
    """The code examples are not in the markdown; without this every how-to
    chunk loses the code that answers it."""
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "t001.py").write_text("app = FastAPI()\n", encoding="utf-8")
    counters = {"inlined": 0, "missing": 0}
    out = prep.inline_includes("{* ../../docs_src/t001.py *}", tmp_path, counters)
    assert "```python" in out and "app = FastAPI()" in out
    assert counters["inlined"] == 1


def test_an_unresolvable_include_is_dropped_and_counted(tmp_path):
    counters = {"inlined": 0, "missing": 0}
    out = prep.inline_includes("{* ../../docs_src/nope.py *}", tmp_path, counters)
    assert out.strip() == ""
    assert counters["missing"] == 1


def test_include_modifiers_are_ignored_but_titles_become_captions(tmp_path):
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "t.py").write_text("x = 1\n", encoding="utf-8")
    counters = {"inlined": 0, "missing": 0}
    out = prep.inline_includes(
        '{* ../../docs_src/t.py hl[2] title["app/main.py"] *}', tmp_path, counters
    )
    assert "app/main.py" in out and "x = 1" in out


def test_admonitions_keep_their_text_and_become_bold_labels():
    out = prep.flatten_admonitions("/// tip\n\nUse a virtualenv.\n\n///")
    assert "**Tip:**" in out
    assert "Use a virtualenv." in out


def test_nested_admonitions_are_flattened_too():
    """Nested blocks use one more slash, so the pattern must accept 3 or more."""
    out = prep.flatten_admonitions("//// note | Heads up\n\nInner text.\n\n////")
    assert "**Note: Heads up**" in out
    assert "Inner text." in out


def test_heading_anchors_are_stripped(tmp_path):
    out = prep.clean(
        "## Query Parameters { #query-parameters }\n",
        tmp_path,
        {"inlined": 0, "missing": 0},
    )
    assert "{ #query-parameters }" not in out
    assert "## Query Parameters" in out


def test_clean_collapses_the_blank_lines_flattening_leaves_behind(tmp_path):
    out = prep.clean("A\n\n\n\n\nB\n", tmp_path, {"inlined": 0, "missing": 0})
    assert "A\n\nB" in out


def test_the_changelog_is_excluded_from_the_corpus():
    """At 694 KB it would be ~40% of the corpus and is almost all PR chatter."""
    assert "release-notes.md" in prep.SKIP_FILES
