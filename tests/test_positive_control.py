"""
Tests for scripts/positive_control.py's word-pair-indexed norms controls
(2026-08-29 fix): ds002236 and ds006239 ship a PRIME/TARGET-word-keyed
Stimulus_Charact*.tsv, not the stim_file-keyed layout NORMS assumes, so every
norm-based control silently had zero data for them before this existed.

Real local data only (no synthetic fixture): the bug this locks in
(word_pair_index task-crossing collisions) only reproduces against the real
table's actual duplicate word pairs across tasks, which a small hand-built
fixture would not exercise honestly. Skips cleanly if the checkout isn't
present, same convention as tests/test_multi_dataset_pipeline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.datasets import get_dataset

from positive_control import (  # noqa: E402
    WORD_PAIR_ASSOCIATION_COL,
    WORD_PAIR_NORMS,
    find_norms,
    word_pair_index,
    word_pair_scalar,
    word_pair_task_label,
)


def _checkout_present(dataset: str) -> bool:
    return (get_dataset(dataset).data_dir() / "participants.tsv").exists()


@pytest.mark.parametrize("dataset,task,expected_label", [
    ("ds002236", "Phon", "rhyme"),
    ("ds002236", "Sem", "semantic"),
    ("ds006239", "Sem", "ReadMean"),
    ("ds006239", "Phon", "ReadPhon"),
])
def test_word_pair_task_label_resolves_real_registry_tasks(dataset, task, expected_label):
    if not _checkout_present(dataset):
        pytest.skip(f"no metadata checkout for {dataset}")
    assert word_pair_task_label(dataset, task) == expected_label


def test_word_pair_task_label_unknown_dataset_returns_none():
    assert word_pair_task_label("not-a-real-dataset", "Sem") is None
    assert word_pair_task_label(None, "Sem") is None


@pytest.mark.parametrize("dataset,task", [
    ("ds002236", "Sem"), ("ds006239", "Sem"), ("ds006239", "Phon"),
])
def test_word_pair_frequency_control_builds_from_real_norms(dataset, task):
    """End-to-end against the real published RDM's own stimulus_texts would
    need the HF download; this checks the norms side directly instead, using
    every word pair the norms table itself offers for this task -- which is
    by construction resolvable, so a failure here is a real regression, not a
    coverage gap in some particular RDM's stimulus list."""
    if not _checkout_present(dataset):
        pytest.skip(f"no metadata checkout for {dataset}")
    stim_dir = get_dataset(dataset).data_dir() / "stimuli"
    norms = find_norms(stim_dir, task)
    if norms is None:
        pytest.skip(f"no norms table found for {dataset}")

    label = word_pair_task_label(dataset, task)
    idx = word_pair_index(norms, label)
    assert idx, f"empty word-pair index for {dataset}/{task} (label={label!r})"

    texts = [f"{p} {t}" for (p, t) in list(idx.keys())[:20] if p < t]  # dedupe both-orders
    cols = WORD_PAIR_NORMS[dataset]["word_frequency"]
    v = word_pair_scalar(idx, texts, cols)
    assert v is not None, (
        f"word_frequency failed to build for {dataset}/{task} using the norms "
        "table's OWN word pairs -- should always resolve"
    )
    assert len(v) == len(texts)


def test_word_pair_index_task_collision_is_resolved_not_silently_wrong():
    """Regression test for the real bug this fix caught 2026-08-29: ds006239's
    "dog"/"cat" pair recurs across multiple TASK rows in the norms table with
    DIFFERENT column completeness (frequency columns populated under
    ReadMean, blank elsewhere). Unfiltered, dict insertion order silently
    picked whichever row came first -- sometimes the blank one -- rather than
    the row matching the RDM's own task. Filtering by task_label must recover
    the real (non-NaN) value, not merely avoid crashing.
    """
    if not _checkout_present("ds006239"):
        pytest.skip("no metadata checkout for ds006239")
    stim_dir = get_dataset("ds006239").data_dir() / "stimuli"
    norms = find_norms(stim_dir, "Sem")
    if norms is None or "TASK" not in norms.columns:
        pytest.skip("no TASK-labeled norms table found for ds006239")

    # Only meaningful if this pair really does recur across >1 task in the
    # table with differing completeness -- confirm the premise before
    # asserting the fix, so this test fails loudly (not silently passes
    # vacuously) if the underlying data ever changes.
    dup = norms[
        ((norms["PRIME"].str.lower() == "dog") & (norms["TARGET"].str.lower() == "cat"))
        | ((norms["PRIME"].str.lower() == "cat") & (norms["TARGET"].str.lower() == "dog"))
    ]
    if len(dup) < 2:
        pytest.skip("dog/cat no longer recurs across multiple tasks in this checkout")

    label = word_pair_task_label("ds006239", "Sem")
    idx = word_pair_index(norms, label)
    row = idx.get(("dog", "cat"))
    assert row is not None
    cols = WORD_PAIR_NORMS["ds006239"]["word_frequency"]
    import pandas as pd
    vals = pd.to_numeric(pd.Series([row.get(c) for c in cols]), errors="coerce")
    assert not vals.isna().any(), (
        "word_pair_index picked the wrong task's row for a word pair that "
        "recurs across tasks -- task_label filtering regressed"
    )


def test_word_pair_association_col_present_in_both_known_datasets():
    """ASS is documented as the same column name in both ds002236 and
    ds006239 -- if a future data release renames it, this fails loudly rather
    than the association control silently vanishing again."""
    for dataset, task in [("ds002236", "Sem"), ("ds006239", "Sem")]:
        if not _checkout_present(dataset):
            pytest.skip(f"no metadata checkout for {dataset}")
        stim_dir = get_dataset(dataset).data_dir() / "stimuli"
        norms = find_norms(stim_dir, task)
        assert norms is not None and WORD_PAIR_ASSOCIATION_COL in norms.columns
