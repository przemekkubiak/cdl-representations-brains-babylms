"""
Self-test for the multi-dataset preprocessing infrastructure (2026-08-26):
ROI groupings, the age-group taxonomy, dataset-aware stimulus identity, the
age-based pattern regrouping script, and FMRIPreprocessor's task/session
resolution -- across all four registered datasets.

RUN THIS FIRST, same as tests/test_masking_pipeline.py. Most of it needs no
GPU and no BOLD download -- only a metadata-only checkout per dataset
(scripts/inspect_dataset.py --bootstrap), which is what most tests here read
directly rather than mocking. Tests that need a checkout skip cleanly if it
isn't present, so this still runs (with reduced coverage) in an environment
that hasn't bootstrapped every dataset.

    pytest tests/test_multi_dataset_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import roi_atlas
from src.datasets import age_groups
from src.datasets.stim_identity import classify_trials
from src.datasets import get_dataset

DATASETS = ["ds003604", "ds001894", "ds006239", "ds002236"]


def _checkout_present(dataset: str) -> bool:
    return (get_dataset(dataset).data_dir() / "participants.tsv").exists()


def _require_checkout(dataset: str):
    if not _checkout_present(dataset):
        pytest.skip(
            f"no metadata checkout for {dataset} -- run "
            f"`python scripts/inspect_dataset.py --dataset {dataset} --bootstrap` first"
        )


# ------------------------------------------------------------- roi_atlas --
def test_phonology_is_union_of_auditory_and_motor():
    assert set(roi_atlas.ROI_SETS["phonology"]) == (
        set(roi_atlas.ROI_SETS["auditory"]) | set(roi_atlas.ROI_SETS["motor"])
    )


def test_all_is_union_of_every_other_set():
    others = set()
    for name in ("language", "auditory", "motor"):
        others |= set(roi_atlas.ROI_SETS[name])
    assert set(roi_atlas.ROI_SETS["all"]) == others


def test_three_analysis_groupings_are_all_available():
    """The three groupings requested for cross-dataset ROI analysis."""
    for name in ("language", "phonology", "all"):
        assert name in roi_atlas.available_roi_sets()


# ---------------------------------------------------------- age_groups.py --
def test_bin_of_covers_the_full_age_range_with_no_gaps():
    bins = age_groups.load_bins()
    # every age from 4 to 18 in 0.1-year steps must land in exactly one bin
    age = 4.0
    while age < 18.0:
        age_groups.bin_of(age, bins)  # raises if it doesn't land anywhere
        age += 0.1


def test_bin_of_boundaries_are_half_open_and_consistent():
    assert age_groups.bin_of(5.99) == "5"
    assert age_groups.bin_of(6.0) == "7"
    assert age_groups.bin_of(11.99) == "11"
    assert age_groups.bin_of(12.0) == "11+"


def test_bin_of_rejects_nan():
    with pytest.raises(ValueError):
        age_groups.bin_of(float("nan"))


def test_representative_age_reproduces_ds003604_nominal_ages_exactly():
    """The whole point of centering bins on 5/7/9: generalizing
    SESSION_TO_AGE must not change ds003604's existing numbers."""
    rep = age_groups.representative_age()
    assert rep["5"] == 5.0
    assert rep["7"] == 7.0
    assert rep["9"] == 9.0


@pytest.mark.parametrize("dataset", DATASETS)
def test_per_subject_ages_has_an_extractor_and_reads_real_data(dataset):
    _require_checkout(dataset)
    ages = age_groups.per_subject_ages(dataset)
    assert len(ages) > 0
    # every extracted age must fall somewhere in the taxonomy without raising
    for subj, sessions in ages.items():
        for ses, age in sessions.items():
            assert isinstance(age, float)
            assert 0 < age < 25, f"{dataset} {subj}/{ses}: implausible age {age}"
            age_groups.bin_of(age)  # must not raise


def test_per_subject_ages_unknown_dataset_raises():
    with pytest.raises(KeyError):
        age_groups.per_subject_ages("ds_not_registered")


# ------------------------------------------------------- stim_identity.py --
def _first_events_tsv(dataset: str, task_glob: str):
    d = get_dataset(dataset).data_dir()
    hits = sorted(d.glob(f"sub-*/**/*task-{task_glob}*_events.tsv"))
    return hits[0] if hits else None


@pytest.mark.parametrize("dataset,phenomenon,task_glob", [
    ("ds002236", "Phon", "AudRhyme"),
    ("ds002236", "Sem", "AudSem"),
    ("ds001894", "Phon", "AAWord"),
    ("ds006239", "Phon", "ReadPhon"),
    ("ds006239", "Sem", "ReadMean"),
])
def test_classify_trials_against_real_events(dataset, phenomenon, task_glob):
    _require_checkout(dataset)
    f = _first_events_tsv(dataset, task_glob)
    if f is None:
        pytest.skip(f"no {task_glob} events.tsv found under {dataset}'s checkout")
    import csv
    with open(f, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    trials = classify_trials(rows, dataset, phenomenon)
    assert len(trials) > 0
    assert all(t.condition in ("positive", "negative") for t in trials)
    # every trial's text should be two non-empty words (all four stim_pair_
    # filename datasets present a pair)
    for t in trials[:5]:
        words = t.text.split()
        assert len(words) == 2 and all(words), f"bad reconstructed text: {t.text!r}"


def test_classify_trials_ds003604_is_unfiltered_and_condition_is_none():
    """github_tsv path must reproduce the pre-2026-08-26 behaviour exactly:
    every row becomes a trial, condition is never set here (filtering, when
    it happens at all, stays downstream in session_based_rsa.py)."""
    _require_checkout("ds003604")
    f = _first_events_tsv("ds003604", "Sem")
    if f is None:
        pytest.skip("no ds003604 Sem events.tsv found")
    import csv
    with open(f, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    trials = classify_trials(rows, "ds003604", "Sem")
    assert len(trials) == len(rows)
    assert all(t.condition is None for t in trials)


def test_classify_trials_unknown_phenomenon_raises():
    from src.contrast_spec import ContrastSpecUnavailable
    with pytest.raises(ContrastSpecUnavailable):
        classify_trials([{"trial_type": "1", "stim1_file": "a", "stim2_file": "b"}],
                         "ds002236", "NotAPhenomenon")


# ------------------------------------------------- FMRIPreprocessor wiring --
@pytest.mark.parametrize("dataset,subject,phenomenon,expect_sessionless", [
    ("ds003604", None, "Sem", False),
    ("ds001894", None, "Phon", False),
    ("ds006239", None, "Phon", False),
    ("ds002236", None, "Phon", True),
])
def test_find_task_runs_resolves_phenomenon_and_sessions(dataset, subject, phenomenon, expect_sessionless):
    _require_checkout(dataset)
    from src.preprocessing.fmri_preprocessing import FMRIPreprocessor, SESSIONLESS_LABEL

    data_dir = get_dataset(dataset).data_dir()
    subjects = sorted(p.name for p in data_dir.glob("sub-*") if p.is_dir())
    if not subjects:
        pytest.skip(f"no subjects in {dataset} checkout")
    subject = subjects[0]

    fp = FMRIPreprocessor(data_dir=str(data_dir), subject_id=subject, task=phenomenon, dataset=dataset)
    runs = fp.find_task_runs()
    if not runs:
        pytest.skip(f"{dataset}/{subject} has no {phenomenon} runs (try a different subject)")

    sessions = {r["session"] for r in runs}
    if expect_sessionless:
        assert sessions == {SESSIONLESS_LABEL}
    else:
        assert SESSIONLESS_LABEL not in sessions

    # every run_info['real_task'] must be a task this dataset actually declares
    # for this phenomenon
    declared = set(get_dataset(dataset).phenomena.get(phenomenon, [phenomenon]))
    assert {r["real_task"] for r in runs} <= declared


def test_find_task_runs_unresolved_phenomenon_falls_back_to_literal_task_name():
    """Manual/ad-hoc use (no phenomenon mapping declared) must still work by
    treating `task` as a literal BIDS task name, not raise."""
    from src.preprocessing.fmri_preprocessing import FMRIPreprocessor

    _require_checkout("ds003604")
    data_dir = get_dataset("ds003604").data_dir()
    subjects = sorted(p.name for p in data_dir.glob("sub-*") if p.is_dir())
    if not subjects:
        pytest.skip("no ds003604 subjects in checkout")
    fp = FMRIPreprocessor(data_dir=str(data_dir), subject_id=subjects[0], task="Gram", dataset="ds003604")
    assert fp._resolve_real_tasks() == ["Gram"]


# --------------------------------------------------- regroup_patterns_by_age
def test_regroup_relabels_by_real_age_bin(tmp_path):
    _require_checkout("ds002236")
    import numpy as np
    from scripts.regroup_patterns_by_age import regroup

    ages = age_groups.per_subject_ages("ds002236")
    subjects = list(ages.keys())[:4]
    if not subjects:
        pytest.skip("no ds002236 age data available")
    for s in subjects:
        np.savez(tmp_path / f"{s}_ses-all_run-01_patterns.npz", dummy=np.zeros(2))

    n = regroup("ds002236", tmp_path, mode="symlink")
    assert n == len(subjects)
    for s in subjects:
        expected_bin = age_groups.bin_of(ages[s][age_groups.SINGLE_SESSION])
        relabeled = tmp_path / f"{s}_ses-{expected_bin}_run-01_patterns.npz"
        assert relabeled.exists(), f"expected {relabeled} to exist"


def test_regroup_skips_ds003604_by_design(tmp_path):
    from scripts.regroup_patterns_by_age import regroup
    (tmp_path / "sub-5002_ses-5_run-01_patterns.npz").touch()
    n = regroup("ds003604", tmp_path, mode="symlink")
    assert n == 0
    assert sorted(p.name for p in tmp_path.glob("*")) == ["sub-5002_ses-5_run-01_patterns.npz"]


def test_regroup_handles_subjects_with_no_age_gracefully(tmp_path, capsys):
    import numpy as np
    from scripts.regroup_patterns_by_age import regroup
    # a subject id that will not appear in ds002236's participants.tsv
    np.savez(tmp_path / "sub-99999999_ses-all_run-01_patterns.npz", dummy=np.zeros(2))
    n = regroup("ds002236", tmp_path, mode="symlink")
    assert n == 0  # nothing relabeled -- must not crash
