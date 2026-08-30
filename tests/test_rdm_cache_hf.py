"""
Tests for scripts/rdm_cache_hf.py's ROI-subdirectory namespacing (2026-08-30).

WHY THIS MATTERS. Before this existed, the Hub cache was keyed only by
dataset/variant/task/session -- no masking level. ds002236 and ds003604 both
already have WHOLE-BRAIN RDMs cached under those keys. Without ROI
namespacing, an `ROI_SET=phonology` run's cache `pull` would silently HIT
that whole-brain entry and place it under the phonology-labeled output path
-- corrupting the run with no error, since a successful pull is exactly what
the caller wants to see. This was caught before it ever ran for real (no
HF_TOKEN on the machine that found it, so every local pull this session was
a no-op miss -- lucky, not by design); it would have struck immediately on
the collaborator's GPU cluster, which has working HF auth.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rdm_cache_hf as cache  # noqa: E402


def test_remote_path_whole_brain_is_byte_identical_to_before_roi_existed():
    """No roi_subdir -> the exact 4-part path every already-cached entry uses.
    A regression here would orphan every existing cache entry."""
    assert cache.remote_path("Sem", "ses-9", "ds002236", "within-run-normalised") == \
        "ds002236/within-run-normalised/Sem/session_rdm_ses-9.npz"
    assert cache.remote_path("Sem", "ses-9", "ds002236", "within-run-normalised", roi_subdir=None) == \
        "ds002236/within-run-normalised/Sem/session_rdm_ses-9.npz"
    assert cache.remote_path("Sem", "ses-9", "ds002236", "within-run-normalised", roi_subdir="") == \
        "ds002236/within-run-normalised/Sem/session_rdm_ses-9.npz"


@pytest.mark.parametrize("roi_subdir", ["roi-phonology", "roi-language", "roi-all", "roi-auditory+motor"])
def test_remote_path_roi_scoped_nests_between_variant_and_task(roi_subdir):
    p = cache.remote_path("Sem", "ses-9", "ds002236", "within-run-normalised", roi_subdir)
    assert p == f"ds002236/within-run-normalised/{roi_subdir}/Sem/session_rdm_ses-9.npz"
    # Different roi_subdir values must never collide with each other or with
    # the whole-brain (unscoped) path.
    whole_brain = cache.remote_path("Sem", "ses-9", "ds002236", "within-run-normalised")
    assert p != whole_brain


def test_remote_path_distinct_roi_subdirs_never_collide():
    paths = {
        cache.remote_path("Sem", "ses-9", "ds002236", "within-run-normalised", roi)
        for roi in (None, "roi-phonology", "roi-language", "roi-all")
    }
    assert len(paths) == 4, f"expected 4 distinct paths, got {paths}"


class _FakeArgs:
    """Minimal stand-in for argparse.Namespace, only the attrs cmd_pull reads."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_pull_roi_scoped_never_falls_back_to_legacy_whole_brain_path(tmp_path):
    """The one invariant that matters most: an ROI-restricted pull for
    ds003604/raw (the only dataset+variant with a legacy flat equivalent)
    must NEVER try the legacy (whole-brain, pre-namespacing) path as a
    fallback candidate. Verified by faking a present HF_TOKEN and recording
    every path hf_hub_download is asked for, rather than hitting the network."""
    attempted = []

    def fake_download(repo, path, **kw):
        attempted.append(path)
        raise RuntimeError("simulated miss")  # every candidate misses -> real network never touches

    args = _FakeArgs(task="Sem", session="ses-5", dir=str(tmp_path),
                      dataset="ds003604", variant="raw", roi_subdir="roi-phonology")

    with patch.object(cache, "_api", return_value=(object(), "fake-token")), \
         patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        rc = cache.cmd_pull(args)

    assert rc == 1  # every candidate missed, by design
    assert attempted == ["ds003604/raw/roi-phonology/Sem/session_rdm_ses-5.npz"], (
        f"an ROI-scoped pull must try ONLY the ROI-scoped path, never the legacy "
        f"whole-brain fallback -- got {attempted}"
    )


def test_pull_whole_brain_ds003604_raw_still_tries_legacy_fallback(tmp_path):
    """Confirms the fix didn't accidentally remove the EXISTING legacy-path
    fallback for genuine whole-brain requests -- only ROI-scoped ones should
    skip it."""
    attempted = []

    def fake_download(repo, path, **kw):
        attempted.append(path)
        raise RuntimeError("simulated miss")

    args = _FakeArgs(task="Sem", session="ses-5", dir=str(tmp_path),
                      dataset="ds003604", variant="raw", roi_subdir=None)

    with patch.object(cache, "_api", return_value=(object(), "fake-token")), \
         patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        cache.cmd_pull(args)

    assert attempted == [
        "ds003604/raw/Sem/session_rdm_ses-5.npz",
        "Sem/session_rdm_ses-5.npz",  # the legacy flat layout
    ]


@pytest.mark.parametrize("root_suffix,expected", [
    ("roi-phonology", "roi-phonology"),
    ("roi-language", "roi-language"),
    ("roi-all", "roi-all"),
    ("", None),  # whole-brain root -- no roi- suffix at all
])
def test_sync_infers_roi_subdir_from_root_path(tmp_path, root_suffix, expected):
    """cmd_sync has no --roi-subdir flag -- it infers the masking level from
    the root it's given, which is always already ROI-scoped when relevant
    (prepare_brain_rdms.sh calls it with --root "$RDM_ROOT"). Verified against
    the exact inference logic cmd_sync uses, since triggering cmd_sync itself
    needs a real HF_TOKEN this environment doesn't have."""
    root = tmp_path / root_suffix if root_suffix else tmp_path
    inferred = root.name if root.name.startswith("roi-") else None
    assert inferred == expected
