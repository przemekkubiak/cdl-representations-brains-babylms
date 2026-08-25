"""Resolve neuro datasets from configs/neuro_datasets.yaml.

Before this module every download path in the repo was hardcoded to ds003604 --
`scripts/batch_download_bold.py` built the annex, S3 and snapshot URLs from the
literal string, with no argument to change it. That made the cross-dataset arm
impossible to run honestly: pointing DATASET= at another accession would have
re-downloaded ds003604 into a directory named for a different study. See
PICKUP.md, "What was deliberately NOT run".

This module is the single place that knows how to turn an accession into URLs,
paths and a contrast spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "neuro_datasets.yaml"

# A dataset may be listed for planning purposes long before it can be run.
READY = "ready"
NEEDS_INSPECTION = "needs_inspection"
UNRESOLVED = "unresolved"


class UnresolvedDatasetError(RuntimeError):
    """Raised when a dataset is referenced but cannot be downloaded.

    Deliberately fatal rather than falling back to a default accession: a silent
    fallback would download ds003604 and label it as another study.
    """


@dataclass
class DatasetSpec:
    key: str
    accession: str | None
    snapshot: str | None
    name: str
    tasks: list[str]
    sessions: list[str]
    phenomena: dict[str, list[str]]
    design: dict[str, Any]
    ages: dict[str, Any]
    stimuli: dict[str, Any]
    contrast_spec: str | None
    status: str
    blocker: str | None = None
    _defaults: dict[str, Any] = field(default_factory=dict, repr=False)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- guards ------------------------------------------------------------

    def require_downloadable(self) -> str:
        """Return the accession, or explain why this dataset cannot be fetched."""
        if not self.accession:
            raise UnresolvedDatasetError(
                f"dataset '{self.key}' has no resolved OpenNeuro accession"
                + (f"\n  {self.blocker.strip()}" if self.blocker else "")
            )
        return self.accession

    @property
    def run_stimulus(self) -> str:
        """'nested' | 'crossed' | 'unknown'.

        'nested' means each stimulus appears in exactly one run, so run identity
        is confounded with stimulus identity and RDMs built without within-run
        normalisation measure acquisition structure, not language. ds003604 is
        nested; that is the confound in hf_results_staging/README.md.
        """
        return (self.design or {}).get("run_stimulus", "unknown")

    def run_stimulus_for_task(self, task: str) -> str:
        """Per-task run/stimulus structure, which can differ within a dataset.

        ds006239 is mixed: LocalSem is crossed while ReadPhon/ReadMean are
        nested. Pooling those under one dataset-level verdict would either
        discard the one clean cell we have or wrongly trust the confounded ones.
        """
        per_task = (self.design or {}).get("per_task_run_stimulus") or {}
        return per_task.get(task, self.run_stimulus)

    @property
    def needs_within_run_norm(self) -> bool:
        return self.run_stimulus != "crossed"

    def needs_within_run_norm_for_task(self, task: str) -> bool:
        return self.run_stimulus_for_task(task) != "crossed"

    def clean_tasks(self) -> list[str]:
        """Tasks whose run/stimulus structure is crossed -- i.e. not confounded."""
        return [t for t in self.tasks if self.run_stimulus_for_task(t) == "crossed"]

    # -- URL construction --------------------------------------------------

    def git_url(self) -> str:
        acc = self.require_downloadable()
        return self._defaults["git_url"].format(accession=acc)

    def annex_bases(self) -> list[str]:
        acc = self.require_downloadable()
        tmpl = self._defaults["annex_base"]
        return [tmpl.format(accession=acc, branch=b) for b in self._defaults["branches"]]

    def s3_base(self) -> str:
        acc = self.require_downloadable()
        return self._defaults["s3_base"].format(accession=acc)

    def snapshot_base(self) -> str | None:
        acc = self.require_downloadable()
        if not self.snapshot:
            return None
        return self._defaults["snapshot_base"].format(accession=acc, snapshot=self.snapshot)

    def candidate_urls(self, rel_path: str, annex_target: str | None = None) -> list[str]:
        """Every URL worth trying for one file, most-likely first.

        `rel_path` is the file's path relative to the dataset root.
        `annex_target` is the git-annex symlink target, if the file is a symlink.
        """
        urls: list[str] = []
        if annex_target and "annex/objects" in annex_target:
            annex_path = annex_target.split("annex/objects/")[1]
            for base in self.annex_bases():
                urls.append(f"{base}/.git/annex/objects/{annex_path}")
        urls.append(f"{self.s3_base()}/{rel_path}")
        snap = self.snapshot_base()
        if snap:
            urls.append(f"{snap}/{rel_path}")

        seen: set[str] = set()
        deduped = []
        for u in urls:
            if u not in seen:
                deduped.append(u)
                seen.add(u)
        return deduped

    # -- paths -------------------------------------------------------------

    def data_dir(self, root: str | Path = "data/brain") -> Path:
        return Path(root) / self.require_downloadable()

    def rdm_dir(self, root: str | Path = "data/processed/fmri") -> Path:
        return Path(root) / self.require_downloadable()

    def stimulus_dir(self, root: str | Path = "data/brain") -> Path | None:
        local = (self.stimuli or {}).get("local")
        if not local:
            return None
        return self.data_dir(root) / local


def load_registry(path: str | Path | None = None) -> dict[str, DatasetSpec]:
    p = Path(path) if path else _REGISTRY_PATH
    with open(p) as fh:
        raw = yaml.safe_load(fh)

    defaults = raw.get("defaults", {})
    out: dict[str, DatasetSpec] = {}
    for key, d in (raw.get("datasets") or {}).items():
        out[key] = DatasetSpec(
            key=key,
            accession=d.get("accession"),
            snapshot=d.get("snapshot"),
            name=d.get("name", key),
            tasks=list(d.get("tasks") or []),
            sessions=list(d.get("sessions") or []),
            phenomena=dict(d.get("phenomena") or {}),
            design=dict(d.get("design") or {}),
            ages=dict(d.get("ages") or {}),
            stimuli=dict(d.get("stimuli") or {}),
            contrast_spec=d.get("contrast_spec"),
            status=d.get("status", UNRESOLVED),
            blocker=d.get("blocker"),
            _defaults=defaults,
            raw=d,
        )
    return out


def get_dataset(key: str, path: str | Path | None = None) -> DatasetSpec:
    reg = load_registry(path)
    # Accept either the registry key or the raw accession.
    if key in reg:
        return reg[key]
    for spec in reg.values():
        if spec.accession == key:
            return spec
    raise KeyError(
        f"unknown dataset '{key}'. Known: {', '.join(sorted(reg))}. "
        "Add it to configs/neuro_datasets.yaml rather than hardcoding a URL."
    )


def list_datasets(path: str | Path | None = None) -> list[DatasetSpec]:
    return list(load_registry(path).values())


if __name__ == "__main__":
    for spec in list_datasets():
        acc = spec.accession or "(unresolved)"
        print(
            f"{spec.key:12s} {acc:10s} {spec.status:16s} "
            f"run/stim={spec.run_stimulus:8s} tasks={','.join(spec.tasks) or '-'}"
        )
        if spec.blocker:
            print(f"             blocker: {' '.join(spec.blocker.split())[:100]}...")
