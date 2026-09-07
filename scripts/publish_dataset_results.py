#!/usr/bin/env python
"""Publish one neuro dataset's results to its OWN HuggingFace dataset repo.

Each neuro dataset gets a separate repo rather than a folder inside the shared
one, so that a dataset whose positive control failed cannot have its numbers
quoted out of a directory listing next to one that passed.

The README is generated from what actually ran -- the ceilings, the control
verdict, the alignment tables -- and leads with the gate. If the control found no
recoverable stimulus signal, the README says the alignment numbers are
uninterpretable BEFORE it shows any of them.

Uploads are additive; nothing is ever deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ORG = "BrainAlign"

DATASET_BLURB = {
    "ds003604": (
        "The flagship dataset of this project — auditory sentence/word-pair "
        "listening in children aged 5, 7 and 9, four phenomena (semantic, "
        "phonological, grammatical, plausibility).",
        "https://openneuro.org/datasets/ds003604",
        "https://openneuro.org/datasets/ds003604",
        "The only dataset with a longitudinal DEVELOPMENTAL axis across three "
        "discrete ages rather than a continuous one (ses-5/ses-7/ses-9). Each "
        "stimulus is presented in exactly one scanner run, which is the source "
        "of the run confound the within-run normalisation in this pipeline "
        "corrects (see the HF repo README for BrainAlign/ds003604-session-rdms "
        "for the measured before/after). Every other dataset here was added to "
        "generalise past this one, not to replace it — treat its numbers as "
        "the reference point the others are compared against, not as one "
        "dataset among four.",
    ),
    "ds001894": (
        "Lytle et al. 2019 — longitudinal word-level phonological processing in "
        "children scanned twice, at roughly 10 and 12 years old.",
        "https://www.nature.com/articles/s41597-019-0338-5",
        "https://openneuro.org/datasets/ds001894/versions/1.4.2",
        "The only longitudinal dataset here: the same children at two timepoints "
        "(ses-T1, ses-T2), which is the closest real analogue to a language "
        "model's checkpoint trajectory. Per-subject age at scan is available. "
        "Trial types cross orthographic with phonological similarity "
        "(O+P+/O+P-/O-P+/O-P-), so Phon and Orth contrasts are decorrelated by "
        "design. ses-T2 has only the VV tasks.",
    ),
    "ds006239": (
        "Wang et al. 2025 — word-level phonological and semantic reading tasks in "
        "children and adolescents aged 10–17.",
        "https://www.sciencedirect.com/science/article/pii/S2352340925009692",
        "https://openneuro.org/datasets/ds006239/versions/1.0.5",
        "Contains **LocalSem**, the only genuinely run/stimulus-CROSSED language "
        "cell across all four datasets in this project: its stimuli recur across "
        "runs, so run identity and stimulus identity are separable and the "
        "scanner-run confound that invalidated the first ds003604 analysis "
        "cannot arise. Per-subject age is NOT recoverable from the release — "
        "participants.tsv has birthdate but no scan date and there are no "
        "*_scans.tsv files — so this dataset is cohort-level only and cannot "
        "carry the developmental axis as published.",
    ),
    "ds002236": (
        "Lytle et al. 2020 — orthographic, phonological and semantic word "
        "processing in school-aged children (8.7–15.5), auditory and visual.",
        "https://pubmed.ncbi.nlm.nih.gov/31956678/",
        "https://openneuro.org/datasets/ds002236/versions/1.0.1",
        "The accession is not stated in the data article; it was resolved to "
        "ds002236 by matching OpenNeuro's own dataset name "
        "(\"Cross-Sectional Multidomain Lexical Processing\") AND the "
        "per-subject age range in participants.tsv (8.67–15.5) against the "
        "range the article reports. Best developmental axis of the four "
        "datasets: explicit per-subject age at scan, continuous rather than "
        "binned. Six tasks crossing modality (auditory/visual) with judgement "
        "(rhyme/spelling/semantic) — a modality control no other dataset here "
        "provides. A third of trials are coded null (Tones/nullsilence.WAV) and "
        "are excluded from the stimulus set.",
    ),
}


def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def build_readme(ds: str, results: Path, gate: str, roi_label: str = "whole-brain") -> str:
    blurb, paper, data_url, notes = DATASET_BLURB.get(
        ds, (f"{ds}", "", f"https://openneuro.org/datasets/{ds}", ""))

    ceil_path = results / f"ceilings_{ds}.csv"
    ceilings = pd.read_csv(ceil_path) if ceil_path.exists() else pd.DataFrame()
    summary = read_json(results / "summary.json")
    control = read_json(results / "control" / "summary.json")
    dims = read_json(results / "control" / "dimensionality_summary.json")
    fam_path = results / "alignment_by_family.csv"
    # Whether the LM grid produced anything at all. Distinct from the gate:
    # a dataset can gate-fail and still have alignment numbers (published for
    # completeness, per the fail branch below), or gate-pass and have none if
    # the grid itself never ran (crashed, wrong env, killed early) -- caught
    # 2026-08-30 when a local run's every checkpoint load failed for
    # environment reasons and this function still said "the alignment numbers
    # below" and listed alignment_by_*.csv/figures in Files with nothing
    # there to back either claim, once for a whole-brain run and again for an
    # ROI one before this check existed.
    grid_ran = bool(summary) and summary.get("n_rows", 0)
    fam = pd.read_csv(fam_path) if fam_path.exists() else pd.DataFrame()

    L: list[str] = []
    A = L.append
    A(f"# Brain–language-model alignment: {ds} ({roi_label})")
    A("")
    A(blurb)
    A("")
    A(f"- Paper: {paper}")
    A(f"- Data: {data_url}")
    A(f"- Generated: {date.today().isoformat()}")
    A(f"- Pipeline: https://github.com/suchirsalhan/cdl-representations-brains-babylms")
    if roi_label != "whole-brain":
        A(f"- Masking: **{roi_label}** -- see DATASETS.md section 10 for the three-level standard "
          f"(phonology/language/all) this is part of, and how it differs from the whole-brain reference.")
    A("")

    # ----- the gate, first, always -----
    A("## Read this first: does the measurement work?")
    A("")
    A("Every alignment number in this dataset is only as meaningful as the brain")
    A("RDMs it was computed against. So before any model result, the same")
    A("pipeline is asked whether *anything* stimulus-driven correlates with those")
    A("RDMs — stimulus duration, intensity, word length, frequency, phoneme and")
    A("syllable counts, an acoustic model of the audio where the stimuli are")
    A("audio, and the study's own condition contrast — each tested by a")
    A("permutation test that shuffles stimulus identity.")
    A("")
    if gate == "pass":
        n = control.get("n_significant_holm", 0)
        tot = control.get("n_stimulus_tests_corrected", "?")
        best = control.get("best_control", "?")
        A(f"**GATE: PASSED.** {n}/{tot} stimulus tests are significant after Holm")
        A(f"correction. The strongest is `{best}` "
          f"(ρ = {control.get('best_rsa', float('nan')):.4f}, "
          f"z = {control.get('best_z', float('nan')):.1f}).")
        A("")
        A("The pipeline recovers real stimulus structure from these RDMs, so a")
        A("null result below is a finding about the models rather than a failure")
        A("of the instrument.")
    elif gate == "fail":
        tot = control.get("n_stimulus_tests_corrected", "?")
        A(f"**GATE: FAILED. 0/{tot} stimulus tests are significant** after Holm")
        A("correction — not the acoustic model of the audio the children actually")
        A("heard, not the study's own experimental contrast.")
        A("")
        if grid_ran:
            A("**The alignment numbers below are therefore uninterpretable as")
            A("evidence about language models.** They measure a representational")
            A("geometry that does not demonstrably encode the stimuli. They are")
            A("published for completeness and for whoever fixes the estimator, not as")
            A("a result. Do not cite them as evidence that models fail to align with")
            A("the developing brain.")
        else:
            A("**No alignment numbers exist in this results directory** -- the")
            A("language-model grid produced zero rows for this run (see this run's")
            A("own logs for why: a real crash, an environment problem, or simply")
            A("never having been run). That is independent of the gate result")
            A("above, which is real either way.")
        if dims:
            A("")
            A("Measured cause, from `control/`:")
            A("")
            A(f"- RDM effective rank: **{dims.get('group_rdm_effective_rank_median')}** "
              f"of {dims.get('n_stim_median')} stimuli")
            A(f"- voxels per pattern: {dims.get('pattern_n_voxels_median'):,}" if
              dims.get('pattern_n_voxels_median') else "")
            # .get(k, default) still returns None when the key EXISTS and is
            # null, which is what the dimensionality summary writes when the
            # global-signal probe did not run. Formatting None with :.2f raised
            # TypeError and took the whole publish down.
            _pc1 = dims.get('pc1_vs_global_signal_median')
            if _pc1 is not None:
                A(f"- leading component vs the pattern's global signal: |ρ| = {_pc1:.2f}")
            A("")
            # Do NOT assert degeneracy: it is a measurement, and on these
            # datasets it is often false. ds003604 sat at rank ~3 of 40-48, which
            # is what made its null uninterpretable; ds002236 measures 70 of 96
            # and ds006239 similarly, so the same sentence would have been a
            # fabricated explanation attached to a real number. Decide from the
            # ratio actually recorded above.
            _rank = dims.get('group_rdm_effective_rank_median')
            _nstim = dims.get('n_stim_median')
            _degenerate = (
                _rank is not None and _nstim not in (None, 0)
                and (_rank / _nstim) < 0.25
            )
            if _degenerate:
                A("This reproduces what was found on ds003604: the per-stimulus GLM")
                A("betas are near-degenerate, so the RDM cannot express stimulus-level")
                A("structure regardless of what it is compared against. The estimator")
                A("is shared across datasets, which is why the failure repeats.")
            else:
                A("Note that this is NOT ds003604's failure mode. There, the RDM")
                A("effective rank was ~3 of 40-48 stimuli -- near-degenerate betas")
                A("that could not express stimulus-level structure at all. The rank")
                A("recorded above is a large fraction of the stimulus count, so these")
                A("RDMs do carry stimulus structure and the control failing here means")
                A("the specific controls tested did not reach significance, not that")
                A("the measurement is uninterpretable. Check `control/` for which")
                A("controls ran: an acoustic or visual control needs the dataset's")
                A("stimulus files present, and reports zero features if they are not.")
    else:
        A("**GATE: NOT RUN.** Treat everything below as provisional.")
    A("")

    # ----- what was built -----
    A("## What was built")
    A("")
    if len(ceilings):
        A(f"{len(ceilings)} task × session cells, each an RDM over the stimuli")
        A("shared by that cell's subjects, with voxel patterns z-scored **within")
        A("run** before aggregation (without that, the RDM measures scanner drift")
        A("rather than language) and an inter-subject noise ceiling.")
        A("")
        cols = [c for c in ["task", "session", "n_stim", "ceiling_lower",
                            "ceiling_upper", "ceiling_n"] if c in ceilings.columns]
        A(ceilings[cols].to_markdown(index=False))
        A("")
    if summary:
        A(f"Model grid: **{summary.get('n_families', '?')} families**, "
          f"{summary.get('n_rows', '?')} alignment rows across "
          f"{summary.get('n_cells', '?')} cells.")
        A("")
        A("| | |")
        A("|---|---|")
        A(f"| mean noise ceiling | {summary.get('ceiling_lower_mean', float('nan')):.3f} |")
        A(f"| best alignment anywhere | {summary.get('best_rsa_abs', float('nan')):.4f} |")
        A(f"| as a fraction of ceiling | {summary.get('best_frac_of_ceiling', float('nan')) * 100:.1f}% |")
        if summary.get("families_equivalent_to_zero") is not None:
            A(f"| families equivalent to zero (TOST ±{summary.get('sesoi')}) | "
              f"{summary['families_equivalent_to_zero']}/{summary.get('n_families')} |")
        if summary.get("scale_trend_rho") is not None:
            A(f"| Pythia scale trend | ρ = {summary['scale_trend_rho']:+.3f}, "
              f"p = {summary.get('scale_trend_p', float('nan')):.2f} |")
        A("")
    if len(fam):
        A("### Per family")
        A("")
        cols = [c for c in ["family", "n_checkpoints", "rsa_mean", "rsa_sd",
                            "rsa_abs_max", "frac_of_ceiling_abs_max",
                            "p_equivalence_tost"] if c in fam.columns]
        A(fam[cols].round(4).to_markdown(index=False))
        A("")

    A("## Dataset-specific notes")
    A("")
    A(notes)
    A("")

    A("## Files")
    A("")
    if not grid_ran:
        A("**No alignment or figure files exist in this results directory** --")
        A("the table below is what a completed run produces; this run's own")
        A("logs say why these are absent.")
        A("")
    A("| path | what | present here |")
    A("|---|---|---|")
    _files = [
        ("alignment_by_checkpoint.csv", "every model × checkpoint × cell, with ceiling"),
        ("alignment_by_family.csv", "per family, with equivalence tests"),
        ("alignment_by_cell.csv", "per task × session"),
        (f"ceilings_{ds}.csv", "noise ceiling per cell"),
        ("control/", "the positive control and RDM dimensionality — the gate"),
        ("scale_ladder.csv", "the Pythia 70M→1.4B scale test"),
        ("fig_*.pdf, fig_*.png", "figures"),
    ]
    for fname, what in _files:
        glob_name = fname.split(",")[0].split()[0]
        present = "✓" if list(results.glob(glob_name)) or (results / glob_name).exists() else "—"
        A(f"| `{fname}` | {what} | {present} |")
    A("")
    A("## Method")
    A("")
    A("Representational similarity analysis. For each cell, a brain RDM over")
    A("stimuli (correlation distance between per-stimulus GLM beta patterns,")
    A("within-run z-scored, aggregated across subjects) is compared by Spearman")
    A("correlation with a model RDM over the same stimuli, taken from each")
    A("checkpoint's hidden states. Alignment is reported raw and as a fraction of")
    A("the inter-subject noise ceiling, and judged against a null built from the")
    A("PARC suite — 18 models differing only by random seed, which is what 'no")
    A("effect' looks like on this measurement.")
    A("")
    A("Null and fixation trials are excluded from the stimulus set. For paired")
    A("designs the stimulus identity is the pair, not either word alone.")
    return "\n".join(x for x in L if x is not None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--gate", default="unknown")
    ap.add_argument("--org", default=ORG)
    ap.add_argument("--roi-set", default=None,
                    help="e.g. 'phonology'/'language'/'all' (see src/preprocessing/roi_atlas.py) -- "
                         "the SAME repo as the whole-brain default (--roi-set omitted), nested under "
                         "its own 'roi-<set>/' path so the four masking levels' results coexist "
                         "without overwriting each other. Match whatever ROI_SET was set to for the "
                         "run being published; omit entirely for a whole-brain run.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    results = Path(a.results)
    if not results.exists():
        print(f"no results at {results}")
        sys.exit(1)

    roi_label = f"roi-{a.roi_set.replace(',', '+')}" if a.roi_set else "whole-brain"
    path_in_repo = roi_label if a.roi_set else None  # None -> repo root, byte-identical to before this existed

    repo_id = f"{a.org}/brain-lm-alignment-{a.dataset}"
    readme = build_readme(a.dataset, results, a.gate, roi_label)
    (results / "README.md").write_text(readme)
    print(f"README: {len(readme)} chars -> {results / 'README.md'}")

    if a.dry_run:
        print(readme[:2000])
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    before = set(api.list_repo_files(repo_id, repo_type="dataset"))
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(results),
        path_in_repo=path_in_repo,
        commit_message=f"{a.dataset} ({roi_label}): alignment results (control gate: {a.gate})")
    after = set(api.list_repo_files(repo_id, repo_type="dataset"))
    print(f"https://huggingface.co/datasets/{repo_id}" + (f"/tree/main/{path_in_repo}" if path_in_repo else ""))
    print(f"files {len(before)} -> {len(after)}")
    removed = before - after
    print("REMOVED:", sorted(removed) if removed else "none")


if __name__ == "__main__":
    main()
