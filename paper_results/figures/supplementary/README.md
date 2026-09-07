# Supplementary figures S1-S9

Built by `scripts/make_supplementary_figures.py` (shared loaders in
`scripts/figlib.py`). Each figure writes `.pdf` (the artefact), `.png` (a
300-DPI preview) and `.csv` (the exact table the panel was drawn from).

```bash
python scripts/make_supplementary_figures.py          # all nine
python scripts/make_supplementary_figures.py S3 S4    # a subset
```

Everything runs offline: the AAL atlas S3 needs is already cached under
`../../../nilearn_data/aal_SPM12`.

## Style

All eight use the MECO house style, vendored verbatim as
`src/viz/plot_style.py` from `beetle-analyze-meco/analyze/plot_style.py` so
this repo stays reproducible on its own. Typography, palette, column widths
(3.3" / 5.0" / 6.9"), Type-42 embedded fonts and the PDF+PNG artefact pair all
come from there. Colour assignments are built from that palette rather than
invented: domains take `B1`/`B2`/`B3`/`B4`, masks take neutral grey for
whole-brain and `B1`/`B2`/`B3` for the three ROIs, and age bands take the
sequential map. Re-vendor rather than edit if the style itself changes.

## What each figure adds to Section 5

| | Question Section 5 does not answer | Headline |
|---|---|---|
| S1 | Does alignment depend on the child's age? | Yes, and strongly — but age is partly confounded with dataset, and where the two can be separated (age 9), the dataset wins |
| S2 | Small relative to what? | Ceilings run 0.24-0.88; ceiling-normalised, phonology reaches ~7% at 1e10 tokens |
| S3 | Where did the ROI analyses look? | The AAL masks themselves, each labelled with the alignment measured under it |
| S4 | Is the domain pattern anatomically specific? | Grammar is: whole-brain ≈ auditory > motor > phonology, going negative under the phonology mask. The other three domains are mask-invariant |
| S5 | What does Table 5's ρ = −0.641 look like? | A clean negative line — plausibility accuracy and plausibility alignment move in opposite directions over training |
| S6 | Does scale or architecture matter? | Scale, no. Architecture, yes: Pythia's grammatical alignment is 3-5x mamba's and rwkv's at matched seeds |
| S7 | Both sides have a localisation measure — do they agree? | Brain-side Gini and model-side Gini sit in different ranges and neither tracks the other |
| S8 | Where in the network does each domain live? | Grammar, semantics and plausibility migrate deeper with training (0.45 → 0.7); phonology stays flat at ~0.45 |
| S9 | Where is the fourth study? | ds001894 appears here, in the only alignment run that covers it — plus a coverage map of both runs across all four studies |

## All four studies, and why they are not all in one figure

Four neuroimaging studies were collected — ds001894, ds002236, ds003604,
ds006239 — and all four are used across this figure set, but **no single
alignment run covers all four**, so they cannot share a pooled panel:

| run | ds001894 | ds002236 | ds003604 | ds006239 |
|---|---|---|---|---|
| 29-family grid (`results/alignment_rows.csv`, the main results) | not run | ✓ | ✓ | ✓ |
| 15-family devai grid (`pipeline/.../devai_grid_wrn`) | ✓ | ✓ | not run | ✓ |
| brain-side localisation (`pipeline/.../fmri_wrn/*/localization`) | ✓ | ✓ | ✓ | ✓ |

`results/coverage_matrix.csv` records ds001894 as "not run" for every variant
of the main grid, which is why it is absent from Figures 1-2 and from S1-S8.
It does have within-run-normalised alignment in the earlier 15-family devai
grid, and that is what S9 draws.

The two alignment runs must not be pooled. They share three datasets and nine
model families, and where they overlap their ρ values differ by up to 0.067 —
wider than the entire effect range the paper reports. So S9's panels A-C stay
wholly inside the devai run, and S1-S8 stay wholly inside the main grid.
ds001894 also appears in S7 panel A, which is brain-side only and does cover
all four.

Getting ds001894 into the main results would mean running the 29-family grid
on it; nothing in the figure code can substitute for that.

## Two things to check before submission

**1. Table 2's phonology row does not reconcile.** Table 2 reports phonology
with *zero* token bins whose CI excludes zero in the positive direction (0
positive / 4 negative / 11 containing zero) and a last-bin mean of +0.0044.
Recomputing from `results/alignment_rows.csv` reproduces the other three
domains closely — plausibility exactly (4 positive / 9 negative / 2 zero),
grammar as 14/0/1 against the reported 15/0/0, semantics as 5/3/7 against
7/3/5 — but gives phonology 6 positive / 5 negative / 4 zero with a last-bin
mean of +0.0079. That is not a CI-width difference: the sign pattern differs.
No subset of the three datasets reproduces the reported row either
(ds003604 alone gives 0/12/3, ds002236 alone 9/0/6, all three 6/5/4). S2's
panel B and Figure 1 will therefore disagree about phonology at high token
counts unless one of them is regenerated. Worth resolving before the
phonology claim in §6.1 ("phonology never reaches a positive, zero-excluding
bin at any point in training") goes out.

**2. The unit a CI is computed over.** Each checkpoint contributes one row per
(dataset, task, session) cell, so treating rows as independent inflates n up
to 8x. Every figure here averages within checkpoint first (`figlib.
DEFAULT_CI_UNIT`), which is also what reproduces Table 4's n = 30 in the last
bin. If the main text's intervals were computed over raw rows, they are
narrower than they should be.

## Data sources

| Figure | Reads |
|---|---|
| S1, S2, S6 | `results/alignment_rows.csv`, `paper_results/parc/parc_seed_spread.csv` |
| S3 | `hf_package/ds003604{,-roiauditory,-roimotor,-roiphonology}`, `results/roi_comparison.csv`, AAL SPM12 via `src/preprocessing/roi_atlas.py` |
| S4 | the same four packages |
| S5, S7, S8 | `hf_package/ds003604` (by-model behaviour, localisation, layerwise), `pipeline/data/processed/fmri_wrn/*/localization` (all four studies) |
| S9 | `pipeline/data/processed/language_models/devai_grid_wrn/{ds001894,ds002236,ds006239}` plus `results/alignment_rows.csv` for the coverage map |

S3 draws mask *definitions* in MNI space, not measured per-voxel effects.
Per-voxel maps cannot be reconstructed from what is on disk — patterns are
stored as flat masked vectors without the mask affine — so a mask outline
coloured by the scalar measured under it is the strongest honest version. See
the note at the top of `scripts/plot_activation_by_age_domain.py`.
