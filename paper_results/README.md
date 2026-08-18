# paper_results

Version-controlled **result summaries** for the paper — the small, text-only
artifacts (claim summaries, isolation comparisons, held-out CV, LaTeX tables).
Populated by `scripts/backup_results.py --git-summary-dir paper_results` at the
end of `slurm/run_devai_grid.sh`, then committed to GitHub.

Layout mirrors the per-dataset results:

```
paper_results/
  <dataset>/devai_summary_<family>.csv        # R1–R6 claim tests
  <dataset>/isolation_comparison_<family>.csv # LM vs brain isolation
  <dataset>/heldout_predictor.csv             # cross-family held-out R²
  <dataset>/table1_model_suite.tex
  <dataset>/table2_claim_tests.tex
```

Full results (all per-checkpoint CSVs + figures) are gitignored and backed up
to the HuggingFace dataset (`BACKUP_HF_REPO`, e.g. `BrainAlign/cdl-devai-results`).
Figures (PDF/PNG) live only in the HF backup, not in git.
