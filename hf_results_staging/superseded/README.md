# Superseded results — kept for completeness, do not use

## `early_tier1/`

An **early, partial** Tier-1 pass run on 2026-08-18 while brain preprocessing was still
finishing, over **7 of the 10** families and only **8 of the 12** task × session cells
(606 alignment rows). It was superseded a few hours later by the official Tier 1, which
covers **10 families and all 12 cells** (3,144 rows) and is what lives in `by-model/` and
`overall/`.

These files are here only so the repository contains everything that was produced. They
are deliberately **not** declared as dataset viewer configs, because their numbers differ
from the official ones purely through incomplete coverage and would otherwise be easy to
mistake for the real thing.

Everything in the root README's warning applies to these files too, and more so: the brain
alignment numbers here are confounded by scanner run **and** computed over partial
coverage. Use `by-model/` and `overall/` instead.
