#!/usr/bin/env python
"""Is the near-zero model alignment a property of the MODELS or of the MAPPING?

The sweep's model RDM is fixed by one choice: feed the trial string to the LM,
mean-pool the last block's hidden states over tokens, RDM = 1 - corr. But every
stimulus in all three datasets is a word PAIR whose experimental manipulation is
the RELATION between the two words ('bad wad' rhymes, 'ape monkey' is
semantically related, 'bacon wipe' is not). Mean-pooling a two-word string
averages that relation away.

This script holds the brain side fixed and varies the model side:

  models   a strong purpose-built sentence embedding model (a positive control
           with known-good semantic geometry), a large well-trained LM, and the
           sweep's own scale of model
  layers   every hidden layer, not just the last
  schemes  pipeline_meanpool  the sweep's exact mapping, as the baseline
           pair_concat        [emb(w1), emb(w2)] --- keeps both words distinct
           pair_absdiff       |emb(w1) - emb(w2)| --- the contrast between them
           pair_cos1d         cos(emb(w1), emb(w2)) --- relatedness as a scalar,
                              scored as the RDM |c_i - c_j|

Each is scored with the sweep's own estimator against the group RDM and against
the additive-residualised RDM, so the numbers are directly comparable to
`results/alignment_rows.csv`.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoModel, AutoTokenizer

ROOT = Path("/local/scratch/sas245/brainalign-evals")
RDM = ROOT / "data/ds003604-session-rdms"
RES = ROOT / "results"

MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "gpt2-large",
    "EleutherAI/pythia-410m",
]


def ut(m):
    return m[np.triu_indices_from(m, k=1)]


def zscore(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def est(a, b):
    return float(stats.spearmanr(zscore(ut(a)), zscore(ut(b))).statistic)


def additive_fit(D):
    D = np.asarray(D, float).copy()
    n = D.shape[0]
    np.fill_diagonal(D, 0.0)
    R = D.sum(axis=1)
    S = (R.sum() / 2.0) / (n - 1.0)
    return (R - S) / (n - 2.0)


def residual(D):
    a = additive_fit(D)
    E = np.asarray(D, float) - (a[:, None] + a[None, :])
    np.fill_diagonal(E, 0.0)
    return E


def corr_rdm(X):
    """The sweep's model RDM: 1 - Pearson correlation across stimuli."""
    X = np.asarray(X, float)
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    C = (X / n) @ (X / n).T
    return 1.0 - C


@torch.no_grad()
def embed(model, tok, strings, device):
    """Mean-pooled hidden states for every layer. -> [n_layers, n_strings, H]."""
    out = []
    for i in range(0, len(strings), 32):
        batch = strings[i:i + 32]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=32)
        enc = {k: v.to(device) for k, v in enc.items()}
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(torch.stack([(h * m).sum(1) / m.sum(1) for h in hs]).float().cpu())
    return torch.cat(out, dim=1).numpy()


def cells():
    for f in sorted(glob.glob(str(RDM / "*/within-run-normalised/*/session_rdm_*.npz"))):
        q = Path(f)
        yield (q.relative_to(RDM).parts[0], q.parts[-2],
               re.sub(r"session_rdm_(.*)\.npz", r"\1", q.name), np.load(f, allow_pickle=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", default="pair_representations.csv")
    args = ap.parse_args()

    cache = list(cells())
    rows = []
    for name in args.models:
        print(f"== {name}", flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModel.from_pretrained(name, dtype=torch.float32).to(args.device).eval()

        for ds, task, ses, d in cache:
            D = np.asarray(d["rdm"], float)
            E = residual(D)
            texts = [str(t) for t in d["stimulus_texts"]]
            pairs = [t.split() for t in texts]
            w1 = [p[0] if p else "" for p in pairs]
            w2 = [p[1] if len(p) > 1 else "" for p in pairs]

            H_pair = embed(model, tok, texts, args.device)
            H1 = embed(model, tok, w1, args.device)
            H2 = embed(model, tok, w2, args.device)
            n_layers = H_pair.shape[0]

            for li in range(n_layers):
                a, b, c = H_pair[li], H1[li], H2[li]
                cos = (b * c).sum(1) / (np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1) + 1e-12)
                schemes = {
                    "pipeline_meanpool": corr_rdm(a),
                    "pair_concat": corr_rdm(np.concatenate([b, c], axis=1)),
                    "pair_absdiff": corr_rdm(np.abs(b - c)),
                    "pair_cos1d": np.abs(cos[:, None] - cos[None, :]),
                }
                for sname, M in schemes.items():
                    rows.append(dict(
                        model=name, dataset=ds, task=task, session=ses,
                        layer=li, layer_frac=round(li / (n_layers - 1), 3),
                        scheme=sname,
                        rsa_full=round(est(D, M), 4),
                        rsa_relational=round(est(E, M), 4)))
            print(f"   {ds}/{task}/{ses} ({n_layers} layers)", flush=True)
        del model
    df = pd.DataFrame(rows)
    df.to_csv(RES / args.out, index=False)

    print()
    print("best |rsa_full| per model x scheme, over all layers and cells:")
    print(df.groupby(["model", "scheme"]).rsa_full.agg(
        median="median", best=lambda s: s.abs().max()).round(4).to_string())
    print()
    print("per model x scheme: mean over cells of the BEST layer for that cell")
    bl = (df.loc[df.groupby(["model", "scheme", "dataset", "task", "session"]).rsa_full
                 .apply(lambda s: s.abs().idxmax())]
          .groupby(["model", "scheme"]).rsa_full.agg(["mean", "max"]).round(4))
    print(bl.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
