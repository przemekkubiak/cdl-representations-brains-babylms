#!/usr/bin/env python
"""Positive control: can this pipeline detect alignment that actually exists?

WHY THIS IS THE BLOCKING EXPERIMENT. Every alignment number we report is a null,
and a null is only worth something if the instrument has demonstrated power. The
noise ceiling (0.85) shows the brain RDMs are *reliable* -- two halves of the
subject pool agree. It does not show they carry *stimulus-driven* signal that
this RSA pipeline can recover. If nothing at all correlates with these RDMs, the
LM null is vacuous: we would have shown only that our measurement is deaf.

ds003604 is an AUDITORY design -- every stimulus is a .wav -- so the textbook
control applies directly: a low-level acoustic model of the stimuli should
correlate with the brain response if anything does. The dataset also ships
per-stimulus psycholinguistic norms (duration, intensity, frequency, phoneme and
syllable counts), which give a battery of cheap controls with published
provenance rather than ones we invented.

Each control RDM is tested against each corrected session RDM with the same RSA
that the model grid uses, and judged by a permutation test that shuffles
stimulus identity -- the null that is actually appropriate here, since it
destroys the stimulus correspondence while preserving both RDMs' internal
structure.

Read the verdict at the bottom of the output, not any single number:

  * some controls significant  -> the pipeline has power; the LM null is real
  * NOTHING significant        -> the RDMs carry no recoverable stimulus signal
                                  and no claim about LMs can be made from them
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = "#2a78d6"
NRM = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
CEIL = "#c9c8c1"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "figure.dpi": 150,
})

# ds003604's tasks, used only as a fallback. The real list is discovered from
# the RDM root, because every other dataset in the registry has different task
# names (AudRhyme/VisSem..., ReadPhon/LocalSem..., AAWord/AVWord...) and a
# hardcoded list silently reports "no RDM for Sem" and produces nothing.
DEFAULT_TASKS = ["Sem", "Phon", "Gram", "Plaus"]
TASKS = list(DEFAULT_TASKS)


def discover_tasks(rdm_root) -> list:
    from pathlib import Path as _P
    root = _P(rdm_root)
    found = sorted(d.name for d in root.iterdir()
                   if d.is_dir() and any(d.glob("session_rdm_*.npz"))) if root.exists() else []
    return found or DEFAULT_TASKS

# Controls that describe HOW the data was acquired rather than WHAT was
# presented. Kept in their own family: on the uncorrected RDMs they are expected
# to be large, and that expectation is what validates the test.
ACQUISITION = {"run_identity", "presentation_order"}

# Per-task column names for the shipped norms. The four tasks use different
# schemas (word-pair tasks name word_A/word_B; sentence tasks give a single
# averaged column), so the mapping is explicit rather than guessed.
NORMS = {
    "duration": {"Sem": ["total_stim_duration"], "Phon": ["total_stim_duration"],
                 "Gram": ["stim_duration"], "Plaus": ["stim_duration"]},
    "intensity": {t: ["stim_average_intensity"] for t in TASKS},
    "word_length": {"Sem": ["word_A_length", "word_B_length"],
                    "Phon": ["word_A_length", "word_B_length"],
                    "Gram": ["stim_averaged_word_length"],
                    "Plaus": ["stim_averaged_word_length"]},
    "n_syllables": {"Sem": ["word_A_number_syllables", "word_B_number_syllables"],
                    "Phon": ["word_A_number_syllables", "word_B_number_syllables"],
                    "Gram": ["stim_total_number_syllable"],
                    "Plaus": ["stim_total_number_syllable"]},
    "n_phonemes": {"Sem": ["word_A_number_ phonemes", "word_B_number_ phonemes"],
                   "Phon": ["word_A_number_ phonemes", "word_B_number_ phonemes"]},
    "log_frequency": {"Sem": ["word_A_frequency", "word_B_frequency"],
                      "Phon": ["word_A_frequency", "word_B_frequency"],
                      "Gram": ["stim_averaged_frequency"],
                      "Plaus": ["stim_averaged_frequency"]},
}


# ------------------------------------------------------------------ rdms ----
def load_cell(rdm_root: Path, task: str, session: str) -> dict | None:
    f = rdm_root / task / f"session_rdm_{session}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return {
        "rdm": np.asarray(z["rdm"], dtype=float),
        "stimuli": [str(s) for s in z["stimuli"]],
        "texts": [str(s) for s in z["stimulus_texts"]],
        "trial_types": [str(s) for s in z["trial_types"]],
        "ceiling": float(z["noise_ceiling_lower"]) if "noise_ceiling_lower" in z else np.nan,
        "wrn": bool(z["within_run_normalized"]) if "within_run_normalized" in z else False,
    }


def triu(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(m, k=1)
    return m[iu]


# -------------------------------------------------------------- controls ----
def rdm_from_scalar(v: np.ndarray) -> np.ndarray:
    """|difference| RDM from a per-stimulus scalar."""
    v = np.asarray(v, dtype=float)
    return np.abs(v[:, None] - v[None, :])


def rdm_from_labels(labels: list[str]) -> np.ndarray:
    """0 within condition, 1 between. The design's own contrast."""
    a = np.asarray(labels, dtype=object)
    return (a[:, None] != a[None, :]).astype(float)


def edit_distance(a: str, b: str) -> float:
    """Normalised Levenshtein. Small n, so a plain DP is fine and keeps the
    dependency list unchanged."""
    if a == b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b), 1)


def rdm_from_texts(texts: list[str]) -> np.ndarray:
    n = len(texts)
    m = np.zeros((n, n))
    low = [t.lower() for t in texts]
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = edit_distance(low[i], low[j])
    return m


def read_wav(path: Path) -> tuple[np.ndarray, int] | None:
    """Mono float signal. Uses the stdlib rather than adding librosa/soundfile:
    these are plain PCM wavs and the venv's torch/cu126 pin is not worth
    disturbing for a spectrogram."""
    try:
        with wave.open(str(path), "rb") as w:
            n, sr, sw, ch = w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
            raw = w.readframes(n)
    except Exception:
        return None
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
    if dt is None:
        return None
    x = np.frombuffer(raw, dtype=dt).astype(np.float64)
    if sw == 1:
        x = x - 128.0
    x /= float(2 ** (8 * sw - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def mel_filterbank(n_fft_bins: int, sr: int, n_mels: int = 40) -> np.ndarray:
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)  # noqa: E731
    lo, hi = hz2mel(50.0), hz2mel(min(8000.0, sr / 2))
    pts = mel2hz(np.linspace(lo, hi, n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft_bins)
    fb = np.zeros((n_mels, n_fft_bins))
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        left = (freqs - l) / max(c - l, 1e-9)
        right = (r - freqs) / max(r - c, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
    return fb


def acoustic_features(paths: dict[str, Path], n_mels: int = 40,
                      n_env: int = 32) -> tuple[dict, dict]:
    """Mean log-mel spectrum and amplitude envelope per stimulus.

    The spectrum is the standard low-level auditory model; the envelope carries
    rhythm and duration structure. Both are computed from the audio the subjects
    actually heard.
    """
    spec, env = {}, {}
    for name, p in paths.items():
        got = read_wav(p)
        if got is None:
            continue
        x, sr = got
        if x.size < 256:
            continue
        f, _, S = signal.spectrogram(x, fs=sr, nperseg=512, noverlap=256,
                                     scaling="spectrum", mode="magnitude")
        fb = mel_filterbank(S.shape[0], sr, n_mels)
        mel = np.log(fb @ S + 1e-10)
        spec[name] = mel.mean(axis=1)
        # envelope resampled to a fixed length so stimuli of different duration
        # are comparable in shape; duration itself is a separate control.
        rms = np.sqrt(np.convolve(x ** 2, np.ones(int(sr * 0.01)) / int(sr * 0.01), "same"))
        idx = np.linspace(0, len(rms) - 1, n_env).astype(int)
        e = rms[idx]
        env[name] = e / (e.max() + 1e-12)
    return spec, env


def image_features(paths: dict[str, Path], size: int = 32) -> tuple[dict, dict]:
    """Low-level visual features: greyscale pixels and mean luminance.

    The visual counterpart of the acoustic control. ds006239 is entirely visual
    (669 bitmaps, no audio at all), so without this its gate reduces to the
    design controls and cannot test whether the RDMs carry any stimulus signal.
    A downsampled greyscale image is the standard first-order model of what early
    visual cortex responds to, and mean luminance is its scalar summary.
    """
    pix, lum = {}, {}
    try:
        from PIL import Image
    except Exception:
        return pix, lum
    for name, path in paths.items():
        try:
            with Image.open(path) as im:
                g = im.convert("L").resize((size, size))
                v = np.asarray(g, dtype=float).ravel() / 255.0
        except Exception:
            continue
        pix[name] = v
        lum[name] = np.array([v.mean()])
    return pix, lum


def _resolve(vecs: dict, identity: str) -> np.ndarray | None:
    """Feature vector for one stimulus identity, which may be a PAIR.

    Pair designs join two stimulus files with "|". The trial presents both, so
    the trial's low-level content is the mean of the two components' features.
    A pair is usable when at least one side has features -- for a mixed-modality
    trial like "a.wav|b.bmp" the audio model sees the audio side and the image
    model sees the image side, which is the honest reading of each.
    """
    parts = [Path(p).name for p in str(identity).split("|")]
    got = [vecs[p] for p in parts if p in vecs]
    if not got:
        return None
    n = min(len(g) for g in got)
    return np.mean([g[:n] for g in got], axis=0)


def rdm_from_vectors(vecs: dict, order: list[str]) -> np.ndarray | None:
    """Correlation-distance RDM over per-stimulus feature vectors."""
    resolved = [_resolve(vecs, s) for s in order]
    if any(r is None for r in resolved):
        return None
    M = np.vstack(resolved)
    M = M - M.mean(axis=1, keepdims=True)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (M @ M.T)


def find_norms(stim_root: Path, task: str) -> pd.DataFrame | None:
    """Locate a per-stimulus norms table, whatever the dataset chose to call it.

    Only ds003604 uses the per-task
    `Stimulus_Characteristics/task-<T>_Stimulus_Characteristics.tsv` layout.
    ds006239 ships a single `Stimulus_Characteristics.tsv`, ds001894 ships
    `WordAudDuration.tsv`/`NonWordAudDuration.tsv`, and ds002236's is a single
    file whose NAME IS MISSPELLED in the release -- `Stimulus_Charactersitics.tsv`.
    Globbing for the misspelling as well is not defensive programming for its own
    sake; without it that dataset silently loses every norm-based control.
    """
    exact = stim_root / "Stimulus_Characteristics" / f"task-{task}_Stimulus_Characteristics.tsv"
    if exact.exists():
        try:
            return pd.read_csv(exact, sep="\t")
        except Exception:
            return None
    frames = []
    for pat in ("*Charact*.tsv", "*Charact*.csv", "*Duration.tsv"):
        for f in sorted(stim_root.rglob(pat)):
            try:
                frames.append(pd.read_csv(f, sep="\t" if f.suffix == ".tsv" else ","))
            except Exception:
                continue
    if not frames:
        return None
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ test ----
def rank_square(m: np.ndarray) -> np.ndarray:
    """Square matrix holding the ranks of the upper-triangle values.

    Lets the permutation test reindex ranks directly: permuting stimulus labels
    maps the triangle onto itself, so the multiset of values is preserved and
    only the pairing changes.
    """
    iu = np.triu_indices_from(m, k=1)
    r = stats.rankdata(m[iu])
    out = np.zeros_like(m, dtype=float)
    out[iu] = r
    return out + out.T


def rsa_with_permutation(brain: np.ndarray, ctrl: np.ndarray, n_perm: int,
                         seed: int = 0) -> dict:
    iu = np.triu_indices_from(brain, k=1)
    b = stats.rankdata(brain[iu])
    b = (b - b.mean()) / (b.std() + 1e-12)
    cr = rank_square(ctrl)
    c = cr[iu]
    c0 = (c - c.mean()) / (c.std() + 1e-12)
    rho = float(np.dot(b, c0) / len(b))

    rng = np.random.default_rng(seed)
    n = brain.shape[0]
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        v = cr[np.ix_(p, p)][iu]
        v = (v - v.mean()) / (v.std() + 1e-12)
        null[k] = np.dot(b, v) / len(b)
    # +1 correction: a permutation p-value can never legitimately be 0.
    p_two = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return {"rsa": rho, "p_perm": p_two,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "z": float((rho - null.mean()) / (null.std(ddof=1) + 1e-12))}


# ------------------------------------------------------------------ main ----
def build_controls(cell: dict, task: str, norms: pd.DataFrame | None,
                   spec: dict, env: dict, pix: dict | None = None,
                   lum: dict | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    pix = pix or {}
    lum = lum or {}
    stim, texts = cell["stimuli"], cell["texts"]

    if norms is not None and "stim_file" in norms.columns:
        idx = norms.drop_duplicates("stim_file").set_index("stim_file")
        for label, per_task in NORMS.items():
            cols = per_task.get(task)
            if not cols or any(c not in idx.columns for c in cols):
                continue
            vals = []
            for s in stim:
                if s not in idx.index:
                    vals.append(np.nan); continue
                v = pd.to_numeric(pd.Series([idx.at[s, c] for c in cols]),
                                  errors="coerce")
                vals.append(float(v.mean()))
            v = np.asarray(vals, dtype=float)
            if np.isnan(v).any():
                continue
            if label == "log_frequency":
                v = np.log10(np.clip(v, 1, None))
            out[label] = rdm_from_scalar(v)

    # Design/acquisition controls. run_identity is the confound the whole
    # correction exists to remove: on the RAW RDMs it should be large, on the
    # corrected ones ~0. That contrast is what proves this permutation test can
    # see structure when structure is present -- a positive control for the
    # TEST, not just for the RDMs.
    if norms is not None and {"stim_file", "run"} <= set(norms.columns):
        idx = norms.drop_duplicates("stim_file").set_index("stim_file")
        if all(s in idx.index for s in stim):
            out["run_identity"] = rdm_from_labels([str(idx.at[s, "run"]) for s in stim])
            pos = {f: i for i, f in enumerate(idx.index)}
            out["presentation_order"] = rdm_from_scalar(
                np.array([pos[s] for s in stim], dtype=float))

    out["condition"] = rdm_from_labels(cell["trial_types"])
    out["text_edit_distance"] = rdm_from_texts(texts)
    m = rdm_from_vectors(spec, stim)
    if m is not None:
        out["acoustic_spectrum"] = m
    m = rdm_from_vectors(env, stim)
    if m is not None:
        out["acoustic_envelope"] = m
    m = rdm_from_vectors(pix, stim)
    if m is not None:
        out["visual_pixels"] = m
    if lum:
        vals = [_resolve(lum, s) for s in stim]
        if all(v is not None for v in vals):
            out["visual_luminance"] = rdm_from_scalar(
                np.array([float(v[0]) for v in vals]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdm-root", default="data/processed/fmri_wrn/ds003604")
    ap.add_argument("--compare-root", default="data/processed/fmri/ds003604",
                    help="uncorrected RDMs, run alongside as a known-positive case "
                         "for the test itself (empty string to skip)")
    ap.add_argument("--stimuli", default="data/brain/ds003604/stimuli")
    ap.add_argument("--sessions", default="ses-5,ses-7,ses-9")
    ap.add_argument("--out", default="paper_results/control")
    ap.add_argument("--perms", type=int, default=5000)
    ap.add_argument("--lm-cells", default="paper_results/corrected/alignment_by_cell.csv",
                    help="for the side-by-side against the language models")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rdm_root, stim_root = Path(a.rdm_root), Path(a.stimuli)
    sessions = [s.strip() for s in a.sessions.split(",") if s.strip()]

    global TASKS
    TASKS = discover_tasks(rdm_root)
    print(f"tasks discovered: {TASKS}")

    # Case-insensitive: ds003604 ships .wav, ds002236 ships .WAV. A lowercase-only
    # glob silently yields zero audio and the acoustic control -- the strongest
    # low-level control for an auditory design -- vanishes without a word.
    wavs = {p.name: p for p in stim_root.rglob("*")
            if p.is_file() and p.suffix.lower() == ".wav" and p.stat().st_size > 1024}
    print(f"{len(wavs)} stimulus wav files readable under {stim_root}")
    spec, env = acoustic_features(wavs) if wavs else ({}, {})
    print(f"acoustic features extracted for {len(spec)} stimuli")
    imgs = {p.name: p for p in stim_root.rglob("*")
            if p.is_file() and p.suffix.lower() in (".bmp", ".jpg", ".jpeg", ".png")
            and p.stat().st_size > 512}
    pix, lum = image_features(imgs) if imgs else ({}, {})
    print(f"{len(imgs)} image files -> visual features for {len(pix)} stimuli")

    variants = [("corrected", rdm_root)]
    if a.compare_root and Path(a.compare_root).exists():
        variants.append(("uncorrected", Path(a.compare_root)))
    print(f"variants: {[v for v, _ in variants]}")

    rows = []
    for variant, root in variants:
     for task in TASKS:
        norms = find_norms(stim_root, task)
        for session in sessions:
            cell = load_cell(root, task, session)
            if cell is None:
                print(f"  no RDM for {variant} {task}/{session}")
                continue
            ctrls = build_controls(cell, task, norms, spec, env, pix, lum)
            for name, c in ctrls.items():
                if c.shape != cell["rdm"].shape or not np.isfinite(c).all():
                    continue
                if np.allclose(triu(c), triu(c)[0]):
                    continue  # constant control carries no information
                r = rsa_with_permutation(cell["rdm"], c, a.perms)
                rows.append({"variant": variant, "task": task, "session": session,
                             "control": name,
                             "n_stim": cell["rdm"].shape[0],
                             "ceiling": cell["ceiling"],
                             "frac_of_ceiling": r["rsa"] / cell["ceiling"]
                             if cell["ceiling"] else np.nan,
                             "within_run_normalized": cell["wrn"], **r})
            print(f"  {variant} {task}/{session}: {len(ctrls)} controls tested")

    if not rows:
        print("no results -- check the RDM root and the stimulus directory")
        return
    d = pd.DataFrame(rows)
    d.to_csv(out / "control_by_cell.csv", index=False)

    # Two different questions, so two different families of tests -- pooling them
    # would let the (huge, expected) run-identity effect on the UNCORRECTED RDMs
    # masquerade as evidence that the corrected RDMs carry stimulus signal.
    d["kind"] = np.where(d["control"].isin(ACQUISITION), "acquisition", "stimulus")

    def holm(sub: pd.DataFrame) -> np.ndarray:
        pv = sub["p_perm"].to_numpy()
        m = len(pv)
        adj = np.empty(m); running = 0.0
        for rank, i in enumerate(np.argsort(pv)):
            running = max(running, (m - rank) * pv[i])
            adj[i] = min(1.0, running)
        return adj

    d["p_holm"] = np.nan
    for (variant, kind), sub in d.groupby(["variant", "kind"]):
        d.loc[sub.index, "p_holm"] = holm(sub)
    d["significant"] = d["p_holm"] < 0.05
    d.to_csv(out / "control_by_cell.csv", index=False)

    summ = (d.groupby("control")
              .agg(n_cells=("rsa", "size"), rsa_mean=("rsa", "mean"),
                   rsa_max=("rsa", "max"), z_max=("z", "max"),
                   p_min=("p_perm", "min"), n_significant=("significant", "sum"),
                   frac_of_ceiling_max=("frac_of_ceiling", "max"))
              .reset_index().sort_values("z_max", ascending=False))
    summ.to_csv(out / "control_summary.csv", index=False)

    print()
    print("  --- POSITIVE CONTROL: what correlates with these brain RDMs? ---")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # The verdict is about STIMULUS controls on the CORRECTED RDMs -- that is the
    # question "can this pipeline detect stimulus-driven alignment in the data
    # the LM null was computed on".
    q = d[(d["variant"] == "corrected") & (d["kind"] == "stimulus")]
    n_sig = int(q["significant"].sum())
    best = q.loc[q["z"].idxmax()] if len(q) else None

    # The acquisition controls answer a different question: does the TEST have
    # power at all, and did the correction do what it claims?
    acq = d[d["kind"] == "acquisition"]
    if len(acq):
        piv = acq.pivot_table(index="control", columns="variant", values="rsa",
                              aggfunc="mean").round(4)
        print()
        print("  --- TEST VALIDATION: acquisition structure, before vs after "
              "correction ---")
        print(piv.to_string())
        piv.to_csv(out / "acquisition_controls.csv")
        pre = acq[acq["variant"] == "uncorrected"]
        if len(pre):
            b = pre.loc[pre["rsa"].idxmax()]
            print(f"  the permutation test recovers {b['control']} at rho "
                  f"{b['rsa']:+.3f} (p={b['p_perm']:.2g}) on the uncorrected RDMs")
            print("  -> the test has demonstrated power; a null below is a real null")
    lm_best = np.nan
    lm_path = Path(a.lm_cells)
    if lm_path.exists():
        lm = pd.read_csv(lm_path)
        if "rsa_max" in lm.columns:
            lm_best = float(lm["rsa_max"].max())

    print()
    print("  --- VERDICT: stimulus controls on the CORRECTED RDMs ---")
    if n_sig:
        print(f"  {n_sig}/{len(q)} stimulus x cell tests significant after Holm correction.")
        print(f"  strongest: {best['control']} in {best['task']}/{best['session']} "
              f"-- rho {best['rsa']:+.4f}, z {best['z']:+.1f}, p {best['p_perm']:.2g} "
              f"({best['frac_of_ceiling'] * 100:.1f}% of ceiling)")
        if np.isfinite(lm_best):
            print(f"  best LM cell anywhere for comparison: {lm_best:+.4f}")
        print("  VERDICT: the pipeline recovers real stimulus structure from these")
        print("           RDMs, so the language-model null is a finding, not a")
        print("           failure of the instrument.")
    else:
        print(f"  0/{len(q)} stimulus tests significant after correction.")
        print("  VERDICT: NO recoverable stimulus signal. The LM null is VACUOUS --")
        print("           it shows the measurement is deaf, not that models fail to")
        print("           align. Do not publish the null on these RDMs.")

    summary = {
        "n_tests": int(len(d)),
        "n_stimulus_tests_corrected": int(len(q)),
        "n_significant_holm": n_sig,
        "controls_tested": sorted(d["control"].unique().tolist()),
        "best_control": None if best is None else str(best["control"]),
        "best_rsa": None if best is None else float(best["rsa"]),
        "best_z": None if best is None else float(best["z"]),
        "best_cell": None if best is None else f"{best['task']}/{best['session']}",
        "best_frac_of_ceiling": None if best is None else float(best["frac_of_ceiling"]),
        "lm_best_rsa_for_comparison": None if not np.isfinite(lm_best) else lm_best,
        "n_permutations": a.perms,
        "verdict": "pipeline has power" if n_sig else "no recoverable stimulus signal",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    summ_q = (q.groupby("control")
                .agg(n_cells=("rsa", "size"), rsa_max=("rsa", "max"),
                     n_significant=("significant", "sum"))
                .reset_index()) if len(q) else pd.DataFrame()
    if len(summ_q):
        fig_control(q, summ_q, lm_best, out / "fig_positive_control")
    print(f"\n  wrote -> {out}")


def fig_control(d: pd.DataFrame, summ: pd.DataFrame, lm_best: float, out: Path) -> None:
    """Every control's best cell, against the LM best and the ceiling."""
    s = summ.sort_values("rsa_max")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(s) + 2.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=MUTED, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)

    ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    if np.isfinite(lm_best):
        ax.axvline(lm_best, color=NRM, lw=1.4, zorder=2)
        ax.annotate("best LM cell", xy=(lm_best, len(s) - 0.4), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=NRM)
    colors = [RAW if n else MUTED for n in s["n_significant"]]
    ax.barh(y, s["rsa_max"], color=colors, height=0.62, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(s["control"], fontsize=8.5)
    ax.set_xlabel("best RSA against a corrected brain RDM (Spearman $\\rho$)")
    ax.set_title("Positive control: what this pipeline can detect\n"
                 "(blue = significant after Holm correction)", loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
