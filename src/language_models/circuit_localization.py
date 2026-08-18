"""Circuit localization for language models: specialization vs. spread.

Ported and generalized from the functional-localizer method in
  - elliepreed/syntax-units
  - anonymous.4open.science/r/syntax-interpretability
(cognitive-neuroscience-style functional localization of LM units; cf. AlKhamissi
et al., "The LLM Language Network", and the Fedorenko sentences>nonwords localizer).

Method (per phenomenon P, per checkpoint):
  1. Feed a *positive* condition (e.g. grammatical / sensible / plausible) and a
     matched *negative* control through the model; hook every transformer block
     and pool activations to one vector per stimulus per layer (last-token pool,
     matching syntax-units).
  2. Per unit u (=(layer, hidden-dim index)): Welch t-test of |positive| vs
     |negative| activations -> selectivity t-value.
  3. Select the top-k% units by t-value across the whole network -> the localized
     "circuit" mask for P (exactly `is_topk` in syntax-units `localize.py`).
  4. Quantify specialization vs spread with concentration + differentiation
     metrics, and (optionally) causally validate by ablating the circuit.

Unlike the source repos, this runs on stock `AutoModelForCausalLM` via forward
hooks (no custom modeling_*.py), so it works for BabyLM (GPT-2) and Pythia
(GPT-NeoX) checkpoints out of the box.

The unit of comparison to the brain: the per-unit t-value map here is the LM
analogue of the per-voxel GLM localizer contrast (condition>control) in the
Wang et al. ds003604 developmental dataset. See PRIVATE_NOTES.md §5-6.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Phenomenon contrasts (positive = condition, negative = control)
# --------------------------------------------------------------------------- #
@dataclass
class PhenomenonContrast:
    """A condition>control localizer contrast for one phenomenon.

    positive: sentences that engage the phenomenon (grammatical / sensible /
              plausible / real-word).
    negative: matched control (ungrammatical / anomalous / implausible / nonword).
    """

    name: str  # Sem | Phon | Gram | Plaus (or any custom key)
    positive: List[str]
    negative: List[str]

    def __post_init__(self):
        n = min(len(self.positive), len(self.negative))
        if n == 0:
            raise ValueError(f"Contrast '{self.name}' has empty positive/negative set")
        # Keep balanced, mirroring LangLocDataset.__len__ = min(pos, neg)
        self.positive = list(self.positive[:n])
        self.negative = list(self.negative[:n])

    def __len__(self) -> int:
        return len(self.positive)


def load_contrast_csv(name: str, path: str) -> PhenomenonContrast:
    """Load a contrast from a CSV.

    Supports two layouts:
      (a) columns ``positive,negative`` (one matched pair per row); or
      (b) syntax-units style: ``stim2..stimN`` sentence columns plus a column
          whose cell is ``+`` (positive) or ``-`` (negative) per row.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    cols = {c.lower(): c for c in df.columns}

    if "positive" in cols and "negative" in cols:
        return PhenomenonContrast(
            name=name,
            positive=[s for s in df[cols["positive"]].tolist() if s],
            negative=[s for s in df[cols["negative"]].tolist() if s],
        )

    # syntax-units layout: build the sentence, then split on the +/- marker cell.
    stim_cols = sorted(
        [c for c in df.columns if c.lower().startswith("stim")],
        key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or 0),
    )
    if not stim_cols:
        raise ValueError(f"{path}: need 'positive'/'negative' or 'stim*' columns")
    sent = df[stim_cols[0]].astype(str)
    for c in stim_cols[1:]:
        sent = sent + " " + df[c].astype(str)
    sent = sent.str.strip()

    pos, neg = [], []
    for i in range(len(df)):
        row = df.iloc[i]
        marker = "".join(v for v in row.values if v in ("+", "-"))
        if "+" in marker:
            pos.append(sent.iloc[i])
        elif "-" in marker:
            neg.append(sent.iloc[i])
    return PhenomenonContrast(name=name, positive=pos, negative=neg)


# --------------------------------------------------------------------------- #
# Activation extraction (forward hooks on transformer blocks)
# --------------------------------------------------------------------------- #
def _get_submodule(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    cur = root
    for part in dotted.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _assert_fits_on_device(model: torch.nn.Module, device: torch.device,
                           headroom: float = 0.75) -> None:
    """Refuse a load that would not comfortably fit in free VRAM.

    This box is shared: an OOM here does not just fail our run, it can disturb other
    work on the card. So we never "see if it works" -- we require the parameter
    footprint (x2, a slack allowance for activations and workspace) to fit inside
    ``headroom`` of what is actually free, and raise otherwise so the caller skips
    this checkpoint and records it rather than risking the device.

    These are small models (pico-decoder, GPT-2 scale), so in practice this is
    uneventful -- but the check has to exist for the case where it is not.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        free_b, _total_b = torch.cuda.mem_get_info(device.index or 0)
    except Exception:  # pragma: no cover - driver/runtime disagreement
        return
    need_b = sum(p.numel() * p.element_size() for p in model.parameters()) * 2
    if need_b > headroom * free_b:
        raise RuntimeError(
            f"refusing to load: needs ~{need_b/2**30:.1f}GiB but only "
            f"{free_b/2**30:.1f}GiB free on {device} (limit {headroom:.0%})"
        )


def resolve_hidden_dim(model: torch.nn.Module) -> int:
    """Width of a transformer block's residual stream, across config dialects.

    ``config.hidden_size`` is the HF norm, but custom architectures loaded with
    ``trust_remote_code=True`` are free to name it whatever they like and are not
    required to provide an ``attribute_map``. PicoDecoderHFConfig (pico-lm and the
    Beetle families, which are the backbone of this study) exposes ``d_model`` and
    no ``hidden_size`` at all, so reading ``hidden_size`` directly raised
    AttributeError and every pico/Beetle checkpoint was skipped by the grid.

    Falls back to the true width of an actual parameter so an unknown config that
    names the field something else still works instead of aborting the checkpoint.
    """
    cfg = model.config
    for attr in ("hidden_size", "d_model", "n_embd", "hidden_dim", "embed_dim", "model_dim"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return int(val)

    # Last resort: infer from the input embedding matrix [vocab, width].
    try:
        emb = model.get_input_embeddings()
        if emb is not None and getattr(emb, "weight", None) is not None:
            return int(emb.weight.shape[-1])
    except (AttributeError, NotImplementedError, IndexError):
        pass

    raise AttributeError(
        f"could not determine hidden width for {type(cfg).__name__}; "
        "add its width field to resolve_hidden_dim()"
    )


def discover_block_layer_names(model: torch.nn.Module) -> Tuple[List[str], int]:
    """Return (ordered block module paths, num_blocks) for common architectures.

    Handles GPT-2 (``transformer.h.<i>``) and GPT-NeoX / Pythia
    (``gpt_neox.layers.<i>``), plus LLaMA-style (``model.layers.<i>``).
    """
    candidates = [
        ("transformer.h", "transformer"),          # GPT-2, BabyLM
        ("gpt_neox.layers", "gpt_neox"),           # Pythia / GPT-NeoX
        ("model.layers", "model"),                 # LLaMA / Mistral / OLMo
        ("model.decoder.layers", "model.decoder"), # OPT
        ("pico_decoder.layers", "pico_decoder"),   # pico-lm / Beetle (PicoDecoderHF)
    ]
    for block_path, _root in candidates:
        try:
            blocks = _get_submodule(model, block_path)
            n = len(blocks)
            if n > 0:
                return [f"{block_path}.{i}" for i in range(n)], n
        except (AttributeError, TypeError):
            continue

    # Generic fallback: the largest nn.ModuleList of repeated blocks is almost
    # always the transformer stack. Works for custom architectures we don't list.
    best_path, best_n = None, 0
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > best_n:
            # require homogeneous children (same type) to avoid embeddings lists
            child_types = {type(c).__name__ for c in mod}
            if len(child_types) == 1:
                best_path, best_n = mod_name, len(mod)
    if best_path is not None:
        logger.info(f"discover_block_layer_names: generic fallback -> {best_path} ({best_n} blocks)")
        return [f"{best_path}.{i}" for i in range(best_n)], best_n

    raise ValueError(
        "Could not discover transformer blocks; add this architecture to "
        "discover_block_layer_names()."
    )


class ActivationExtractor:
    """Loads a checkpoint and extracts per-block pooled activations."""

    def __init__(
        self,
        model_name: str,
        revision: Optional[str] = None,
        cache_dir: str = ".cache/huggingface",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        # Accept "repo@revision" refs (as emitted by the model zoo / trajectory script)
        if revision is None and "@" in model_name:
            model_name, revision = model_name.split("@", 1)
        self.model_name = model_name
        self.revision = revision
        os.environ.setdefault("HF_HOME", cache_dir)

        logger.info(f"Loading {model_name} (revision={revision})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, revision=revision, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        _assert_fits_on_device(self.model, self.device)
        self.model.to(self.device).eval()

        self.layer_names, self.num_layers = discover_block_layer_names(self.model)
        self.hidden_dim = resolve_hidden_dim(self.model)
        logger.info(f"  {self.num_layers} blocks x {self.hidden_dim} units")

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
        # hidden: [B, T, H], mask: [B, T]
        if pooling == "mean":
            m = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        if pooling == "sum":
            return (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(1)
        # last-token (default, matches syntax-units): last non-pad position
        idx = mask.sum(1).long() - 1  # [B]
        return hidden[torch.arange(hidden.size(0)), idx]

    @torch.no_grad()
    def extract(
        self, sentences: Sequence[str], pooling: str = "last-token", batch_size: int = 8
    ) -> np.ndarray:
        """Return activations of shape [n_sentences, num_layers, hidden_dim]."""
        acts = np.zeros((len(sentences), self.num_layers, self.hidden_dim), dtype=np.float32)
        modules = [_get_submodule(self.model, n) for n in self.layer_names]

        store: "OrderedDict[int, torch.Tensor]" = OrderedDict()

        def make_hook(li):
            def hook(_m, _inp, out):
                store[li] = out[0] if isinstance(out, (tuple, list)) else out
            return hook

        handles = [m.register_forward_hook(make_hook(i)) for i, m in enumerate(modules)]
        try:
            for b0 in range(0, len(sentences), batch_size):
                batch = list(sentences[b0 : b0 + batch_size])
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=128
                ).to(self.device)
                store.clear()
                self.model(**enc)
                for li in range(self.num_layers):
                    pooled = self._pool(
                        store[li].float(), enc["attention_mask"], pooling
                    )  # [B, H]
                    acts[b0 : b0 + len(batch), li] = pooled.cpu().numpy()
        finally:
            for h in handles:
                h.remove()
        return acts


# --------------------------------------------------------------------------- #
# Selectivity + top-k localization  (faithful to syntax-units `localize`)
# --------------------------------------------------------------------------- #
def selectivity_tvalues(
    pos_acts: np.ndarray, neg_acts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch t-test of |positive| vs |negative| per unit.

    Inputs [n, L, H]; returns (t_matrix, p_matrix) of shape [L, H].
    Absolute value matches syntax-units (magnitude of response, sign-agnostic).
    """
    pos = np.abs(pos_acts)
    neg = np.abs(neg_acts)
    t, p = ttest_ind(pos, neg, axis=0, equal_var=False)  # -> [L, H]
    return np.nan_to_num(t), np.nan_to_num(p, nan=1.0)


def topk_mask(t_matrix: np.ndarray, percentage: float) -> np.ndarray:
    """Binary [L, H] mask of the top `percentage`% units by t-value network-wide.

    Mirrors `is_topk` in syntax-units/localize.py.
    """
    L, H = t_matrix.shape
    num_units = int((percentage / 100.0) * H * L)
    num_units = max(1, num_units)
    flat = t_matrix.flatten()
    # rank: rix < num_units selects the largest t-values
    _, rix = np.unique(-flat, return_inverse=True)
    return np.where(rix < num_units, 1, 0).reshape(L, H)


# --------------------------------------------------------------------------- #
# Specialization-vs-spread metrics
# --------------------------------------------------------------------------- #
def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector. 0 = uniform (spread),
    ->1 = concentrated (localized)."""
    x = np.abs(np.asarray(x, dtype=np.float64)).flatten()
    if x.sum() == 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    cum = np.cumsum(xs)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def normalized_entropy(x: np.ndarray) -> float:
    """Shannon entropy of the |value| distribution, normalized to [0,1].
    1 = maximally spread, 0 = all mass on one unit."""
    x = np.abs(np.asarray(x, dtype=np.float64)).flatten()
    s = x.sum()
    if s == 0:
        return 1.0
    p = x / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(x)))


def layer_center_of_mass(t_matrix: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Selectivity-weighted mean layer index (normalized to [0,1]).

    Tracks *where* in depth a phenomenon lives; its trajectory over checkpoints
    tells us whether P migrates to a stable layer band during training."""
    L = t_matrix.shape[0]
    w = np.abs(t_matrix)
    if mask is not None:
        w = w * mask
    per_layer = w.sum(axis=1)
    if per_layer.sum() == 0:
        return float("nan")
    com = float(np.average(np.arange(L), weights=per_layer))
    return com / max(1, L - 1)


@dataclass
class LocalizationResult:
    phenomenon: str
    model_ref: str
    step: Optional[int]
    tokens: Optional[int]
    percentage: float
    t_matrix: np.ndarray  # [L, H]
    mask: np.ndarray  # [L, H] top-k circuit
    num_layers: int
    hidden_dim: int
    metrics: Dict[str, float] = field(default_factory=dict)

    def units_per_layer(self) -> np.ndarray:
        return self.mask.sum(axis=1)

    def to_row(self) -> dict:
        row = {
            "phenomenon": self.phenomenon,
            "model_ref": self.model_ref,
            "step": self.step,
            "tokens": self.tokens,
            "percentage": self.percentage,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
        }
        row.update(self.metrics)
        return row


class CircuitLocalizer:
    """Runs the localizer contrast for one checkpoint and computes metrics."""

    def __init__(self, extractor: ActivationExtractor, percentage: float = 1.0,
                 pooling: str = "last-token", batch_size: int = 8):
        self.ex = extractor
        self.percentage = percentage
        self.pooling = pooling
        self.batch_size = batch_size

    def localize(
        self, contrast: PhenomenonContrast, step: Optional[int] = None,
        tokens: Optional[int] = None,
    ) -> LocalizationResult:
        pos = self.ex.extract(contrast.positive, self.pooling, self.batch_size)
        neg = self.ex.extract(contrast.negative, self.pooling, self.batch_size)
        t, _p = selectivity_tvalues(pos, neg)
        mask = topk_mask(t, self.percentage)

        metrics = {
            # concentration / localization of selectivity across all units
            "gini": gini(t),
            "entropy": normalized_entropy(t),
            # peak strength + how far it sits above the bulk
            "max_t": float(np.max(t)),
            "mean_topk_t": float(t.flatten()[np.argsort(-t.flatten())][: mask.sum()].mean()),
            # depth signature
            "layer_com": layer_center_of_mass(t, mask),
            # how many layers actually host the circuit (spread across depth)
            "n_active_layers": int((mask.sum(axis=1) > 0).sum()),
            "n_units": int(mask.sum()),
        }
        return LocalizationResult(
            phenomenon=contrast.name,
            model_ref=(f"{self.ex.model_name}@{self.ex.revision}" if self.ex.revision
                       else self.ex.model_name),
            step=step,
            tokens=tokens,
            percentage=self.percentage,
            t_matrix=t,
            mask=mask,
            num_layers=self.ex.num_layers,
            hidden_dim=self.ex.hidden_dim,
            metrics=metrics,
        )

    def cross_validation_consistency(
        self, contrast: PhenomenonContrast, num_folds: int = 2, seed: int = 42
    ) -> float:
        """Split-half reliability: fraction of the circuit shared across folds,
        relative to chance. Mirrors syntax-units `cross_validation` (mask AND
        across folds), normalized so 1.0 == perfectly reliable, ~0 == chance."""
        rng = np.random.default_rng(seed)
        n = len(contrast)
        order = rng.permutation(n)
        folds = np.array_split(order, num_folds)
        inter = None
        for f in folds:
            sub = PhenomenonContrast(
                name=contrast.name,
                positive=[contrast.positive[i] for i in f],
                negative=[contrast.negative[i] for i in f],
            )
            res = self.localize(sub)
            inter = res.mask if inter is None else (inter & res.mask)
        k = int((self.percentage / 100.0) * self.ex.hidden_dim * self.ex.num_layers)
        k = max(1, k)
        expected = k * (k / (self.ex.hidden_dim * self.ex.num_layers)) ** (num_folds - 1)
        observed = int(inter.sum())
        return float((observed - expected) / (k - expected)) if k > expected else 0.0


# --------------------------------------------------------------------------- #
# Cross-phenomenon specialization (differentiation / overlap)
# --------------------------------------------------------------------------- #
def jaccard_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = mask_a.astype(bool), mask_b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def selectivity_index(
    t_by_phenomenon: Dict[str, np.ndarray], target: str, mask: np.ndarray
) -> float:
    """Specialization of the target circuit: how much more it responds to its own
    phenomenon than to the others, on the SAME units.

        SI = (t_target - mean_{others} t) / (t_target + mean_{others} t)

    averaged over the circuit's units. High SI = specialized; ~0 = shared."""
    m = mask.astype(bool)
    t_self = np.abs(t_by_phenomenon[target])[m].mean()
    others = [np.abs(t)[m].mean() for k, t in t_by_phenomenon.items() if k != target]
    if not others:
        return float("nan")
    t_other = float(np.mean(others))
    denom = t_self + t_other
    return float((t_self - t_other) / denom) if denom else 0.0


def overlap_matrix(results: Dict[str, LocalizationResult]) -> pd.DataFrame:
    """Pairwise Jaccard overlap of phenomenon circuits (differentiation matrix)."""
    keys = list(results.keys())
    M = pd.DataFrame(np.eye(len(keys)), index=keys, columns=keys)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            v = jaccard_overlap(results[a].mask, results[b].mask)
            M.loc[a, b] = v
            M.loc[b, a] = v
    return M


def specialization_summary(results: Dict[str, LocalizationResult]) -> pd.DataFrame:
    """Per-phenomenon selectivity index + mean cross-phenomenon overlap.

    Lower mean overlap + higher SI over training == progressive differentiation,
    the LM analogue of increasing cortical specialization with age."""
    t_by = {k: r.t_matrix for k, r in results.items()}
    M = overlap_matrix(results)
    rows = []
    for k, r in results.items():
        others = [c for c in M.columns if c != k]
        rows.append(
            {
                "phenomenon": k,
                "selectivity_index": selectivity_index(t_by, k, r.mask),
                "mean_overlap_with_others": float(M.loc[k, others].mean()) if others else 0.0,
                **r.metrics,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Causal validation: ablate the localized circuit and measure the task deficit
# (generic forward-hook port of syntax-units `ablation.py`; no custom modeling)
# --------------------------------------------------------------------------- #
def random_mask_like(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random same-size mask drawn from the *unselected* units (syntax-units
    `random_mask_from_remaining`)."""
    flat = mask.flatten()
    k = int(flat.sum())
    remaining = np.where(flat == 0)[0]
    pick = rng.choice(remaining, size=min(k, remaining.size), replace=False)
    out = np.zeros_like(flat)
    out[pick] = 1
    return out.reshape(mask.shape)


class AblationValidator:
    """Zero- (or mean-) ablate a localized circuit and measure the minimal-pair
    accuracy drop, vs. random same-size ablations. A causal circuit shows a
    larger drop than random (AlKhamissi et al.; syntax-units `ablation.py`)."""

    def __init__(self, extractor: ActivationExtractor):
        self.ex = extractor
        self.model = extractor.model
        self.tokenizer = extractor.tokenizer
        self.device = extractor.device
        self._modules = [_get_submodule(self.model, n) for n in extractor.layer_names]
        self._handles: list = []

    # --- ablation hooks -------------------------------------------------- #
    def _install(self, mask: np.ndarray, mode: str, fill: float) -> None:
        self._clear()
        for li, module in enumerate(self._modules):
            units = np.where(mask[li] > 0)[0]
            if units.size == 0:
                continue
            idx = torch.as_tensor(units, device=self.device, dtype=torch.long)
            fill_val = 0.0 if mode == "zero" else float(fill)

            def hook(_m, _inp, out, idx=idx, fill_val=fill_val):
                is_tuple = isinstance(out, (tuple, list))
                h = out[0] if is_tuple else out
                h = h.clone()
                h[..., idx] = fill_val
                if is_tuple:
                    return (h,) + tuple(out[1:])
                return h

            self._handles.append(module.register_forward_hook(hook))

    def _clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # --- minimal-pair scoring ------------------------------------------- #
    @torch.no_grad()
    def _mean_logprob(self, sentences: Sequence[str], batch_size: int = 8) -> np.ndarray:
        scores = np.zeros(len(sentences), dtype=np.float64)
        for b0 in range(0, len(sentences), batch_size):
            batch = list(sentences[b0 : b0 + batch_size])
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=128).to(self.device)
            logits = self.model(**enc).logits.float()  # [B, T, V]
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            labels = enc["input_ids"][:, 1:]
            tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            m = enc["attention_mask"][:, 1:].float()
            seq = (tok_lp * m).sum(1) / m.sum(1).clamp(min=1)
            scores[b0 : b0 + len(batch)] = seq.cpu().numpy()
        return scores

    def minimal_pair_accuracy(self, contrast: PhenomenonContrast, batch_size: int = 8) -> float:
        """Fraction of pairs where mean-logprob(positive) > mean-logprob(negative)."""
        p = self._mean_logprob(contrast.positive, batch_size)
        n = self._mean_logprob(contrast.negative, batch_size)
        return float((p > n).mean())

    @torch.no_grad()
    def extract_ablated(
        self, sentences: Sequence[str], mask: np.ndarray,
        pooling: str = "mean", batch_size: int = 8, random: bool = False, seed: int = 42,
    ) -> np.ndarray:
        """Pooled activations with `mask` (or a random same-size mask) zero-ablated.

        Used for the causal-alignment test: rebuild the LM RDM with the localized
        circuit knocked out and re-run RSA against the brain."""
        m = random_mask_like(mask, np.random.default_rng(seed)) if random else mask
        self._install(m, "zero", 0.0)
        try:
            return self.ex.extract(sentences, pooling, batch_size)
        finally:
            self._clear()

    # --- the validation ------------------------------------------------- #
    def validate(
        self, contrast: PhenomenonContrast, mask: np.ndarray, n_random: int = 4,
        mode: str = "zero", fill: float = 0.0, seed: int = 42, batch_size: int = 8,
    ) -> Dict[str, float]:
        acc_none = self.minimal_pair_accuracy(contrast, batch_size)

        self._install(mask, mode, fill)
        acc_loc = self.minimal_pair_accuracy(contrast, batch_size)
        self._clear()

        rng = np.random.default_rng(seed)
        rand_accs = []
        for _ in range(n_random):
            self._install(random_mask_like(mask, rng), mode, fill)
            rand_accs.append(self.minimal_pair_accuracy(contrast, batch_size))
            self._clear()
        acc_rand = float(np.mean(rand_accs)) if rand_accs else float("nan")

        return {
            "acc_none": acc_none,
            "acc_localized_ablated": acc_loc,
            "acc_random_ablated": acc_rand,
            "drop_localized": acc_none - acc_loc,
            "drop_random": acc_none - acc_rand,
            # causal specificity: how much more the circuit matters than random units
            "causal_selectivity": (acc_none - acc_loc) - (acc_none - acc_rand),
        }
