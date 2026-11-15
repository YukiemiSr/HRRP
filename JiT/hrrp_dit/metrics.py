"""Distribution and shape metrics for generated HRRP views.

The hand-crafted feature KL and scattering-centre power KL follow the
ADA-GAN evaluation protocol (Song et al., JSTARS 2022) and are reused here so
that the DiT results are directly comparable with the GAN baseline in the
sibling ``Multi_View`` project.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

FEATURE_NAMES = [
    "scattering_center_number",
    "target_length",
    "half_power_scattering_centers",
    "symmetry",
    "dispersibility",
    "fluctuation_power",
    "normalization_power",
    "half_power_ratio",
]
FEATURE_LABELS = [
    "Scattering centres",
    "Target length",
    "Half-power centres",
    "Symmetry",
    "Dispersibility",
    "Fluctuation power",
    "Contrast",
    "Half-power ratio",
]


# ---------------------------------------------------------------------------
# Scattering-centre detection
# ---------------------------------------------------------------------------
def local_maxima(profile: np.ndarray, height: float = 0.0, min_distance: int = 3) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return np.zeros((0,), dtype=np.int64)
    threshold = height * float(p.max())
    idx = np.where((p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] > threshold))[0] + 1
    if idx.size == 0:
        idx = np.array([int(np.argmax(p))], dtype=np.int64)
    order = idx[np.argsort(p[idx])[::-1]]
    chosen: List[int] = []
    for i in order:
        if all(abs(int(i) - j) >= int(min_distance) for j in chosen):
            chosen.append(int(i))
    return np.array(sorted(chosen), dtype=np.int64)


def scattering_centers(profile: np.ndarray, height_ratio: float = 0.1, min_distance: int = 5) -> np.ndarray:
    return local_maxima(profile, height=height_ratio, min_distance=min_distance)


def os_cfar_cells(
    profile: np.ndarray,
    guard: int = 4,
    ref: int = 32,
    alpha: float = 2.5,
    order: float = 0.5,
) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float64).reshape(-1)
    length = p.size
    if length <= 2 * (guard + ref):
        return scattering_centers(p)
    detected = np.zeros(length, dtype=bool)
    min_val = float(p.min())
    padded = np.pad(p, (guard + ref, guard + ref), constant_values=min_val)
    k = int(order * 2 * ref)
    for i in range(guard + ref, guard + ref + length):
        cells = np.concatenate(
            [padded[i - guard - ref : i - guard], padded[i + guard : i + guard + ref]]
        )
        noise = np.partition(cells, k)[k]
        detected[i - guard - ref] = p[i - guard - ref] > alpha * noise
    if not detected.any():
        return scattering_centers(p)
    cells_out: List[int] = []
    i = 0
    while i < length:
        if not detected[i]:
            i += 1
            continue
        j = i
        while j + 1 < length and detected[j + 1]:
            j += 1
        cells_out.append(int(i + np.argmax(p[i : j + 1])))
        i = j + 1
    cells_out = [c for c in cells_out if p[c] >= 0.05 * p.max()]
    return np.array(sorted(cells_out), dtype=np.int64)


# ---------------------------------------------------------------------------
# Eight hand-crafted features
# ---------------------------------------------------------------------------
def handcrafted_features(profile: np.ndarray) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float64).reshape(-1)
    p = p / (np.linalg.norm(p) + 1e-12)
    length = p.size
    peaks = scattering_centers(p, height_ratio=0.1, min_distance=4)
    peak_vals = p[peaks] if peaks.size else np.array([p.max()])
    max_val = float(p.max())

    n_scatter = float(len(peaks))
    target_length = float(peaks[-1] - peaks[0] + 1) if peaks.size >= 2 else 0.0

    half_idx = peaks[peak_vals >= 0.5 * max_val]
    n_half = float(len(half_idx))

    if peaks.size >= 2:
        lo, hi = int(peaks[0]), int(peaks[-1]) + 1
    else:
        lo, hi = 0, length
    mid = (lo + hi) // 2
    e_left = float(np.sum(p[lo:mid] ** 2))
    e_right = float(np.sum(p[mid:hi] ** 2))
    symmetry = float(min(e_left, e_right) / (max(e_left, e_right) + 1e-12))

    dispersibility = float(np.mean(np.diff(peaks))) if peaks.size >= 2 else 0.0

    ratios = []
    for c in peaks:
        lo_n = max(0, int(c) - 3)
        hi_n = min(length, int(c) + 4)
        neigh = np.concatenate([p[lo_n:c], p[int(c) + 1 : hi_n]])
        if neigh.size:
            ratios.append(float(p[c] / (neigh.mean() + 1e-12)))
    fluctuation = float(np.mean(ratios)) if ratios else 0.0

    seg = p[lo:hi] if hi > lo else p
    contrast = float(np.std(seg) / (np.mean(seg) + 1e-12))

    total_power = float(np.sum(p**2)) + 1e-12
    half_power = float(np.sum(p[half_idx] ** 2) / total_power) if half_idx.size else 0.0

    return np.array(
        [n_scatter, target_length, n_half, symmetry, dispersibility, fluctuation, contrast, half_power],
        dtype=np.float64,
    )


def feature_matrix(profiles: np.ndarray) -> np.ndarray:
    return np.stack([handcrafted_features(p) for p in profiles], axis=0)


# ---------------------------------------------------------------------------
# KL divergences
# ---------------------------------------------------------------------------
def histogram_kl(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bins: int = 30,
    discrete: bool = False,
) -> float:
    p_samples = np.asarray(p_samples, dtype=np.float64).ravel()
    q_samples = np.asarray(q_samples, dtype=np.float64).ravel()
    p_samples = p_samples[np.isfinite(p_samples)]
    q_samples = q_samples[np.isfinite(q_samples)]
    if p_samples.size == 0 or q_samples.size == 0:
        return float("nan")
    if discrete:
        lo = int(np.floor(min(p_samples.min(), q_samples.min())))
        hi = int(np.ceil(max(p_samples.max(), q_samples.max()))) + 1
        edges = np.arange(lo - 0.5, hi + 0.5, 1.0)
    else:
        lo = float(min(p_samples.min(), q_samples.min()))
        hi = float(max(p_samples.max(), q_samples.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
    p_hist, _ = np.histogram(p_samples, bins=edges)
    q_hist, _ = np.histogram(q_samples, bins=edges)
    alpha = 1.0
    p_prob = (p_hist + alpha) / (p_hist.sum() + alpha * len(p_hist))
    q_prob = (q_hist + alpha) / (q_hist.sum() + alpha * len(q_hist))
    return float(np.sum(p_prob * np.log(p_prob / q_prob)))


def feature_kl(real_feats: np.ndarray, fake_feats: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    discrete_flags = [True, False, True, False, False, False, False, False]
    for j, name in enumerate(FEATURE_NAMES):
        out[name] = histogram_kl(
            real_feats[:, j], fake_feats[:, j], bins=30, discrete=discrete_flags[j]
        )
    out["mean"] = float(np.nanmean([out[n] for n in FEATURE_NAMES]))
    return out


def scattering_power_kl(
    real: np.ndarray,
    fake: np.ndarray,
    cells: np.ndarray,
    bins: int = 30,
) -> Tuple[float, np.ndarray]:
    real_p = np.asarray(real, dtype=np.float64) ** 2
    fake_p = np.asarray(fake, dtype=np.float64) ** 2
    kls = [histogram_kl(real_p[:, c], fake_p[:, c], bins=bins) for c in np.asarray(cells, dtype=np.int64)]
    kls_arr = np.asarray(kls, dtype=np.float64)
    return float(np.nanmean(kls_arr)), kls_arr


def select_cells_from_real(real: np.ndarray, method: str = "cfar") -> np.ndarray:
    mean_profile = np.mean(real, axis=0)
    if method == "cfar":
        cells = os_cfar_cells(mean_profile)
        if cells.size < 3:
            cells = scattering_centers(mean_profile, height_ratio=0.1, min_distance=6)
    else:
        cells = scattering_centers(mean_profile, height_ratio=0.1, min_distance=6)
    if cells.size == 0:
        cells = np.array([int(np.argmax(mean_profile))], dtype=np.int64)
    return cells


# ---------------------------------------------------------------------------
# Profile-space metrics
# ---------------------------------------------------------------------------
def mean_profile_metrics(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    mu_r = np.mean(real, axis=0).astype(np.float64)
    mu_g = np.mean(fake, axis=0).astype(np.float64)
    denom = (np.linalg.norm(mu_r) * np.linalg.norm(mu_g)) + 1e-12
    cosine = float(np.dot(mu_r, mu_g) / denom)
    rel_l2 = float(np.linalg.norm(mu_g - mu_r) / (np.linalg.norm(mu_r) + 1e-12))
    corr = float(np.corrcoef(mu_r, mu_g)[0, 1]) if mu_r.std() > 1e-12 else float("nan")
    peak_shift = float(abs(int(np.argmax(mu_g)) - int(np.argmax(mu_r))))
    # Per-sample cosine to the real mean: robust to sample-wise speckle.
    fake_norm = fake / (np.linalg.norm(fake, axis=1, keepdims=True) + 1e-12)
    mean_norm = mu_r / (np.linalg.norm(mu_r) + 1e-12)
    sample_cos = float(np.mean(fake_norm @ mean_norm))
    return {
        "mean_cosine": cosine,
        "mean_rel_l2": rel_l2,
        "mean_corr": corr,
        "peak_shift": peak_shift,
        "sample_cosine_to_real_mean": sample_cos,
    }


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    d = aa + bb - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def mmd_rbf(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Optional[float] = None,
    max_n: int = 512,
    seed: int = 0,
) -> float:
    """Biased RBF-kernel MMD; subsamples to ``max_n`` points for speed."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) > max_n:
        x = x[rng.choice(len(x), size=max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), size=max_n, replace=False)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if sigma is None:
        z = np.concatenate([x, y], axis=0)
        if len(z) > 256:
            z = z[rng.choice(len(z), size=256, replace=False)]
        dz = _pairwise_sq_dists(z, z)
        med = np.median(dz[dz > 0])
        sigma = float(np.sqrt(max(med, 1e-12)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
    s2 = 2.0 * sigma**2
    kxx = np.exp(-_pairwise_sq_dists(x, x) / s2).mean()
    kyy = np.exp(-_pairwise_sq_dists(y, y) / s2).mean()
    kxy = np.exp(-_pairwise_sq_dists(x, y) / s2).mean()
    return float(max(kxx + kyy - 2.0 * kxy, 0.0))


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Nan-aware mean of a list of metric dictionaries."""
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float))})
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
