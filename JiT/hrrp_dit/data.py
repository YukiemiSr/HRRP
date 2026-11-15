"""BUAA HRRP parsing, file-level splits and paired conditioning batches.

Each BUAA ``.dat`` record is 6472 bytes::

    <5I 5d i 800d 2I

with fields

    start flags, two time words, target id, yaw, pitch, distance, azimuth,
    course, class label, 800 HRRP amplitudes, end flags.

The **aspect angle** of an HRRP is the angle between the radar line of sight
and the target's body axis.  The BUAA recordings do not store it directly, but
``yaw`` is exactly ``|azimuth - course|`` (verified on the whole dataset), i.e.
the target aspect angle.  We therefore use ``yaw`` as the azimuth/aspect
condition requested by the task and keep the raw ``azimuth``/``course`` fields
for reference.  ``data.angle_field`` can be changed in the config, but yaw is
the physically meaningful conditioning variable for HRRP view synthesis.

Splits are made **per recording file** (80/10/10, stratified by class) so that
temporally adjacent profiles from one flight never straddle two splits.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Recording format
# ---------------------------------------------------------------------------
RECORD_DTYPE = np.dtype(
    [
        ("start_flag1", "<u4"),
        ("start_flag2", "<u4"),
        ("time_hi", "<u4"),
        ("time_lo", "<u4"),
        ("target_id", "<u4"),
        ("yaw", "<f8"),
        ("pitch", "<f8"),
        ("distance", "<f8"),
        ("azimuth", "<f8"),
        ("course", "<f8"),
        ("label", "<i4"),
        ("data", "<f8", (800,)),
        ("end_flag1", "<u4"),
        ("end_flag2", "<u4"),
    ]
)
RECORD_SIZE = RECORD_DTYPE.itemsize  # 6472
START_FLAGS = (0x55AAAA55, 0xAA5555AA)
END_FLAGS = (0xBB6666BB, 0x66BBBBAA)


def parse_dat_file(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Parse one recording.

    Returns
    -------
    x : (N, 800) float32, per-profile L2-normalised amplitudes
    labels : (N,) int32, original BUAA class labels
    meta : dict of (N,) float64 arrays: yaw, pitch, azimuth, course
    """
    rec = np.fromfile(path, dtype=RECORD_DTYPE)
    if rec.size == 0:
        raise RuntimeError(f"No complete records in {path}")
    valid = (
        (rec["start_flag1"] == START_FLAGS[0])
        & (rec["start_flag2"] == START_FLAGS[1])
        & (rec["end_flag1"] == END_FLAGS[0])
        & (rec["end_flag2"] == END_FLAGS[1])
    )
    if not np.all(valid):
        rec = rec[valid]
        if rec.size == 0:
            raise RuntimeError(f"No valid records in {path}")

    x = rec["data"].astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
    labels = rec["label"].astype(np.int32)
    meta = {
        "yaw": rec["yaw"].astype(np.float64),
        "pitch": rec["pitch"].astype(np.float64),
        "azimuth": rec["azimuth"].astype(np.float64),
        "course": rec["course"].astype(np.float64),
    }
    return x, labels, meta


def _angle_from_meta(meta: Dict[str, np.ndarray], field: str) -> np.ndarray:
    if field == "aspect":
        # |azimuth - course| folded to [0, 180]
        d = np.mod(meta["azimuth"] - meta["course"], 360.0)
        return np.minimum(d, 360.0 - d)
    if field not in meta:
        raise ValueError(f"unknown angle field {field!r}")
    return np.mod(meta[field], 360.0)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
def discover_files(
    data_root: str,
    subdir: str = "internal_data",
    subdirs: Optional[Sequence[str]] = None,
) -> List[str]:
    names = list(subdirs) if subdirs else [subdir]
    files: List[str] = []
    for name in names:
        found = sorted(glob.glob(os.path.join(data_root, name, "*.dat")))
        if not found:
            raise FileNotFoundError(
                f"No .dat files under {os.path.join(data_root, name)}"
            )
        files.extend(found)
    return files


@dataclass
class _FileRecord:
    x: np.ndarray
    labels: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    azimuth: np.ndarray
    course: np.ndarray


def _split_files_by_profile_count(
    files: Sequence[str],
    counts: Dict[str, int],
    rng: np.random.Generator,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[str]]:
    """Greedy file-level split that balances the *number of profiles*.

    Recording files differ in size by two orders of magnitude.  Splitting by
    file count (the naive 80/10/10) can leave a small class with almost no test
    profiles.  Large files are placed first into the split with the largest
    remaining capacity; small files fill the gaps.  Splits remain file-level,
    so profiles from one flight never straddle train and test.
    """
    names = ("train", "val", "test")
    out: Dict[str, List[str]] = {k: [] for k in names}
    fl = [files[i] for i in rng.permutation(len(files))]
    if len(fl) < 3:
        out["train"] = fl
        return out

    total = float(sum(counts[f] for f in fl))
    targets = {
        "train": (1.0 - val_ratio - test_ratio) * total,
        "val": val_ratio * total,
        "test": test_ratio * total,
    }
    remaining = dict(targets)
    for f in sorted(fl, key=lambda p: counts[p], reverse=True):
        k = max(names, key=lambda n: remaining[n])
        out[k].append(f)
        remaining[k] -= counts[f]

    # Guarantee that every split has at least one recording.
    for empty in [k for k in names if not out[k]]:
        donor = max(names, key=lambda k: len(out[k]))
        if len(out[donor]) <= 1:
            break
        smallest = min(out[donor], key=lambda p: counts[p])
        out[donor].remove(smallest)
        out[empty].append(smallest)
    return out


def prepare_dataset(
    data_root: str,
    out_dir: str,
    subdir: str = "internal_data",
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    angle_field: str = "yaw",
    split_by: str = "profiles",
    force: bool = False,
    subdirs: Optional[Sequence[str]] = None,
    label_json: Optional[str] = None,
    class_order: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Parse all recordings, assign file-level splits and cache ``.npz`` arrays.

    The stored ``x`` is L2-normalised per profile (values in [0, 1]).  A single
    global amplitude scale ``signal_scale = std(train x)`` is also recorded; the
    diffusion model trains on ``x / signal_scale`` (about unit variance), which
    keeps the Gaussian noise scale matched to the data scale.
    """
    meta_path = os.path.join(out_dir, "meta.json")
    if (
        not force
        and os.path.isfile(meta_path)
        and all(os.path.isfile(os.path.join(out_dir, f"{s}.npz")) for s in ("train", "val", "test"))
    ):
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    files = discover_files(data_root, subdir, subdirs)
    label_path = label_json or os.path.join(data_root, "total_label.json")
    label_names: Dict[str, str] = {}
    if os.path.isfile(label_path):
        with open(label_path, "r", encoding="utf-8") as fh:
            label_names = {str(k): str(v) for k, v in json.load(fh).items()}

    groups: Dict[int, List[str]] = {}
    cache: Dict[str, _FileRecord] = {}
    for path in files:
        x, labels, meta = parse_dat_file(path)
        uniq = np.unique(labels)
        lab = int(uniq[0]) if uniq.size == 1 else int(np.bincount(labels).argmax())
        groups.setdefault(lab, []).append(path)
        cache[path] = _FileRecord(
            x=x,
            labels=labels,
            yaw=meta["yaw"].astype(np.float32),
            pitch=meta["pitch"].astype(np.float32),
            azimuth=meta["azimuth"].astype(np.float32),
            course=meta["course"].astype(np.float32),
        )

    if class_order:
        order = sorted({int(v) for v in class_order})
        missing = sorted(set(groups) - set(order))
        if missing:
            raise ValueError(f"raw labels {missing} are absent from class_order")
        orig_labels = order
    else:
        orig_labels = sorted(groups)
    class_names = [label_names.get(str(l), f"class_{l}") for l in orig_labels]
    label_to_idx = {l: i for i, l in enumerate(orig_labels)}

    rng = np.random.default_rng(seed)
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    counts = {p: len(cache[p].labels) for p in cache}
    for lab in orig_labels:
        if lab not in groups:
            # class_order may reserve slots for classes absent from this subset.
            continue
        fl = sorted(groups[lab])
        if split_by == "files":
            # Legacy behaviour: split by number of recordings.
            fl = [fl[i] for i in rng.permutation(len(fl))]
            n_test = max(1, int(round(len(fl) * test_ratio))) if len(fl) >= 3 else 0
            n_val = max(1, int(round(len(fl) * val_ratio))) if len(fl) >= 3 else 0
            while n_test + n_val >= len(fl):
                if n_val >= n_test and n_val > 0:
                    n_val -= 1
                else:
                    n_test -= 1
            parts = {
                "test": fl[:n_test],
                "val": fl[n_test : n_test + n_val],
                "train": fl[n_test + n_val :],
            }
        elif split_by == "profiles":
            parts = _split_files_by_profile_count(fl, counts, rng, val_ratio, test_ratio)
        else:
            raise ValueError(f"unknown split_by {split_by!r}")
        for k in splits:
            splits[k].extend(parts[k])

    os.makedirs(out_dir, exist_ok=True)
    summary: Dict[str, object] = {
        "data_root": os.path.abspath(data_root),
        "subdir": subdir,
        "subdirs": list(subdirs) if subdirs else [subdir],
        "label_json": os.path.abspath(label_path),
        "seed": seed,
        "length": int(next(iter(cache.values())).x.shape[1]),
        "angle_field": angle_field,
        "angle_max": 180.0,
        "split_by": split_by,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "orig_labels": orig_labels,
        "class_names": class_names,
        "split_files": {},
        "split_records": {},
        "split_class_counts": {},
        "files": {},
    }

    train_x_ref: Optional[np.ndarray] = None
    for split_name, paths in splits.items():
        xs, ys, origs = [], [], []
        yaw, pitch, az, course = [], [], [], []
        fids = []
        names: List[str] = []
        for path in paths:
            r = cache[path]
            n = len(r.labels)
            xs.append(r.x)
            ys.append(np.array([label_to_idx[int(v)] for v in r.labels], dtype=np.int64))
            origs.append(r.labels.astype(np.int32))
            yaw.append(r.yaw)
            pitch.append(r.pitch)
            az.append(r.azimuth)
            course.append(r.course)
            fids.append(np.full(n, len(names), dtype=np.int32))
            names.append(os.path.basename(path))

        if xs:
            arr = dict(
                x=np.concatenate(xs),
                y=np.concatenate(ys),
                orig_label=np.concatenate(origs),
                yaw=np.concatenate(yaw),
                pitch=np.concatenate(pitch),
                azimuth=np.concatenate(az),
                course=np.concatenate(course),
                file_id=np.concatenate(fids),
            )
        else:
            length = int(summary["length"])
            arr = dict(
                x=np.zeros((0, length), np.float32),
                y=np.zeros((0,), np.int64),
                orig_label=np.zeros((0,), np.int32),
                yaw=np.zeros((0,), np.float32),
                pitch=np.zeros((0,), np.float32),
                azimuth=np.zeros((0,), np.float32),
                course=np.zeros((0,), np.float32),
                file_id=np.zeros((0,), np.int32),
            )
        if split_name == "train":
            train_x_ref = arr["x"]
        np.savez(os.path.join(out_dir, f"{split_name}.npz"), files=np.array(names, dtype=object), **arr)
        summary["split_files"][split_name] = names
        summary["split_records"][split_name] = int(len(arr["y"]))
        summary["split_class_counts"][split_name] = np.bincount(
            arr["y"], minlength=len(orig_labels)
        ).tolist()
        summary["files"][split_name] = len(names)
        print(
            f"[prepare] {split_name:5s}: {len(names):4d} files | "
            f"{len(arr['y']):6d} profiles"
        )

    assert train_x_ref is not None and train_x_ref.size > 0
    signal_scale = float(np.std(train_x_ref))
    summary["signal_scale"] = signal_scale
    summary["train_std"] = signal_scale

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[prepare] signal_scale (std of unit-L2 train x) = {signal_scale:.6f}")
    return summary


# ---------------------------------------------------------------------------
# Split container
# ---------------------------------------------------------------------------
@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray
    orig_label: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    azimuth: np.ndarray
    course: np.ndarray
    file_id: np.ndarray
    files: List[str]
    angle_field: str = "yaw"

    def __len__(self) -> int:
        return len(self.y)

    @property
    def angle(self) -> np.ndarray:
        if self.angle_field == "aspect":
            d = np.mod(self.azimuth - self.course, 360.0)
            return np.minimum(d, 360.0 - d).astype(np.float32)
        return getattr(self, self.angle_field)


def load_split(root: str, name: str) -> Split:
    path = os.path.join(root, f"{name}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} missing; run scripts/prepare_data.py first")
    meta_path = os.path.join(root, "meta.json")
    angle_field = "yaw"
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            angle_field = str(json.load(fh).get("angle_field", "yaw"))
    z = np.load(path, allow_pickle=True)
    return Split(
        x=z["x"].astype(np.float32),
        y=z["y"].astype(np.int64),
        orig_label=z["orig_label"].astype(np.int32),
        yaw=z["yaw"].astype(np.float32),
        pitch=z["pitch"].astype(np.float32),
        azimuth=z["azimuth"].astype(np.float32),
        course=z["course"].astype(np.float32),
        file_id=z["file_id"].astype(np.int32),
        files=[str(f) for f in z["files"].tolist()],
        angle_field=angle_field,
    )


def subset(split: Split, mask: np.ndarray) -> Split:
    mask = np.asarray(mask, dtype=bool)
    return Split(
        x=split.x[mask],
        y=split.y[mask],
        orig_label=split.orig_label[mask],
        yaw=split.yaw[mask],
        pitch=split.pitch[mask],
        azimuth=split.azimuth[mask],
        course=split.course[mask],
        file_id=split.file_id[mask],
        files=split.files,
        angle_field=split.angle_field,
    )


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------
def near_angles(angles: np.ndarray, centers: Sequence[float], width: float) -> np.ndarray:
    a = np.asarray(angles, dtype=np.float32)[:, None]
    c = np.asarray(list(centers), dtype=np.float32)[None, :]
    return (np.abs(a - c) <= float(width)).any(axis=1)


def class_mask(split: Split, class_idx: int) -> np.ndarray:
    return split.y == int(class_idx)


# ---------------------------------------------------------------------------
# Paired training batcher
# ---------------------------------------------------------------------------
class PairedBatcher:
    """Sample ``(x_src, a_src) -> x_tgt at a_tgt`` tuples in memory.

    For every target profile ``x_tgt`` we draw a source profile whose aspect
    differs by ``[min_source_delta, max_source_delta]`` degrees.  Restricting
    the angular gap to a local window is important: a source 100 degrees away
    carries little information about the target, so the diffusion objective
    prefers to ignore it.  With probability ``p_same_file`` the source comes
    from the **same recording** (same physical target and flight), otherwise
    from another recording of the same class.  ``p_self`` keeps exact
    self-reconstruction pairs at ``a_tgt = a_src``.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        angle: np.ndarray,
        signal_scale: float,
        batch_size: int = 128,
        min_source_delta: float = 2.0,
        max_source_delta: Optional[float] = 20.0,
        p_self: float = 0.1,
        p_same_file: float = 0.0,
        file_id: Optional[np.ndarray] = None,
        target_mode: str = "absolute",
        residual_scale: float = 1.0,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        if target_mode not in ("absolute", "residual"):
            raise ValueError(f"unknown target_mode {target_mode!r}")
        self.x = np.ascontiguousarray(x, dtype=np.float32)  # unit-L2 profiles
        self.signal_scale = float(signal_scale)
        self.target_mode = target_mode
        self.residual_scale = float(residual_scale)
        self.y = np.asarray(y, dtype=np.int64)
        self.angle = np.asarray(angle, dtype=np.float32)
        self.file_id = None if file_id is None else np.asarray(file_id, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.min_source_delta = float(min_source_delta)
        self.max_source_delta = None if max_source_delta is None else float(max_source_delta)
        self.p_self = float(p_self)
        self.p_same_file = float(p_same_file)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self.class_pools = {
            int(c): np.where(self.y == c)[0] for c in np.unique(self.y)
        }
        self.file_pools: Dict[int, np.ndarray] = {}
        if self.file_id is not None:
            self.file_pools = {
                int(f): np.where(self.file_id == f)[0] for f in np.unique(self.file_id)
            }
        self._sorted: Dict[tuple, tuple] = {}
        for key, pool in list(self.class_pools.items()):
            self._sorted[("class", key)] = (pool[np.argsort(self.angle[pool])], np.sort(self.angle[pool]))
        for key, pool in self.file_pools.items():
            self._sorted[("file", key)] = (pool[np.argsort(self.angle[pool])], np.sort(self.angle[pool]))

    def __len__(self) -> int:
        return len(self.x)

    def _candidates(self, kind: str, key: int, a: float) -> np.ndarray:
        idx, ang = self._sorted[(kind, key)]
        lo = a - self.max_source_delta if self.max_source_delta is not None else -np.inf
        hi = a - self.min_source_delta
        ranges = [idx[np.searchsorted(ang, lo) : np.searchsorted(ang, hi, side="right")]]
        lo2 = a + self.min_source_delta
        hi2 = a + self.max_source_delta if self.max_source_delta is not None else np.inf
        ranges.append(idx[np.searchsorted(ang, lo2) : np.searchsorted(ang, hi2, side="right")])
        return np.concatenate(ranges) if ranges else np.zeros((0,), np.int64)

    def _sample_source(self, tgt: np.ndarray) -> np.ndarray:
        src = tgt.copy()
        b = len(tgt)
        self_mask = self.rng.random(b) < self.p_self
        same_pref = np.zeros(b, dtype=bool)
        if self.file_id is not None and self.p_same_file > 0:
            same_pref = (self.rng.random(b) < self.p_same_file) & (~self_mask)
        for k in range(b):
            if self_mask[k]:
                continue
            i = int(tgt[k])
            a = float(self.angle[i])
            picked = -1
            if same_pref[k]:
                cand = self._candidates("file", int(self.file_id[i]), a)
                if cand.size:
                    picked = int(cand[self.rng.integers(0, cand.size)])
            if picked < 0:
                cand = self._candidates("class", int(self.y[i]), a)
                if cand.size:
                    picked = int(cand[self.rng.integers(0, cand.size)])
            if picked < 0:
                # Fallback: any other profile of the same class.
                pool = self.class_pools[int(self.y[i])]
                pool = pool[pool != i]
                picked = int(pool[self.rng.integers(0, len(pool))]) if pool.size else i
            src[k] = picked
        return src

    def sample(self, n: Optional[int] = None) -> Dict[str, torch.Tensor]:
        n = self.batch_size if n is None else int(n)
        tgt = self.rng.integers(0, len(self.x), size=n)
        src = self._sample_source(tgt)
        x_tgt = self.x[tgt]
        x_src = self.x[src]
        cond = x_src / self.signal_scale
        if self.target_mode == "absolute":
            var = x_tgt / self.signal_scale
        else:
            var = (x_tgt - x_src) / self.residual_scale
        out = {
            "x": torch.from_numpy(var).unsqueeze(1).to(self.device),
            "x_cond": torch.from_numpy(cond).unsqueeze(1).to(self.device),
            "x_src_unit": torch.from_numpy(x_src).unsqueeze(1).to(self.device),
            "a_src": torch.from_numpy(self.angle[src]).to(self.device),
            "a_tgt": torch.from_numpy(self.angle[tgt]).to(self.device),
            "y": torch.from_numpy(self.y[tgt]).to(self.device),
            "idx_tgt": torch.from_numpy(tgt).to(self.device),
            "idx_src": torch.from_numpy(src).to(self.device),
        }
        return out


class ClassBatcher:
    """Random profiles with class labels and target azimuths (class-conditional).

    Used only when ``model.cond_mode == 'class'``: the target is an absolute
    HRRP and the condition is ``(y, a_tgt)``.  ``x_cond`` is a zero tensor so
    the training/generation interfaces stay unchanged; the class model ignores
    it and interprets ``a_src`` as the class label.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        angle: np.ndarray,
        signal_scale: float,
        batch_size: int = 128,
        seed: int = 0,
        device: str | torch.device = "cpu",
        class_balanced: bool = False,
    ):
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.angle = np.asarray(angle, dtype=np.float32)
        self.signal_scale = float(signal_scale)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self.class_balanced = bool(class_balanced)
        self.num_classes = int(self.y.max()) + 1 if len(self.y) else 0
        if self.class_balanced:
            order = np.argsort(self.y, kind="stable")
            starts = np.searchsorted(self.y[order], np.arange(self.num_classes))
            ends = np.searchsorted(
                self.y[order], np.arange(self.num_classes), side="right"
            )
            self._sorted_idx = order.astype(np.int64)
            self._class_starts = starts
            self._class_counts = ends - starts
            if bool((self._class_counts <= 0).any()):
                raise ValueError(
                    "class_balanced needs every class present in the split."
                )

    def __len__(self) -> int:
        return len(self.x)

    def sample(self, n: Optional[int] = None) -> Dict[str, torch.Tensor]:
        n = self.batch_size if n is None else int(n)
        if self.class_balanced:
            # Uniform over classes, then uniform over that class's records.
            cls = self.rng.integers(0, self.num_classes, size=n)
            pos = self.rng.integers(0, self._class_counts[cls])
            idx = self._sorted_idx[self._class_starts[cls] + pos]
        else:
            idx = self.rng.integers(0, len(self.x), size=n)
        var = self.x[idx] / self.signal_scale
        return {
            "x": torch.from_numpy(var).unsqueeze(1).to(self.device),
            "x_cond": torch.zeros(n, 1, self.x.shape[1], dtype=torch.float32, device=self.device),
            "a_src": torch.from_numpy(self.y[idx].astype(np.float32)).to(self.device),
            "a_tgt": torch.from_numpy(self.angle[idx]).to(self.device),
            "y": torch.from_numpy(self.y[idx]).to(self.device),
            "idx_tgt": torch.from_numpy(idx).to(self.device),
            "idx_src": torch.from_numpy(idx).to(self.device),
        }


def make_class_val_batch(
    split: Split,
    signal_scale: float,
    n: int = 1024,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Dict[str, torch.Tensor]:
    batcher = ClassBatcher(
        split.x, split.y, split.angle,
        signal_scale=signal_scale, batch_size=n, seed=seed, device=device,
    )
    return batcher.sample(n)


class FixedPairs:
    """Fixed source/target pairs used for validation and evaluation."""

    def __init__(
        self,
        x_src: np.ndarray,
        x_ref: np.ndarray,
        a_src: np.ndarray,
        a_tgt: np.ndarray,
        y: np.ndarray,
        idx_src: Optional[np.ndarray] = None,
        idx_ref: Optional[np.ndarray] = None,
    ):
        self.x_src = x_src
        self.x_ref = x_ref
        self.a_src = a_src
        self.a_tgt = a_tgt
        self.y = y
        self.idx_src = idx_src
        self.idx_ref = idx_ref


def make_val_pairs(
    split: Split,
    signal_scale: float,
    n: int = 1024,
    min_source_delta: float = 2.0,
    max_source_delta: Optional[float] = 20.0,
    p_self: float = 0.1,
    p_same_file: float = 0.0,
    target_mode: str = "absolute",
    residual_scale: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Dict[str, torch.Tensor]:
    """A fixed paired batch in scaled units for validation loss."""
    batcher = PairedBatcher(
        split.x,
        split.y,
        split.angle,
        signal_scale=signal_scale,
        batch_size=n,
        min_source_delta=min_source_delta,
        max_source_delta=max_source_delta,
        p_self=p_self,
        p_same_file=p_same_file,
        file_id=split.file_id,
        target_mode=target_mode,
        residual_scale=residual_scale,
        seed=seed,
        device=device,
    )
    return batcher.sample(n)


# ---------------------------------------------------------------------------
# Evaluation sets for view synthesis
# ---------------------------------------------------------------------------
def _name_to_idx(class_names: Sequence[str], names: Optional[Sequence[str]]) -> List[int]:
    if names is None:
        return list(range(len(class_names)))
    out = []
    for n in names:
        n = str(n)
        if n.isdigit():
            out.append(int(n))
        elif n in class_names:
            out.append(class_names.index(n))
        else:
            raise ValueError(f"unknown class {n!r}; known: {list(class_names)}")
    return sorted(set(out))


def build_generation_set(
    split: Split,
    class_names: Sequence[str],
    source_angles: Sequence[float],
    source_width: float,
    target_angles: Sequence[float],
    target_width: float,
    n_per_angle: int,
    n_real: int,
    min_source_delta: float = 2.0,
    max_source_delta: Optional[float] = 20.0,
    require_same_file: bool = False,
    classes: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Build the view-synthesis protocol.

    For each class and each target angle, real test profiles from the acquired
    azimuths are used as source conditions, and real test profiles around the
    target azimuth are kept as the reference distribution.

    If ``require_same_file`` is true, sources are restricted to recordings that
    also contain a target-angle reference, i.e. source and target come from the
    same physical target/flight.
    """
    rng = np.random.default_rng(seed)
    chosen = _name_to_idx(class_names, classes)
    angles = split.angle

    src_x, src_a, tgt_a, ys, src_idx = [], [], [], [], []
    real_x, real_a, real_y = [], [], []
    counts: Dict[str, Dict[str, int]] = {}

    for c in chosen:
        cm = split.y == c
        src_mask_all = cm & near_angles(angles, source_angles, source_width)
        src_pool = np.where(src_mask_all)[0]
        cname = class_names[c]
        counts[cname] = {}
        for a in target_angles:
            ref_mask = cm & (np.abs(angles - float(a)) <= float(target_width))
            ref_pool = np.where(ref_mask)[0]
            if src_pool.size == 0 or ref_pool.size == 0:
                counts[cname][f"{float(a):g}"] = 0
                continue
            # Keep the source within the angular range seen during training.
            delta = np.abs(angles[src_pool] - float(a))
            ok = delta >= float(min_source_delta)
            if max_source_delta is not None:
                ok &= delta <= float(max_source_delta)
            pool = src_pool[ok] if ok.any() else src_pool
            if require_same_file:
                ref_files = np.unique(split.file_id[ref_pool])
                same = np.isin(split.file_id[pool], ref_files)
                if same.any():
                    pool = pool[same]
            take = rng.choice(pool, size=int(n_per_angle), replace=len(pool) < int(n_per_angle))
            src_x.append(split.x[take])
            src_a.append(angles[take])
            src_idx.append(take)
            tgt_a.append(np.full(len(take), float(a), dtype=np.float32))
            ys.append(np.full(len(take), c, dtype=np.int64))

            ref_take = ref_pool
            if n_real > 0 and len(ref_take) > int(n_real):
                ref_take = rng.choice(ref_take, size=int(n_real), replace=False)
            real_x.append(split.x[ref_take])
            real_a.append(np.full(len(ref_take), float(a), dtype=np.float32))
            real_y.append(np.full(len(ref_take), c, dtype=np.int64))
            counts[cname][f"{float(a):g}"] = int(len(ref_take))

    return dict(
        x_src=np.concatenate(src_x) if src_x else np.zeros((0, split.x.shape[1]), np.float32),
        x_real=np.concatenate(real_x) if real_x else np.zeros((0, split.x.shape[1]), np.float32),
        a_src=np.concatenate(src_a) if src_a else np.zeros((0,), np.float32),
        a_tgt=np.concatenate(tgt_a) if tgt_a else np.zeros((0,), np.float32),
        y=np.concatenate(ys) if ys else np.zeros((0,), np.int64),
        idx_src=np.concatenate(src_idx) if src_idx else np.zeros((0,), np.int64),
        real_a_tgt=np.concatenate(real_a) if real_a else np.zeros((0,), np.float32),
        real_y=np.concatenate(real_y) if real_y else np.zeros((0,), np.int64),
        class_names=list(class_names),
        target_angles=[float(a) for a in target_angles],
        source_angles=[float(a) for a in source_angles],
        counts=counts,
    )


def build_self_recon_set(
    split: Split,
    class_names: Sequence[str],
    angle_centers: Sequence[float],
    width: float,
    n_per_angle: int,
    classes: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    chosen = _name_to_idx(class_names, classes)
    angles = split.angle
    xs, src_a, tgt_a, ys, idx = [], [], [], [], []
    for c in chosen:
        cm = split.y == c
        for a in angle_centers:
            pool = np.where(cm & (np.abs(angles - float(a)) <= float(width)))[0]
            if pool.size == 0:
                continue
            take = rng.choice(pool, size=int(n_per_angle), replace=len(pool) < int(n_per_angle))
            xs.append(split.x[take])
            src_a.append(angles[take])
            tgt_a.append(angles[take].copy())
            ys.append(np.full(len(take), c, dtype=np.int64))
            idx.append(take)
    return dict(
        x_src=np.concatenate(xs) if xs else np.zeros((0, split.x.shape[1]), np.float32),
        a_src=np.concatenate(src_a) if src_a else np.zeros((0,), np.float32),
        a_tgt=np.concatenate(tgt_a) if tgt_a else np.zeros((0,), np.float32),
        y=np.concatenate(ys) if ys else np.zeros((0,), np.int64),
        idx_src=np.concatenate(idx) if idx else np.zeros((0,), np.int64),
        class_names=list(class_names),
    )


def build_sweep_set(
    split: Split,
    class_names: Sequence[str],
    class_name: str,
    source_angle: float,
    source_width: float,
    target_angles: Sequence[float],
    n_sources: int,
    seed: int = 0,
) -> Dict[str, object]:
    """Fixed source profiles swept over a dense grid of target angles."""
    rng = np.random.default_rng(seed)
    classes = [int(class_name)] if str(class_name).isdigit() else [class_names.index(str(class_name))]
    c = classes[0]
    angles = split.angle
    pool = np.where((split.y == c) & (np.abs(angles - float(source_angle)) <= float(source_width)))[0]
    if pool.size == 0:
        raise ValueError(f"no profiles of {class_name} near {source_angle} deg")
    take = rng.choice(pool, size=min(int(n_sources), len(pool)), replace=False)
    x_src = split.x[take]
    a_src = angles[take]
    n = len(take)
    x_rep = np.repeat(x_src, len(target_angles), axis=0)
    a_src_rep = np.repeat(a_src, len(target_angles))
    a_tgt = np.tile(np.asarray(target_angles, dtype=np.float32), n)
    y = np.full(n * len(target_angles), c, dtype=np.int64)
    return dict(
        x_src=x_rep,
        a_src=a_src_rep,
        a_tgt=a_tgt,
        y=y,
        class_names=list(class_names),
        class_name=class_names[c],
        source_angle=float(source_angle),
        sweep_angles=[float(a) for a in target_angles],
    )


# ---------------------------------------------------------------------------
# Class-conditional (label + target azimuth) evaluation sets
# ---------------------------------------------------------------------------
def build_class_generation_set(
    split: Split,
    class_names: Sequence[str],
    target_angles: Sequence[float],
    target_width: float,
    n_per_angle: int,
    n_real: int,
    classes: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Build ``(y, a_tgt) -> HRRP`` evaluation grids for the class model.

    There is no source HRRP: each generated entry is one requested
    (class, target azimuth) pair.  Real test profiles in the target window are
    stored as the reference distribution.
    """
    rng = np.random.default_rng(seed)
    chosen = _name_to_idx(class_names, classes)
    angles = split.angle
    ys, tgt_a, real_x, real_a, real_y = [], [], [], [], []
    counts: Dict[str, Dict[str, int]] = {}
    for c in chosen:
        cm = split.y == c
        cname = class_names[c]
        counts[cname] = {}
        for a in target_angles:
            ref_mask = cm & (np.abs(angles - float(a)) <= float(target_width))
            ref_pool = np.where(ref_mask)[0]
            if len(ref_pool) < 4:
                counts[cname][f"{float(a):g}"] = 0
                continue
            ys.append(np.full(int(n_per_angle), c, dtype=np.int64))
            tgt_a.append(np.full(int(n_per_angle), float(a), dtype=np.float32))
            ref_take = ref_pool
            if n_real > 0 and len(ref_take) > int(n_real):
                ref_take = rng.choice(ref_take, size=int(n_real), replace=False)
            real_x.append(split.x[ref_take])
            real_a.append(np.full(len(ref_take), float(a), dtype=np.float32))
            real_y.append(np.full(len(ref_take), c, dtype=np.int64))
            counts[cname][f"{float(a):g}"] = int(len(ref_take))
    return dict(
        y=np.concatenate(ys) if ys else np.zeros((0,), np.int64),
        a_tgt=np.concatenate(tgt_a) if tgt_a else np.zeros((0,), np.float32),
        x_real=np.concatenate(real_x) if real_x else np.zeros((0, split.x.shape[1]), np.float32),
        real_a_tgt=np.concatenate(real_a) if real_a else np.zeros((0,), np.float32),
        real_y=np.concatenate(real_y) if real_y else np.zeros((0,), np.int64),
        class_names=list(class_names),
        target_angles=[float(a) for a in target_angles],
        counts=counts,
    )


def build_class_sweep_set(
    split: Split,
    class_names: Sequence[str],
    class_name: str,
    target_angles: Sequence[float],
    n_per_angle: int,
    seed: int = 0,
) -> Dict[str, object]:
    """Class-conditional angle sweep (no source): repeated label+angle pairs."""
    c = int(class_name) if str(class_name).isdigit() else class_names.index(str(class_name))
    y = []
    a = []
    for ang in target_angles:
        y.append(np.full(int(n_per_angle), c, dtype=np.int64))
        a.append(np.full(int(n_per_angle), float(ang), dtype=np.float32))
    return dict(
        y=np.concatenate(y),
        a_tgt=np.concatenate(a),
        class_name=class_names[c],
        sweep_angles=[float(x) for x in target_angles],
        class_names=list(class_names),
    )
