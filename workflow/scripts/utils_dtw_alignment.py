import colorsys
import hashlib
import logging
import math
import os
import random
import time
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from dtaidistance import dtw as dtaid_dtw
from fastdtw import fastdtw
from joblib import Parallel, delayed
from matplotlib import colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from ruptures import Pelt
from scipy import stats
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import resample
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from tslearn.barycenters import (
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_subgradient,
)


from utils_classification import extract_interpretable_features
from utils_dtw import (
    dtw_distance_numba,
    dtw_features_to_templates,
    dtw_path,
    dtw_path_numba,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nanopore-peptide-classifier")


def remove_zero_padding(signal):
    nz = np.nonzero(signal)[0]
    return signal[: nz[-1] + 1] if nz.size > 0 else np.array([])


def normalize_signal(sig, norm):
    if not norm:
        return sig
    med = np.median(sig)
    mad = np.median(np.abs(sig - med))
    return (sig - med) / mad if mad != 0 else sig - med


def detect_template_change_points(template, pen):
    # Using PELT on the template
    bkps = Pelt(model="l2").fit(template).predict(pen=pen)
    return bkps


def project_cpts_to_trace(template, trace, cpts):
    # Compute DTW path from template to trace
    _, path = fastdtw(template, trace, dist=2)
    # invert mapping: for each template index, collect trace indices
    ref_len = len(template)
    buckets = [[] for _ in range(ref_len)]
    for i_ref, j_tr in path:
        if i_ref < ref_len:
            buckets[i_ref].append(j_tr)
    # project each change point by median of mapped trace indices
    trace_cpts = []
    for cp in cpts:
        if cp < len(buckets) and buckets[cp]:
            trace_cpts.append(int(np.median(buckets[cp])))
    # ensure sorted and unique, add end
    trace_cpts = sorted(set(trace_cpts))
    if trace_cpts[-1] != len(trace):
        trace_cpts.append(len(trace))
    return trace_cpts


def extract_event_medians(signal, cpts):
    medians = []
    prev = 0
    for cp in cpts:
        segment = signal[prev:cp]
        if segment.size:
            medians.append(np.median(segment))
        prev = cp
    return np.array(medians)


def plot_dtw_event_median_profiles(
    X_valid, y_true, y_pred, output_path, target_classes, params
):
    """
    1) Build a Soft-DTW barycenter template from all correct signals
    2) Detect change points on template
    3) Project those change points to each trace via DTW
    4) Extract event medians per trace
    5) Plot per-class profiles (background + median), cross-class diffs, alignment counts
    """
    os.makedirs(output_path, exist_ok=True)
    for sub in ["example_projections", "profiles", "counts"]:
        os.makedirs(os.path.join(output_path, sub), exist_ok=True)

    # 1) collect correctly classified, trimmed, normalized traces
    mask = y_true == y_pred
    correct_traces = []
    classes = []
    for sig, yt, yp in zip(X_valid, y_true, y_pred):
        if yt == yp and yt in target_classes:
            trace = remove_zero_padding(sig.squeeze())
            if trace.size:
                trace = normalize_signal(trace, params.get("normalize", False))
                correct_traces.append(trace)
                classes.append(yt)

    # 2) compute barycenter template
    # pad sequences to max length by edge padding
    max_len = max(len(t) for t in correct_traces)
    padded = np.array(
        [np.pad(t, (0, max_len - len(t)), mode="edge") for t in correct_traces]
    )
    template = dtw_barycenter_averaging_subgradient(padded, max_iter=50, tol=1e-3)

    # 3) detect change-points on template
    cpts_template = detect_template_change_points(template, pen=params["penalty"])

    # 4) project to each trace and extract medians
    class_medians = {c: [] for c in target_classes}
    alignment_counts = {
        c: np.zeros(len(cpts_template), dtype=int) for c in target_classes
    }

    for trace, cls in zip(correct_traces, classes):
        # pad trace to template length
        trace_p = np.pad(trace, (0, len(template) - len(trace)), mode="edge")
        # project
        trace_cpts = project_cpts_to_trace(template, trace_p, cpts_template)
        med = extract_event_medians(trace_p, trace_cpts)
        if med.size == len(cpts_template):
            class_medians[cls].append(med)
            # count contributions
            for i, _ in enumerate(cpts_template):
                alignment_counts[cls][i] += 1

    # 5) plotting
    x = np.arange(1, len(cpts_template) + 1)
    # a) background+median
    plt.figure(figsize=(10, 6))
    for cls in target_classes:
        data = np.array(class_medians[cls])
        if data.size == 0:
            continue
        for row in data:
            plt.step(x, row, where="mid", color="gray", alpha=0.1)
        med = data.mean(axis=0)
        sd = data.std(axis=0)
        plt.step(x, med, where="mid", label=f"{cls} (n={len(data)})")
        plt.fill_between(x, med - sd, med + sd, alpha=0.2, step="mid")
    plt.title("DTW-Template Aligned Event Profiles")
    plt.xlabel("Event Index")
    plt.ylabel("Current")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "profiles", "dtw_template_profiles.png"))
    plt.close()

    # b) median only
    plt.figure(figsize=(10, 6))
    for cls in target_classes:
        data = np.array(class_medians[cls])
        if data.size == 0:
            continue
        med = data.mean(axis=0)
        sd = data.std(axis=0)
        plt.step(x, med, where="mid", label=f"{cls} (n={len(data)})")
        plt.fill_between(x, med - sd, med + sd, alpha=0.2, step="mid")
    plt.title("DTW-Template Event Medians (Median ± SD)")
    plt.xlabel("Event Index")
    plt.ylabel("Current")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "profiles", "dtw_template_median_only.png"))
    plt.close()

    # c) cross-class difference
    plt.figure(figsize=(10, 6))
    base = np.array(class_medians[target_classes[0]]).mean(axis=0)
    for cls in target_classes[1:]:
        data = np.array(class_medians[cls])
        if data.size == 0:
            continue
        diff = data.mean(axis=0) - base
        plt.step(x, diff, where="mid", label=f"{cls} - {target_classes[0]}")
    plt.axhline(0, linestyle="--", color="k")
    plt.title("Cross-Class Median Differences (Template-Based)")
    plt.xlabel("Event Index")
    plt.ylabel("Delta Current")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "profiles", "dtw_template_cross_class_diff.png")
    )
    plt.close()

    # d) per-position counts
    plt.figure(figsize=(10, 4))
    for cls in target_classes:
        cnt = alignment_counts[cls]
        plt.step(x, cnt, where="mid", label=cls)
    plt.title("Alignment Contribution Counts per Event Index")
    plt.xlabel("Event Index")
    plt.ylabel("Number of Traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "counts", "dtw_template_alignment_counts.png")
    )
    plt.close()

    logger.info(f"Plotted template-based DTW event profiles in {output_path}")


def downsample_signal(sig, factor):
    if factor <= 1:
        return sig
    n = len(sig)
    m = n // factor
    if m < 2:
        return sig.copy()
    sig = sig[: m * factor]
    sig = sig.reshape((m, factor)).mean(axis=1)
    return sig


def _ensure_1d_float(arr):
    a = np.asarray(arr)
    if a.size == 0:
        return np.array([], dtype=float)
    # If an array of shape (n,1) or (1,n) or higher dims, flatten to 1D
    if a.ndim > 1:
        a = a.flatten()
    # If dtype is object, try to coerce elements -> float
    try:
        a = a.astype(float)
    except Exception:
        # last resort: convert elementwise
        a = np.array([float(x) for x in a.ravel()], dtype=float)
    return a


def dtw_distance(a, b, window=None):
    a = _ensure_1d_float(a)
    b = _ensure_1d_float(b)

    if a.size == 0 or b.size == 0:
        return float("inf")

    d, _ = fastdtw(a, b, dist=2, radius=3)
    return float(d)


def compute_class_barycenters(
    X,
    y,
    max_samples_per_class=200,
    downsample_factor=1,
    random_state=42,
    window_pct=0.2,
):
    """
    X: ndarray shape (N, L) or list of 1D arrays (variable length allowed if using dtw)
    y: labels (N,)
    returns dict label -> centroid (1D np.array), and example indices used
    """
    rng = np.random.RandomState(random_state)
    labels = np.unique(y)
    centroids = {}
    used_idx = {}
    X_list = [np.asarray(x) for x in X]
    for lbl in labels:
        idxs = np.where(y == lbl)[0]
        if len(idxs) == 0:
            continue
        sel = (
            idxs
            if len(idxs) <= max_samples_per_class
            else rng.choice(idxs, max_samples_per_class, replace=False)
        )
        samples = [downsample_signal(X_list[i], downsample_factor) for i in sel]
        # equalize lengths by padding/truncation for tslearn barycenter
        lengths = [len(s) for s in samples]
        Lm = int(np.median(lengths))
        samples_eq = [
            (
                s
                if len(s) == Lm
                else np.interp(np.linspace(0, len(s) - 1, Lm), np.arange(len(s)), s)
            )
            for s in samples
        ]
        samples_eq = np.stack(samples_eq)  # shape (M, Lm)
        centroid = None
        centroid = dtw_barycenter_averaging(samples_eq, max_iter=20)
        centroids[lbl] = centroid
        used_idx[lbl] = sel
    return centroids, used_idx


def distances_to_centroid(
    X_list, centroid, downsample_factor=1, window_pct=0.2, n_jobs=1
):
    """
    X_list: list of 1-D signals (arrays or lists)
    centroid: 1-D array
    returns numpy array of distances (same order)
    """
    # precompute centroid downsampled and window
    centroid_ds = downsample_signal(centroid, downsample_factor)
    if centroid_ds.size == 0:
        raise ValueError("centroid is empty")

    w = max(1, int(len(centroid_ds) * window_pct))

    def _d(idx):
        s = downsample_signal(X_list[idx], downsample_factor)
        return dtw_distance(s, centroid_ds, window=w)

    res = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_d)(i) for i in range(len(X_list))
    )
    return np.array(res, dtype=float)


def compute_class_medoids_fixed(
    signals,
    labels,
    labels_runID,
    n_medoids=1,
    window_frac=0.10,
    n_jobs=None,
    norm="path",
    total_target=1000,
    sampling_strategy="proportional",
    random_state=0,
    hard_cap_per_run=None,
    seed=385,
):
    """
    Compute medoids per class and return:
      - templates: list of medoid signals
      - template_names: list of names ("medoid_<class>_0")
      - medoid_index_map: dict class_label -> list of indices INTO templates list (0..M-1)
    Ensures medoid_index_map refers to template positions (NOT original-signal indices).
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    labels_runID = np.asarray(labels_runID)
    unique_labels_runID = np.unique(labels_runID)

    templates = []
    template_names = []
    medoid_index_map = {}

    rng = np.random.default_rng(seed)

    for lab in unique_labels_runID:
        idxs = np.where(labels_runID == lab)[0].tolist()
        if len(idxs) == 0:
            medoid_index_map[lab] = []
            continue
        # subset signals for this class
        sub_signals = [signals[i] for i in idxs]

        if len(sub_signals) > total_target:
            chosen_idx = rng.choice(len(sub_signals), total_target, replace=False)
            sub_signals = [sub_signals[i] for i in chosen_idx]

        # compute pairwise DTW and pick medoid(s) (lowest sum-of-distances)
        D = get_pairwise_DTW(
            sub_signals, window_frac=window_frac, n_jobs=n_jobs, norm=norm
        )
        sums = np.sum(D, axis=1)
        order_local = np.argsort(sums)[
            :n_medoids
        ]  # indices within sub_signals (0..len(sub_signals)-1)
        chosen_template_positions = []
        for k_local in order_local:
            # global template index will be current length of templates list
            tmpl_pos = len(templates)
            chosen_template_positions.append(tmpl_pos)
            # append the actual medoid signal (map local index to original index)
            orig_idx = idxs[int(k_local)]
            templates.append(signals[orig_idx])
            template_names.append(f"medoid_{lab}_{len(chosen_template_positions)-1}")
        medoid_index_map[lab] = chosen_template_positions

    return templates, template_names, medoid_index_map


def compute_dtw_thresholds_by_percentile(
    train_signals,
    train_labels,
    train_labels_runID,
    templates,
    medoid_index_map,
    fraction=0.10,
    percentile=None,
    min_class_size=5,
    dtw_window_frac=0.10,
    n_jobs=None,
    norm="path",
    transform_fn=None,
):
    """
    Compute per-class thresholds from training data.
    - fraction: fraction to REMOVE per-class (e.g. 0.10 removes top 10% farthest)
    - percentile: optional override (e.g. 90 means keep <= 90th percentile)
      If None, percentile = 100 * (1 - fraction).
    Returns dict: {class_label: threshold_value}, also returns 'global' entry and stats.
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    if percentile is None:
        percentile = 100.0 * (1.0 - float(fraction))

    # compute dtw matrix: rows = train_signals, cols = templates
    df_dtw = dtw_features_to_templates(
        train_signals,
        templates,
        template_names=None,
        window_frac=dtw_window_frac,
        n_jobs=n_jobs,
        norm=norm,
    )
    mat = df_dtw.values.astype(float)
    if transform_fn is not None:
        mat = transform_fn(mat)

    labels_arr = np.asarray(train_labels_runID)
    classes = np.unique(labels_arr)
    thresholds = {}
    stats = {"per_class": {}}

    for cls in classes:
        cls_idx = np.where(labels_arr == cls)[0]
        n_cls = len(cls_idx)
        if n_cls < min_class_size:
            thresholds[cls] = np.inf  # don't remove any for small classes
            stats["per_class"][cls] = {
                "n": n_cls,
                "threshold": None,
                "reason": "too_small",
            }
            continue
        medoid_cols = medoid_index_map.get(cls, [])
        if not medoid_cols:
            # no medoids for this class (shouldn't happen if medoids computed on train)
            thresholds[cls] = np.inf
            stats["per_class"][cls] = {
                "n": n_cls,
                "threshold": None,
                "reason": "no_medoid",
            }
            continue
        # distances to own-class medoids: take min across medoids
        dvals = np.min(mat[cls_idx[:, None], medoid_cols], axis=1)
        thr = float(np.percentile(dvals, percentile))
        thresholds[cls] = thr
        stats["per_class"][cls] = {"n": n_cls, "threshold": thr, "pct": percentile}
    # global fallback threshold (optional): maybe use max of per-class thresholds
    thresholds["_global_max"] = float(
        max(v for v in thresholds.values() if np.isfinite(v))
        if any(np.isfinite(list(thresholds.values())))
        else np.inf
    )
    stats["thresholds"] = thresholds
    stats["train_shape"] = mat.shape
    return thresholds, stats


def apply_dtw_thresholds(
    signals,
    labels,
    labels_runID,
    templates,
    medoid_index_map,
    thresholds,
    dtw_window_frac=0.10,
    n_jobs=None,
    norm="path",
    transform_fn=None,
):
    """
    Apply per-class thresholds to a dataset (could be train/val/test).
    Returns keep_mask (bool array) and stats {removed_count, removed_indices, per_class_counts}.
    Samples whose class has threshold==inf are kept.
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    if len(signals) == 0:
        return np.array([], dtype=bool), {
            "removed_indices": [],
            "removed_count": 0,
            "n_total": 0,
        }

    df_dtw = dtw_features_to_templates(
        signals,
        templates,
        template_names=None,
        window_frac=dtw_window_frac,
        n_jobs=n_jobs,
        norm=norm,
    )
    mat = df_dtw.values.astype(float)
    if transform_fn is not None:
        mat = transform_fn(mat)

    labels_arr = np.asarray(labels_runID)
    N = len(signals)
    keep_mask = np.ones(N, dtype=bool)
    per_class_removed = {}

    for cls in np.unique(labels_arr):
        cls_idx = np.where(labels_arr == cls)[0]
        medoid_cols = medoid_index_map.get(cls, [])
        thr = thresholds.get(cls, np.inf)
        if len(cls_idx) == 0:
            per_class_removed[cls] = {"n": 0, "removed": []}
            continue
        if np.isinf(thr):
            per_class_removed[cls] = {"n": 0, "removed": []}
            continue
        if not medoid_cols:
            per_class_removed[cls] = {"n": 0, "removed": []}
            continue
        # compute distances to own medoids
        dvals = np.min(mat[cls_idx[:, None], medoid_cols], axis=1)
        # remove those > thr (strictly greater)
        to_remove_local = np.where(dvals > thr)[0]
        removed_global = cls_idx[to_remove_local].tolist()
        keep_mask[removed_global] = False
        per_class_removed[cls] = {
            "n": len(removed_global),
            "removed": removed_global,
            "threshold": float(thr),
        }
    removed_indices_all = np.where(~keep_mask)[0].tolist()
    stats = {
        "removed_indices": removed_indices_all,
        "removed_count": len(removed_indices_all),
        "n_total": N,
        "per_class": per_class_removed,
    }
    return keep_mask, stats


def filter_outliers_by_dtw(
    signals,
    labels,
    templates,
    template_names=None,
    medoid_index_map=None,
    fraction_filter=0.10,
    strategy="per_class",
    min_class_size=20,
    max_remove_per_class=None,
    dtw_window_frac=0.10,
    n_jobs=None,
    norm="path",
    resamp_len=None,
    transform_fn=None,
):
    """
    Filter outlier signals based on DTW distance to their class medoid(s).

    Parameters
    ----------
    signals : list of 1D numpy arrays
        Signals to evaluate (e.g. X_train_raw or X_val_raw or X_test_raw).
    labels : list/array of same length
        Class labels for each signal (must be known for val/test if you wish to remove by class).
    templates : list of 1D numpy arrays
        Medoid templates (computed on training data).
    template_names : optional list of strings (len = len(templates))
        Names for templates; if None auto-generated.
    medoid_index_map : optional dict class_label -> list of template indices (global indices into templates).
        If None, we will infer classes from template_names assuming names like "medoid_<class>_k".
    fraction_filter : float (0..1)
        Fraction to remove (per class if strategy='per_class', or globally if 'global').
    strategy : "per_class" or "global" or "iqr"
        - "per_class": remove top `fraction_filter` farthest per class (recommended).
        - "global": remove top `fraction_filter` farthest across all signals.
        - "iqr": remove samples > Q3 + k*IQR per class; in this case fraction_filter treated as k (e.g. 2.0).
    min_class_size : int
        Skip removal for classes with fewer than this many samples.
    max_remove_per_class : int or None
        Cap number removed per class (useful to avoid wiping small classes by mistake).
    dtw_window_frac, n_jobs, norm, resamp_len : forwarded to DTW computation.
    transform_fn : optional callable(distances_vector) -> transformed distances
        e.g. lambda d: np.log1p(d) or similar.

    Returns
    -------
    keep_mask : boolean numpy array length N (True = keep)
    stats : dict with details:
      - 'N_total', 'N_removed_total', 'removed_indices', 'per_class' : {class: {n, removed_indices, threshold, distances}}
      - 'distances': array of per-signal distance-to-own-medoid
      - 'df_dtw': (optional) the full distance DataFrame (if small)
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    N = len(signals)
    if N == 0:
        return np.array([], dtype=bool), {
            "N_total": 0,
            "N_removed_total": 0,
            "per_class": {},
        }

    labels_arr = np.asarray(labels)
    unique_labels, counts = np.unique(labels_arr, return_counts=True)
    # prepare template names
    M = len(templates)
    if template_names is None:
        template_names = [f"tmpl_{i}" for i in range(M)]

    # Build medoid_index_map if not provided by parsing template_names (assume format medoid_<class>_<k>)
    if medoid_index_map is None:
        medoid_index_map = {}
        for tidx, tname in enumerate(template_names):
            # try parse class from name
            parts = str(tname).split("_")
            if len(parts) >= 2 and parts[0].lower().startswith("medoid"):
                # medoid_<class>_k or medoid<class>_k; try flexible parsing
                cls = parts[1]
            else:
                # fallback: place in global "__all__"
                cls = "__unknown__"
            medoid_index_map.setdefault(cls, []).append(tidx)

    # Prefer fast template->signals function if available (avoid repeated full matrices)
    try:
        # dtw_features_to_templates_fast returns DataFrame (N x M)
        df_dtw = dtw_features_to_templates(
            signals,
            templates,
            template_names=template_names,
            window_frac=dtw_window_frac,
            n_jobs=n_jobs,
            norm=norm,
        )
        # ensure numpy
        dtw_mat = df_dtw.values.astype(float)
    except Exception:
        # fallback to slower version
        df_dtw = dtw_features_to_templates(
            signals,
            templates,
            template_names=template_names,
            window_frac=dtw_window_frac,
            n_jobs=n_jobs,
            norm=norm,
        )
        dtw_mat = df_dtw.values.astype(float)

    # optional resampling or transformation of distances (e.g. log1p)
    if transform_fn is not None:
        dist_mat = transform_fn(dtw_mat)
    else:
        dist_mat = dtw_mat

    dist_to_own = np.full(N, np.nan, dtype=float)
    # For signals whose class has no medoid (rare), we keep NaN
    for i in range(N):
        cls = labels_arr[i]
        cls_medoids = medoid_index_map.get(cls, [])
        if len(cls_medoids) == 0:
            dist_to_own[i] = np.nan
            continue
        # take min distance across medoids of that class
        dist_to_own[i] = float(np.min(dist_mat[i, cls_medoids]))

    # Prepare stats container
    stats = {
        "N_total": N,
        "distances": dist_to_own,
        "per_class": {},
        "df_dtw": (
            df_dtw if df_dtw.shape[0] <= 5000 else None
        ),  # keep df_dtw only when modest size
    }

    # initialize keep mask True
    keep_mask = np.ones(N, dtype=bool)

    # Removal strategies
    if strategy == "per_class":
        for cls in unique_labels:
            cls_idx = np.where(labels_arr == cls)[0]
            n_cls = len(cls_idx)
            stats["per_class"][cls] = {"n": n_cls, "removed": [], "threshold": None}
            # skip if small
            if n_cls < min_class_size:
                stats["per_class"][cls]["reason"] = "too_small"
                continue
            # get distances for class (exclude NaN)
            dcls = dist_to_own[cls_idx]
            valid_mask = ~np.isnan(dcls)
            if valid_mask.sum() == 0:
                stats["per_class"][cls]["reason"] = "no_medoid_or_no_valid_dist"
                continue
            dvals = dcls[valid_mask]
            idxs_valid = cls_idx[valid_mask]
            # number to remove
            k = int(math.floor(fraction_filter * n_cls))
            if k <= 0:
                stats["per_class"][cls]["reason"] = "k_zero"
                continue
            if max_remove_per_class is not None:
                k = min(k, int(max_remove_per_class))
            # if k >= n_cls, cap to n_cls-1
            if k >= n_cls:
                k = max(0, n_cls - 1)
            # sort by descending distance (farthest first), deterministic tie-breaker by original index
            order_desc = np.argsort(-dvals, kind="mergesort")  # stable
            remove_local_order = order_desc[:k]
            remove_global_idxs = idxs_valid[remove_local_order].tolist()
            # apply removal
            keep_mask[remove_global_idxs] = False
            stats["per_class"][cls]["removed"] = remove_global_idxs
            stats["per_class"][cls]["threshold"] = (
                float(np.max(dvals[remove_local_order])) if k > 0 else None
            )
            stats["per_class"][cls]["removed_count"] = len(remove_global_idxs)

    elif strategy == "global":
        # remove top fraction globally (ignoring class)
        valid_idx = np.where(~np.isnan(dist_to_own))[0]
        n_valid = len(valid_idx)
        k = int(math.floor(fraction_filter * n_valid))
        if k > 0:
            dvalid = dist_to_own[valid_idx]
            order_desc = np.argsort(-dvalid, kind="mergesort")
            remove_global_idxs = valid_idx[order_desc[:k]].tolist()
            keep_mask[remove_global_idxs] = False
            stats["N_removed_total"] = len(remove_global_idxs)
            stats["removed_indices"] = remove_global_idxs
        else:
            stats["N_removed_total"] = 0
            stats["removed_indices"] = []
    elif strategy == "iqr":
        # remove elements per-class beyond Q3 + k * IQR, where fraction_filter is 'k' multiplier
        k_mul = float(fraction_filter)
        for cls in unique_labels:
            cls_idx = np.where(labels_arr == cls)[0]
            n_cls = len(cls_idx)
            stats["per_class"][cls] = {"n": n_cls, "removed": [], "threshold": None}
            if n_cls < min_class_size:
                stats["per_class"][cls]["reason"] = "too_small"
                continue
            dcls = dist_to_own[cls_idx]
            valid_mask = ~np.isnan(dcls)
            if valid_mask.sum() == 0:
                stats["per_class"][cls]["reason"] = "no_medoid_or_no_valid_dist"
                continue
            dvals = dcls[valid_mask]
            idxs_valid = cls_idx[valid_mask]
            q1 = np.percentile(dvals, 25)
            q3 = np.percentile(dvals, 75)
            iqr = q3 - q1
            cutoff = q3 + k_mul * iqr
            # determine removals
            remove_mask_local = dvals > cutoff
            if remove_mask_local.sum() == 0:
                stats["per_class"][cls]["removed"] = []
                stats["per_class"][cls]["threshold"] = float(cutoff)
                continue
            remove_global_idxs = idxs_valid[remove_mask_local].tolist()
            if max_remove_per_class is not None:
                remove_global_idxs = remove_global_idxs[:max_remove_per_class]
            keep_mask[remove_global_idxs] = False
            stats["per_class"][cls]["removed"] = remove_global_idxs
            stats["per_class"][cls]["threshold"] = float(cutoff)
            stats["per_class"][cls]["removed_count"] = len(remove_global_idxs)
    else:
        raise ValueError("Unknown strategy: choose 'per_class'|'global'|'iqr'")

    # finalize stats
    removed_indices_all = np.where(~keep_mask)[0].tolist()
    stats["N_removed_total"] = len(removed_indices_all)
    stats["removed_indices"] = removed_indices_all
    stats["fraction_filter"] = fraction_filter
    stats["strategy"] = strategy
    stats["min_class_size"] = min_class_size
    stats["max_remove_per_class"] = max_remove_per_class

    return keep_mask, stats


def _dtw_distance_fast(a, b, radius=5):
    """Compute DTW distance using fastdtw (a & b must be 1D floats)."""
    a = _ensure_1d_float(a)
    b = _ensure_1d_float(b)
    if a.size == 0 or b.size == 0:
        return float("inf")
    d, _ = fastdtw(a, b, radius=radius, dist=2)
    return float(d)


def _dtw_distance_dtaid(a, b, window=10):
    a = _ensure_1d_float(a)
    b = _ensure_1d_float(b)
    # dtaidistance expects python lists / numpy arrays; it has multiple routines.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # dtaidistance.dtw.distance is exact and uses C speed.
    return float(dtaid_dtw.distance(a, b))


def _dtw_distance_numpy(a, b, window=5):
    """
    Exact DTW (constrained if window given). Pure numpy implementation.
    Window is integer (Sakoe-Chiba). This is slower but robust fallback.
    """
    a = _ensure_1d_float(a)
    b = _ensure_1d_float(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    if window is None:
        w = max(n, m)  # unconstrained
    else:
        w = max(window, abs(n - m))

    # initialize cost matrix with inf
    D = np.full((n + 1, m + 1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jmin = max(1, i - w)
        jmax = min(m, i + w)
        ai = a[i - 1]
        for j in range(jmin, jmax + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def _pairwise_dtw_condensed(signals, window_frac=0.1, n_jobs=1, norm=True):
    """
    Compute condensed pairwise DTW distances for list `signals` (already resampled).
    Returns full square matrix.
    """
    n = len(signals)
    D = np.zeros((n, n), dtype=float)

    # precompute lengths for normalization
    lengths = [len(s) for s in signals]

    # compute indexes for upper triangle (i<j)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def _worker(i, j):
        # dtw_distance_numba now returns (cost, path_len)
        raw_cost, path_len = dtw_distance_numba(
            signals[i], signals[j], window_frac=window_frac
        )

        # choose normalization
        if norm is True or str(norm).lower() == "path":
            denom = path_len
            if denom <= 0:
                return i, j, float("inf")
            return i, j, float(raw_cost) / float(denom)
        elif str(norm).lower() == "sum":
            denom = lengths[i] + lengths[j]
            if denom <= 0:
                return i, j, float("inf")
            return i, j, float(raw_cost) / float(denom)
        else:
            # no normalization
            return i, j, float(raw_cost)

    t0 = time.perf_counter()
    # parallel computation
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_worker)(i, j) for (i, j) in pairs
    )

    t1 = time.perf_counter()
    logger.info(
        "dtaidistance.distance_matrix_fast took %.2fs for %d series", t1 - t0, n
    )
    for i, j, val in results:
        D[i, j] = val
        D[j, i] = val

    return D


def _resample_signal(sig, target_len):
    """Resample 1D signal to target_len using scipy.signal.resample (fast, vectorized)."""
    sig = _ensure_1d_float(sig)
    if sig.size == 0:
        return np.zeros(target_len, dtype=float)
    if sig.size == target_len:
        return sig
    # resample returns floating values
    return resample(sig, target_len)


def visualize_distance_embedding(labels, D, umap_neighbors=200, outfile=None, seed=385):
    reducer = umap.UMAP(
        metric="precomputed",
        n_neighbors=umap_neighbors,
        min_dist=0.1,
        random_state=seed,
    )
    emb = reducer.fit_transform(D)

    # simple scatter colored by label
    plt.figure(figsize=(8, 6))
    unique = list(sorted(set(labels)))
    palette = sns.color_palette("tab10", n_colors=max(10, len(unique)))
    label_to_col = {lab: palette[i % len(palette)] for i, lab in enumerate(unique)}
    cols = [label_to_col[l] for l in labels]
    plt.scatter(emb[:, 0], emb[:, 1], c=cols, s=12, alpha=0.8)
    for lab in unique:
        plt.scatter([], [], c=[label_to_col[lab]], label=str(lab))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.title("DTW distance embedding with UMAP")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    return emb


def visualize_tsne_distance(
    labels, D, perplexity=30, n_iter=1000, outfile=None, seed=385, figsize=(8, 6)
):
    """
    Run t-SNE on a precomputed pairwise distance matrix and plot colored scatter by label.
    Returns 2D embedding array (n_samples, 2).
    """
    labels = list(labels)
    n = len(labels)
    if D.shape[0] != D.shape[1] or D.shape[0] != n:
        raise ValueError("D must be square and match length of labels")

    # adjust perplexity if too large for sample count
    max_perp = max(2, min(50, (n - 1) // 3))
    if perplexity > max_perp:
        perf_used = max_perp
    else:
        perf_used = perplexity

    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=perf_used,
        max_iter=n_iter,
        random_state=seed,
        init="random",
    )
    emb = tsne.fit_transform(D)

    # plotting
    plt.figure(figsize=figsize)
    unique = list(sorted(set(labels)))
    palette = sns.color_palette("tab10", n_colors=max(10, len(unique)))
    label_to_col = {lab: palette[i % len(palette)] for i, lab in enumerate(unique)}
    cols = [label_to_col[l] for l in labels]
    plt.scatter(emb[:, 0], emb[:, 1], c=cols, s=12, alpha=0.85, linewidths=0)

    # legend
    for lab in unique:
        plt.scatter([], [], c=[label_to_col[lab]], label=str(lab))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

    plt.title(f"DTW distances: t-SNE (perplexity={perf_used})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    return emb


def plot_silhouette_results(
    sep_dict, outfile_prefix=None, emb=None, labels=None, figsize=(8, 4)
):
    """
    Create and save silhouette plots.

    Inputs:
      - sep_dict: dict returned by compute_distance_separability(D, labels)
          expected keys: "global_silhouette", "sample_silhouette", "per_class_silhouette", "silhouette_df"
      - outfile_prefix: if provided, PNG files are saved with this prefix; otherwise figs are returned (but still created)
      - emb: optional 2D embedding (n_samples,2) for coloring samples by silhouette
      - labels: required if `emb` is provided and used for legend (list-like)
    Returns:
      dict mapping 'per_class_bar','histogram','emb_scatter' keys to filenames (or None if not saved)
    """

    outfiles = {"per_class_bar": None, "histogram": None, "emb_scatter": None}

    # Extract values safely
    global_sil = sep_dict.get("global_silhouette", np.nan)
    sample_sil = sep_dict.get("sample_silhouette", None)
    per_class = sep_dict.get("per_class_silhouette", pd.Series(dtype=float))
    sil_df = sep_dict.get("silhouette_df", None)

    sns.set_style("whitegrid")

    # 1) Per-class mean silhouette bar plot (sorted)
    if isinstance(per_class, (pd.Series, pd.DataFrame)) and len(per_class) > 0:
        pc = per_class.sort_values(ascending=True)  # ascending so worst on left
        fig, ax = plt.subplots(
            1, 1, figsize=(max(figsize[0], 0.4 * len(pc)), figsize[1])
        )
        colors = sns.color_palette("viridis", n_colors=len(pc))
        ax.barh(pc.index.astype(str), pc.values, color=colors)
        ax.set_xlabel("Mean silhouette score (per class)")
        ax.set_title(f"Per-class mean silhouette (global={global_sil:.3f})")
        ax.set_xlim(-1.0, 1.0)  # silhouette range
        plt.tight_layout()

        if outfile_prefix:
            fname = f"{outfile_prefix}_silhouette_per_class.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            outfiles["per_class_bar"] = fname
            plt.close(fig)
        else:
            outfiles["per_class_bar"] = None

    # 2) Histogram / density of per-sample silhouette
    if sample_sil is not None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # histogram
        sns.histplot(sample_sil, bins=30, kde=True, stat="density", ax=ax)
        ax.axvline(
            np.nanmedian(sample_sil),
            color="k",
            linestyle="--",
            lw=1,
            label=f"median={np.nanmedian(sample_sil):.3f}",
        )
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("Sample silhouette score")
        ax.set_title("Distribution of sample silhouette values")
        ax.legend()
        plt.tight_layout()
        if outfile_prefix:
            fname = f"{outfile_prefix}_silhouette_hist.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            outfiles["histogram"] = fname
            plt.close(fig)
        else:
            outfiles["histogram"] = None

    # 3) Optional: embedding colored by class, fading by sample silhouette (scaled to data min/max)
    if emb is not None:
        emb = np.asarray(emb)
        n = emb.shape[0]
        if sample_sil is None:
            raise ValueError(
                "sample_silhouette required in sep_dict to color embedding"
            )
        if emb.shape[0] != len(sample_sil):
            raise ValueError("emb and sample_silhouette length mismatch")

        # compute observed silhouette range and map to alpha
        sample_sil = np.asarray(sample_sil)
        sil_min = float(np.nanmin(sample_sil))
        sil_max = float(np.nanmax(sample_sil))

        alpha_min = 0.05
        alpha_max = 1.0
        if np.isclose(sil_max, sil_min):
            # all equal -> use opaque
            alpha_vals = np.full(n, alpha_max)
        else:
            # scale silhouette to 0..1 using observed min/max
            scaled = (sample_sil - sil_min) / (sil_max - sil_min)
            scaled = np.clip(scaled, 0.0, 1.0)
            alpha_vals = alpha_min + (alpha_max - alpha_min) * scaled

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(
                max(8, 0.6 * len(np.unique(labels)) if labels is not None else 8),
                6,
            ),
        )

        # prepare class colors
        unique_labels = np.unique(labels)
        palette = sns.color_palette("tab10", n_colors=max(10, len(unique_labels)))
        label_to_col = {
            lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)
        }

        # Plot each class separately, using per-point RGBA colors (class color + per-sample alpha)
        for lab in unique_labels:
            mask = np.asarray(labels) == lab
            if not np.any(mask):
                continue
            rgba_colors = [
                to_rgba(label_to_col[lab], alpha=float(a)) for a in alpha_vals[mask]
            ]
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=rgba_colors,
                s=18,
                edgecolors="none",
                label=str(lab),
            )

        # Build a legend with solid-color proxies (fully opaque) for each class
        class_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=label_to_col[lab],
                markersize=6,
                label=str(lab),
            )
            for lab in unique_labels
        ]
        class_legend = ax.legend(
            handles=class_handles,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize="small",
            title="Class",
        )
        ax.add_artist(class_legend)  # keep class legend

        # Build alpha (silhouette -> opacity) legend with three example points (low, median, high)
        med = float(np.nanmedian(sample_sil))
        low_val = sil_min
        mid_val = med
        high_val = sil_max
        alpha_low = (
            alpha_min
            if np.isclose(sil_max, sil_min)
            else (
                alpha_min
                + (alpha_max - alpha_min) * ((low_val - sil_min) / (sil_max - sil_min))
            )
        )
        alpha_mid = (
            alpha_min
            if np.isclose(sil_max, sil_min)
            else (
                alpha_min
                + (alpha_max - alpha_min) * ((mid_val - sil_min) / (sil_max - sil_min))
            )
        )
        alpha_high = alpha_max

        alpha_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=6,
                alpha=alpha_low,
                label=f"low: {low_val:.3f}",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=6,
                alpha=alpha_mid,
                label=f"med: {mid_val:.3f}",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=6,
                alpha=alpha_high,
                label=f"high: {high_val:.3f}",
            ),
        ]
        ax.legend(
            handles=alpha_handles,
            bbox_to_anchor=(1.05, 0.55),
            loc="upper left",
            fontsize="small",
            title="Silhouette → opacity",
        )

        # Optionally overlay centroids (always drawn in opaque text boxes)
        if labels is not None:
            try:
                labs = np.asarray(labels)
                uniq = np.unique(labs)
                for u in uniq:
                    mask = labs == u
                    if np.sum(mask) > 0:
                        cx, cy = emb[mask][:, 0].mean(), emb[mask][:, 1].mean()
                        ax.text(
                            cx,
                            cy,
                            str(u),
                            fontsize=8,
                            ha="center",
                            va="center",
                            bbox=dict(
                                facecolor="white", alpha=0.7, edgecolor="none", pad=1
                            ),
                        )
            except Exception:
                pass

        ax.set_title("Embedding (UMAP/t-SNE) colored by class; faded by silhouette")
        ax.set_xlabel("Embedding 1")
        ax.set_ylabel("Embedding 2")
        plt.tight_layout()

        if outfile_prefix:
            fname = f"{outfile_prefix}_silhouette_embedding.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            outfiles["emb_scatter"] = fname
            plt.close(fig)
        else:
            outfiles["emb_scatter"] = None

    return outfiles


def plot_within_between_boxplots(
    D,
    labels,
    outdir=None,
):
    """
    Simplified per-class within/between DTW summary and plotting.

    Returns dict with:
      - pairwise_df: each pair (i<j) with distance and labels
      - class_pair_stats: class x class mean/median/count
      - within_between_summary: per-class within / between summary
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    if D.shape[0] != D.shape[1] or D.shape[0] != n:
        raise ValueError("D must be square and match length of labels")

    # Build pairwise upper triangle table
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "label_i": labels[i],
                    "label_j": labels[j],
                    "distance": float(D[i, j]),
                    "same": labels[i] == labels[j],
                }
            )
    pairwise_df = pd.DataFrame(rows)

    # Per-class within / between summary
    classes = np.unique(labels)
    wb_rows = []
    class_pair_rows = []
    for a in classes:
        # within distances for class a
        within_mask = (pairwise_df["same"]) & (pairwise_df["label_i"] == a)
        within = pairwise_df.loc[within_mask, "distance"].to_numpy()
        wb_rows.append(
            {
                "class": a,
                "type": "within",
                "count": int(within.size),
                "mean": float(np.nanmean(within)) if within.size else np.nan,
                "median": float(np.nanmedian(within)) if within.size else np.nan,
            }
        )

        # between distances for class a (pairs where a is one member but pair is not same)
        between_mask = (~pairwise_df["same"]) & (
            (pairwise_df["label_i"] == a) | (pairwise_df["label_j"] == a)
        )
        between = pairwise_df.loc[between_mask, "distance"].to_numpy()
        wb_rows.append(
            {
                "class": a,
                "type": "between",
                "count": int(between.size),
                "mean": float(np.nanmean(between)) if between.size else np.nan,
                "median": float(np.nanmedian(between)) if between.size else np.nan,
            }
        )

        # pairwise class×class statistics (mean/median/count)
        for b in classes:
            mask_ab = (
                (pairwise_df["label_i"] == a) & (pairwise_df["label_j"] == b)
            ) | ((pairwise_df["label_i"] == b) & (pairwise_df["label_j"] == a))
            d = pairwise_df.loc[mask_ab, "distance"].to_numpy()
            class_pair_rows.append(
                {
                    "class_a": a,
                    "class_b": b,
                    "mean": float(np.nanmean(d)) if d.size else np.nan,
                    "median": float(np.nanmedian(d)) if d.size else np.nan,
                    "count": int(d.size),
                }
            )

    within_between_summary = pd.DataFrame(wb_rows)
    class_pair_stats = pd.DataFrame(class_pair_rows)

    plot_df_rows = []
    for _, r in pairwise_df.iterrows():
        t = "within" if r["same"] else "between"
        # assign to both classes so each class sees the pair
        plot_df_rows.append(
            {"class": r["label_i"], "type": t, "distance": r["distance"]}
        )
        plot_df_rows.append(
            {"class": r["label_j"], "type": t, "distance": r["distance"]}
        )
    plot_df = pd.DataFrame(plot_df_rows)

    # compute median table (per class x type) from the same data we will plot
    med_df = plot_df.groupby(["class", "type"])["distance"].median().unstack()

    sns.set_theme(style="whitegrid")
    fig_violin = plt.figure(figsize=(max(10, len(classes) * 0.5), 6))
    ax = sns.violinplot(
        data=plot_df,
        x="class",
        y="distance",
        hue="type",
        split=True,
        inner=None,
        cut=0,
        density_norm="width",
    )
    sns.boxplot(
        data=plot_df,
        x="class",
        y="distance",
        hue="type",
        showcaps=True,
        boxprops={"facecolor": "none"},
        showfliers=False,
        whis=1.5,
        width=0.15,
        ax=ax,
    )

    # unify legend (violin + boxplot create duplicate handles)
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels_[:2], title="type")

    # annotate medians: place small markers and numeric labels slightly above the median
    # get plotting order from axis ticks (in case seaborn reorders)
    plotted_order = [t.get_text() for t in ax.get_xticklabels()]
    x_positions = {cls: idx for idx, cls in enumerate(plotted_order)}
    # small horizontal offset for within/between markers
    offset = 0.18

    for cls in plotted_order:
        i = x_positions[cls]
        med_within = (
            med_df.loc[cls]["within"]
            if ("within" in med_df.columns and cls in med_df.index)
            else np.nan
        )
        med_between = (
            med_df.loc[cls]["between"]
            if ("between" in med_df.columns and cls in med_df.index)
            else np.nan
        )

        if not np.isnan(med_within):
            ax.scatter(
                i - offset,
                med_within,
                marker="o",
                s=36,
                edgecolor="black",
                facecolor="white",
                zorder=10,
            )
            ax.text(
                i - offset,
                med_within,
                f"{med_within:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

        if not np.isnan(med_between):
            ax.scatter(
                i + offset,
                med_between,
                marker="o",
                s=36,
                color="black",
                facecolor="white",
                zorder=10,
            )
            ax.text(
                i + offset,
                med_between,
                f"{med_between:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_ylabel("DTW Distance")
    ax.set_xlabel("Peptide class")
    ax.set_title("Within vs Between DTW distances per class (medians shown)")
    plt.tight_layout()
    if outdir:
        base_name = "dtw_within_between"
        fname = os.path.join(outdir, base_name)
        fig_violin.savefig(f"{fname}.png", dpi=600, bbox_inches="tight")
        fig_violin.savefig(f"{fname}.svg", bbox_inches="tight")
    plt.close(fig_violin)

    # Plot: class × class mean distance heatmap
    # build pivot (class x class mean)
    pivot = class_pair_stats.pivot(index="class_a", columns="class_b", values="mean")
    pivot = pivot.reindex(index=classes, columns=classes)

    # convert to numeric matrix (keep NaNs)
    pivot_vals = pivot.to_numpy(dtype=float)

    # RAW heatmap: clip upper colorbar at 99th percentile for readability
    vmax_raw = np.nanpercentile(pivot_vals, 99)
    fig_raw = plt.figure(
        figsize=(max(6, len(classes) * 0.35), max(5, len(classes) * 0.35))
    )
    sns.heatmap(
        pivot,
        cmap="Blues",
        square=True,
        cbar_kws={"shrink": 0.7},
        vmin=np.nanmin(pivot_vals),
        vmax=vmax_raw,
    )
    plt.title("Mean DTW distance (raw; clipped 99th pct)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    if outdir:
        base_name = "dtw_heatmap_raw"
        fname = os.path.join(outdir, base_name)
        fig_raw.savefig(f"{fname}.png", dpi=600, bbox_inches="tight")
        fig_raw.savefig(f"{fname}.svg", bbox_inches="tight")
    plt.close(fig_raw)

    # ROW-normalize (min-max) per reference class (rows)
    H = pivot_vals.copy()
    H_norm = np.full_like(H, np.nan, dtype=float)
    mins = np.nanmin(H, axis=1, keepdims=True)
    maxs = np.nanmax(H, axis=1, keepdims=True)
    denom = maxs - mins
    denom[denom == 0] = 1.0  # avoid division by zero for constant rows
    H_norm = (H - mins) / denom

    # put back into DataFrame with same index/cols
    pivot_rownorm = pd.DataFrame(H_norm, index=classes, columns=classes)

    # save normalized table to CSV
    # csvname = f"{outfile_prefix}_heatmap_row_normalized.csv"
    # pivot_rownorm.to_csv(csvname)

    # Plot row-normalized heatmap
    fig_row = plt.figure(
        figsize=(max(6, len(classes) * 0.35), max(5, len(classes) * 0.35))
    )
    sns.heatmap(
        pivot_rownorm, cmap="Blues", center=0.5, square=True, cbar_kws={"shrink": 0.7}
    )
    plt.title("Mean DTW distance (row min-max normalized: 0..1)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    if outdir:
        base_name = "dtw_heatmap_row_normalized"
        fname = os.path.join(outdir, base_name)
        fig_row.savefig(f"{fname}.png", dpi=600, bbox_inches="tight")
        fig_row.savefig(f"{fname}.svg", bbox_inches="tight")
    plt.close(fig_row)

    # Combined figure with both plots side-by-side (useful for manuscript)
    fig_comb, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(max(12, len(classes) * 0.7), max(5, len(classes) * 0.35))
    )
    sns.heatmap(
        pivot,
        cmap="Blues",
        square=True,
        cbar_kws={"shrink": 0.7},
        vmin=np.nanmin(pivot_vals),
        vmax=vmax_raw,
        ax=ax1,
    )
    ax1.set_title("Raw mean DTW (clipped 99th pct)")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=8)
    ax1.set_ylabel("Peptide class")
    ax1.set_xlabel("Peptide class")
    sns.heatmap(
        pivot_rownorm,
        cmap="Blues",
        center=0.5,
        square=True,
        cbar_kws={"shrink": 0.7},
        ax=ax2,
    )
    ax2.set_title("Row min-max normalized (0..1)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels([], visible=False)
    ax2.set_ylabel("Peptide class")
    ax2.set_xlabel("Peptide class")

    plt.tight_layout()
    if outdir:
        base_name = "dtw_heatmap_both"
        fname = os.path.join(outdir, base_name)
        fig_comb.savefig(f"{fname}.png", dpi=600, bbox_inches="tight")
        fig_comb.savefig(f"{fname}.svg", bbox_inches="tight")
    plt.close(fig_comb)

    # per-class boxplots to each other class
    ncols = min(4, len(classes))
    nrows = int(math.ceil(len(classes) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 4, max(3, nrows * 2.5))
    )
    axes = np.array(axes).reshape(-1)

    for idx, cls in enumerate(classes):
        axc = axes[idx]
        mask_cls = (pairwise_df["label_i"] == cls) | (pairwise_df["label_j"] == cls)
        df_cls = pairwise_df.loc[mask_cls].copy()

        # normalize to consistent direction
        df_cls["other"] = df_cls.apply(
            lambda r: r["label_j"] if r["label_i"] == cls else r["label_i"], axis=1
        )

        # build palette
        unique_others = df_cls["other"].unique()
        palette_dict = {
            o: ("#1f77b4" if o == cls else "lightgray") for o in unique_others
        }

        sns.boxplot(
            data=df_cls,
            x="other",
            y="distance",
            hue="other",
            ax=axc,
            showfliers=False,
            palette=palette_dict,
        )
        axc.set_title(f"{cls} -> others")
        axc.set_xlabel("")
        axc.set_ylabel("DTW Distance")
        axc.tick_params(axis="x", rotation=45, labelsize=7)
        # for k in range(len(classes), axes.size):
        #    fig.delaxes(axes[k])
        # annotate median values under each box
        medians = df_cls.groupby("other")["distance"].median()
        order = [t.get_text() for t in axc.get_xticklabels()]
        for xtick, other in enumerate(order):
            if other in medians:
                median_val = medians[other]
                axc.text(
                    xtick,
                    axc.get_ylim()[0]
                    - 0.15
                    * (axc.get_ylim()[1] - axc.get_ylim()[0]),  # slightly below y-axis
                    f"{median_val:.1f}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="black",
                    rotation=45,
                )
        plt.tight_layout()

    for k in range(len(classes), axes.size):
        ax_to_remove = axes[k]
        if ax_to_remove in fig.axes:
            fig.delaxes(ax_to_remove)

    if outdir:
        base_name = "dtw_per_class_boxplots"
        fname = os.path.join(outdir, base_name)
        fig.savefig(f"{fname}.png", dpi=600, bbox_inches="tight")
        fig.savefig(f"{fname}.svg", bbox_inches="tight")
        plt.close(fig)

    return {
        "pairwise_df": pairwise_df,
        "class_pair_stats": class_pair_stats,
        "within_between_summary": within_between_summary,
    }


def get_pairwise_DTW(
    X_list,
    resamp_len=None,
    downsample_factor=None,
    dtw_radius=None,
    window_frac=0.1,
    n_jobs=None,
    norm="path",
):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    n = len(X_list)
    if n == 0:
        raise ValueError("Empty input list")

    # decide target length: either user-specified or based on median length/downsample
    orig_lengths = [len(_ensure_1d_float(x)) for x in X_list]
    median_len = int(np.median(orig_lengths))

    if resamp_len is not None and downsample_factor is not None:
        logging.warning(
            "Both resamp_len and downsample_factor specified; using resamp_len only"
        )
        target_len = (
            max(20, median_len // max(1, downsample_factor))
            if resamp_len is None
            else int(resamp_len)
        )
        X_list = [_resample_signal(x, target_len) for x in X_list]
        orig_lengths = [len(_ensure_1d_float(x)) for x in X_list]
        median_len = int(np.median(orig_lengths))
        logger.info(
            "Resampling %d signals -> length %d (median original length %d)",
            n,
            target_len,
            median_len,
        )

    logging.info(
        "Original lengths: min %d, max %d, median %d",
        min(orig_lengths),
        max(orig_lengths),
        median_len,
    )

    logger.info("Computing pairwise DTW distances (n=%d) with %d workers", n, n_jobs)
    logger.info("Using DTW window fraction %.3f", window_frac)
    D = _pairwise_dtw_condensed(
        X_list, window_frac=window_frac, n_jobs=n_jobs, norm=norm
    )
    return D


def trim_trailing_constant_padding(sig, pad_value=0.0, min_trailing_len=None):
    """
    Remove trailing padding equal to pad_value.
    If pad_value occurs legitimately inside the signal, this only trims long trailing runs.
    min_trailing_len: if None use max(50, int(0.05*len(sig)))
    """
    a = np.asarray(sig)
    return a[a != 0.0]


def compute_distance_separability(D, labels):
    """
    Compute separability statistics using a precomputed distance matrix D and class labels.
    Returns a dict with:
      - global_silhouette: float
      - sample_silhouette: np.ndarray (len == n_samples)
      - per_class_silhouette: pd.Series (index=class -> mean silhouette)
      - silhouette_df: DataFrame with columns ['index','label','silhouette']
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    if D.shape[0] != D.shape[1] or D.shape[0] != n:
        raise ValueError("D must be square and match length of labels")

    # If only one class present or only one sample per class silhouette is undefined
    unique_classes, counts = np.unique(labels, return_counts=True)
    if unique_classes.size < 2 or np.any(counts < 2):
        return {
            "global_silhouette": np.nan,
            "sample_silhouette": np.full(n, np.nan),
            "per_class_silhouette": pd.Series(dtype=float),
            "silhouette_df": pd.DataFrame(
                {
                    "index": np.arange(n),
                    "label": labels,
                    "silhouette": np.full(n, np.nan),
                }
            ),
        }

    # Global silhouette (precomputed distances)
    try:
        global_sil = silhouette_score(D, labels, metric="precomputed")
        sample_sil = silhouette_samples(D, labels, metric="precomputed")
    except Exception as e:
        # Fallback: sometimes numeric issues; return NaNs but raise context
        raise RuntimeError(f"Silhouette computation failed: {e}")

    sil_df = pd.DataFrame(
        {"index": np.arange(n), "label": labels, "silhouette": sample_sil}
    )
    per_class = (
        sil_df.groupby("label")["silhouette"].mean().sort_values(ascending=False)
    )

    return {
        "global_silhouette": float(global_sil),
        "sample_silhouette": sample_sil,
        "per_class_silhouette": per_class,
        "silhouette_df": sil_df,
    }


def plot_hierarchical_signals(
    D,
    signals,
    labels,
    outfile="hierarchical_signals.png",
    max_per_class=30,
    seed=385,
    sample_name_prefix="",
    figsize_per_row=0.25,
    cmap="tab10",
    annotate_max_cluster_size=10,
):
    """
    Create a left->right figure: dendrogram (with DTW distances) | class label | raw signal plot per leaf-row.

    annotate_max_cluster_size: annotate merges only when BOTH child clusters have size <= this value.
                             Set to a large value to annotate many merges; set to 1 for leaf-leaf only.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    unique = np.unique(labels)

    # choose subset indices up to max_per_class per class
    chosen = []
    for u in unique:
        idxs = np.where(labels == u)[0].tolist()
        if len(idxs) == 0:
            continue
        if len(idxs) <= max_per_class:
            sel = idxs
        else:
            sel = rng.choice(idxs, max_per_class, replace=False).tolist()
        chosen.extend(sel)
    chosen = sorted(chosen, key=lambda i: (str(labels[i]), i))

    if len(chosen) < 2:
        raise ValueError("Need at least 2 samples for hierarchical clustering")

    # Subset
    D_sub = D[np.ix_(chosen, chosen)].astype(float)
    sigs_sub = [signals[i] for i in chosen]
    labs_sub = [labels[i] for i in chosen]
    names_sub = [f"{sample_name_prefix}{i}" for i in chosen]
    n = len(chosen)

    # linkage
    condensed = squareform(D_sub, checks=False)
    Z = hierarchy.linkage(condensed, method="average")

    # prepare clusters map for computing true mean pairwise distances between clusters
    # cluster ids: 0..n-1 original leaves; new cluster created at id = n + row_idx
    clusters = {i: set([i]) for i in range(n)}

    # figure layout
    fig_h = max(4, n * figsize_per_row)
    fig_w = 12
    fig = plt.figure(figsize=(fig_w, fig_h))
    w_dend = 0.20
    w_label = 0.12
    w_signal = 0.66
    left0 = 0.02
    gutter = 0.01

    # dendrogram coloring (root darker)
    heights = Z[:, 2]
    hmin, hmax = float(np.nanmin(heights)), float(np.nanmax(heights))
    cmap_tree = plt.get_cmap("viridis_r")
    norm = mcolors.Normalize(vmin=hmin, vmax=hmax)

    def link_color_func(link_id):
        if link_id < n:
            return mcolors.to_hex((0.65, 0.65, 0.65))
        idx = int(link_id - n)
        height = Z[idx, 2]
        return mcolors.to_hex(cmap_tree(norm(height)))

    # dendrogram axis
    ax_dend = fig.add_axes([left0, 0.05, w_dend - gutter, 0.9])
    dendro = hierarchy.dendrogram(
        Z,
        orientation="left",
        labels=None,
        ax=ax_dend,
        no_labels=True,
        color_threshold=None,
        link_color_func=link_color_func,
    )
    ax_dend.set_xlabel("DTW distance")

    # We will annotate merges selectively. As we iterate Z rows, update clusters dict.
    icoords = np.array(dendro["icoord"])
    dcoords = np.array(dendro["dcoord"])

    # compute axis width for offset
    x_range = None

    for row_idx, (xs, ys) in enumerate(zip(icoords, dcoords)):
        zrow = Z[row_idx]
        idx1, idx2 = int(zrow[0]), int(zrow[1])
        new_id = n + row_idx
        # compute cluster members sets
        members1 = clusters[idx1]
        members2 = clusters[idx2]
        clusters[new_id] = members1.union(members2)

    # After building clusters dict, annotate (we loop again to use dendro coords)
    # Precompute axis limits for offset
    x_min, x_max = ax_dend.get_xlim()
    x_span = x_max - x_min
    x_off = 0.02 * x_span  # shift annotations left by 2% of axis width

    # iterate merges again to annotate if both child clusters small enough
    for row_idx, (xs, ys) in enumerate(zip(icoords, dcoords)):
        zrow = Z[row_idx]
        idx1, idx2 = int(zrow[0]), int(zrow[1])
        members1 = clusters[idx1]
        members2 = clusters[idx2]
        size1 = len(members1)
        size2 = len(members2)

        # compute true mean pairwise DTW between cluster1 and cluster2 using D_sub
        # note: members indices refer to 0..n-1
        ids1 = np.array(sorted(list(members1)), dtype=int)
        ids2 = np.array(sorted(list(members2)), dtype=int)
        if ids1.size > 0 and ids2.size > 0:
            # mean of cross-distance matrix
            mean_pairwise = float(D_sub[np.ix_(ids1, ids2)].mean())
        else:
            mean_pairwise = float(zrow[2])

        # annotate only if both cluster sizes <= annotate_max_cluster_size (keeps figure readable)
        if size1 <= annotate_max_cluster_size and size2 <= annotate_max_cluster_size:
            x_mid = np.mean(xs)
            y_mid = np.mean(ys)
            # place annotation to the left of the horizontal link (so it doesn't overlap labels)
            ax_dend.text(
                x_mid - x_off,
                y_mid,
                f"{mean_pairwise:.0f}",
                fontsize=6,
                va="center",
                ha="right",
                color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
                clip_on=True,
            )

    # get leaf order top->bottom
    leaf_order = dendro["leaves"]

    # label column
    ax_label = fig.add_axes([left0 + w_dend + gutter, 0.05, w_label - gutter, 0.9])
    ax_label.axis("off")

    # signals axis
    ax_sig = fig.add_axes(
        [left0 + w_dend + w_label + 2 * gutter, 0.05, w_signal - 2 * gutter, 0.9]
    )
    max_len = int(max(len(s) for s in sigs_sub))
    ax_sig.set_xlim(0, max_len)
    ax_sig.set_ylim(-0.5, n - 0.5)
    ax_sig.invert_yaxis()
    ax_sig.set_xlabel("Sample index (raw signal x)")
    ax_sig.set_yticks([])

    palette = sns.color_palette(cmap, n_colors=max(10, n))

    for rowpos, leaf_idx in enumerate(leaf_order):
        sig = sigs_sub[leaf_idx]
        lab = labs_sub[leaf_idx]
        name = names_sub[leaf_idx]
        color = palette[rowpos % len(palette)]

        xs = np.arange(len(sig))
        ys = sig - np.nanmin(sig)
        if np.nanmax(np.abs(ys)) > 0:
            ys = ys / (np.nanmax(np.abs(ys)) + 1e-12)
        else:
            ys = ys * 0.0

        amp = 0.6
        yvals = rowpos - amp / 2.0 + ys * amp

        ax_sig.plot(xs, yvals, color=color, linewidth=1.0)
        ax_sig.hlines(
            rowpos, 0, max(0, len(sig) - 1), color="lightgray", linewidth=0.5, alpha=0.6
        )

        y_text = (rowpos + 0.5) / n
        ax_label.text(
            0.02,
            y_text,
            str(lab),
            transform=ax_label.transAxes,
            fontsize=8,
            va="center",
            ha="left",
        )
        ax_label.text(
            0.02,
            y_text - (0.012),
            name,
            transform=ax_label.transAxes,
            fontsize=6,
            va="center",
            ha="left",
            color="dimgray",
        )

    ax_sig.set_yticks(np.arange(n))
    ax_sig.set_yticklabels([names_sub[i] for i in leaf_order], fontsize=6)
    ax_sig.yaxis.set_ticks_position("right")
    ax_sig.tick_params(axis="y", which="both", length=0)
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(0, 1)

    # title and layout
    fig.suptitle(
        "Hierarchical clustering (DTW) + signals (each row = one sample)",
        y=0.995,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.92, left=0.02, right=0.98, bottom=0.03, wspace=0.02)

    # save
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outfile


def umap_on_features(
    features_df, labels, n_neighbors=30, min_dist=0.1, outfile=None, seed=385
):
    """
    Run UMAP on scaled feature matrix and plot colored by label.
    Returns embedding (n_samples, 2).
    """
    X = features_df.values
    # handle NaNs: simple impute by column median (you can replace with fancier imputation)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(col_medians, inds[1])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    reducer = umap.UMAP(
        metric="euclidean",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    emb = reducer.fit_transform(Xs)

    # plotting
    plt.figure(figsize=(8, 6))
    unique = list(sorted(set(labels)))
    palette = sns.color_palette("tab10", n_colors=max(10, len(unique)))
    label_to_col = {lab: palette[i % len(palette)] for i, lab in enumerate(unique)}
    cols = [label_to_col[l] for l in labels]
    plt.scatter(emb[:, 0], emb[:, 1], c=cols, s=14, alpha=0.9)
    for lab in unique:
        plt.scatter([], [], c=[label_to_col[lab]], label=str(lab))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.title("UMAP on features")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
    return emb


def umap_on_combined_distance(
    features_df,
    D_dtw,
    labels,
    alpha=0.5,
    n_neighbors=80,
    min_dist=0.1,
    outfile=None,
    seed=385,
):
    """
    Combine normalized DTW distance and feature-space distance into one combined distance:
      D_combined = alpha * D_dtw_norm + (1-alpha) * D_feat_norm
    Then run UMAP with metric='precomputed' on D_combined.
    alpha=1.0 -> only DTW; alpha=0 -> only features.
    """
    # features -> scaled Euclidean pairwise distances
    X = features_df.values
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(col_medians, inds[1])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    D_feat = pairwise_distances(Xs, metric="euclidean")

    # normalize both matrices to 0..1
    def norm_mat(M):
        M = M.astype(float)
        M -= np.nanmin(M)
        mx = np.nanmax(M)
        if mx <= 0:
            return np.zeros_like(M)
        return M / mx

    Dd = norm_mat(D_dtw)
    Df = norm_mat(D_feat)
    D_comb = alpha * Dd + (1.0 - alpha) * Df
    # ensure symmetry and zero diagonal
    D_comb = (D_comb + D_comb.T) / 2.0
    np.fill_diagonal(D_comb, 0.0)

    reducer = umap.UMAP(
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    emb = reducer.fit_transform(D_comb)

    # plotting
    plt.figure(figsize=(8, 6))
    unique = list(sorted(set(labels)))
    palette = sns.color_palette("tab10", n_colors=max(10, len(unique)))
    label_to_col = {lab: palette[i % len(palette)] for i, lab in enumerate(unique)}
    cols = [label_to_col[l] for l in labels]
    plt.scatter(emb[:, 0], emb[:, 1], c=cols, s=14, alpha=0.9)
    for lab in unique:
        plt.scatter([], [], c=[label_to_col[lab]], label=str(lab))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.title(f"UMAP on combined distance (alpha={alpha})")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
    return emb


def unify_length(X_list, target_length=None, padding_value=0):
    # X_list: list of arrays with shape (N_i, 1, L_i)
    lengths = [x.shape[-1] for x in X_list]
    if target_length is None:
        target_length = max(lengths)
    out = []
    for x in X_list:
        if x.shape[-1] == target_length:
            out.append(x)
        elif x.shape[-1] < target_length:
            pad = np.full(
                (x.shape[0], x.shape[1], target_length - x.shape[-1]),
                padding_value,
                dtype=x.dtype,
            )
            out.append(np.concatenate([x, pad], axis=-1))
        else:  # truncate (should be rare if you consistently pad to fixed_length)
            out.append(x[..., :target_length])
    return out


def plot_feature_boxplots(
    features_df,
    labels,
    feature_names=None,
    outfile=None,
    max_cols=4,
    figsize_per_plot=(4, 4),
    show_points=False,
    point_alpha=0.6,
    title=None,
    sharey=False,
):
    """
    Plot boxplots for features grouped by class labels.

    Parameters
    ----------
    features_df : pd.DataFrame
        Rows = samples, columns = feature names.
    labels : array-like (len == features_df.shape[0])
        Class label for each row (can be list, np.array, pd.Series).
    feature_names : list or None
        Names of features (subset of features_df.columns) to plot.
        If None, plot all columns from features_df.
    outfile : str or None
        If provided, save the figure to this path (png/pdf/etc).
    max_cols : int
        Max number of subplots per row.
    figsize_per_plot : tuple(float, float)
        Width, height per subplot (in inches).
    show_points : bool
        Overlay raw sample points (swarmplot / strip plot) on top of boxplots.
    point_alpha : float
        Alpha for overlayed points.
    title : str or None
        Figure title.
    sharey : bool
        Whether to share y-axis across subplots.

    Returns
    -------
    fig, axs
        Matplotlib figure and axes array.
    """
    # Validate inputs
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("features_df must be a pandas DataFrame")
    labels = pd.Series(labels, name="class")
    if len(labels) != len(features_df):
        raise ValueError("labels length must match features_df number of rows")

    # Decide features to plot
    if feature_names is None:
        feature_names = list(features_df.columns)
    else:
        # keep only those present
        feature_names = [f for f in feature_names if f in features_df.columns]
        if len(feature_names) == 0:
            raise ValueError("None of the feature_names exist in features_df.columns")

    n = len(feature_names)
    ncols = min(max_cols, n) if n > 0 else 1
    nrows = math.ceil(n / ncols)
    fig_w = figsize_per_plot[0] * ncols
    fig_h = figsize_per_plot[1] * nrows
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False, sharey=sharey
    )
    axs = axs.flatten()

    # Convert features + labels into a long DataFrame for seaborn
    df_long = pd.concat(
        [features_df.reset_index(drop=True), labels.reset_index(drop=True)], axis=1
    )

    # Order classes by frequency (optional)
    class_order = df_long["class"].value_counts().index.tolist()

    for i, feat in enumerate(feature_names):
        ax = axs[i]
        # If feature contains NaN or non-numeric, coerce
        col = pd.to_numeric(df_long[feat], errors="coerce")
        plot_df = pd.DataFrame({feat: col, "class": df_long["class"]}).dropna(
            subset=[feat]
        )
        if plot_df.empty:
            ax.text(0.5, 0.5, f"No numeric data for {feat}", ha="center", va="center")
            ax.set_title(feat)
            ax.set_xticks([])
            continue

        sns.boxplot(
            x="class",
            y=feat,
            data=plot_df,
            ax=ax,
            order=class_order,
            showcaps=True,
            showfliers=False,
            boxprops={"linewidth": 0.8},
            medianprops={"linewidth": 1.2},
        )
        if show_points:
            # use stripplot for large n, swarmplot otherwise
            try:
                if len(plot_df) / len(class_order) > 30:
                    sns.stripplot(
                        x="class",
                        y=feat,
                        data=plot_df,
                        ax=ax,
                        order=class_order,
                        alpha=point_alpha,
                        jitter=True,
                    )
                else:
                    sns.swarmplot(
                        x="class",
                        y=feat,
                        data=plot_df,
                        ax=ax,
                        order=class_order,
                        alpha=point_alpha,
                    )
            except Exception:
                # fallback to stripplot if swarmplot fails (e.g., too many points)
                sns.stripplot(
                    x="class",
                    y=feat,
                    data=plot_df,
                    ax=ax,
                    order=class_order,
                    alpha=point_alpha,
                    jitter=True,
                )

        ax.set_title(feat)
        ax.set_xlabel("")  # keep x-axis label clean
        # rotate x tick labels only if many classes or special characters
        if len(class_order) > 6 or any(len(str(c)) > 6 for c in class_order):
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        # tighten y-limits a bit for visibility (avoid clipping median/mus)
        ymin, ymax = ax.get_ylim()
        pad = (ymax - ymin) * 0.05
        ax.set_ylim(ymin - pad, ymax + pad)

    # remove unused axes
    for j in range(n, len(axs)):
        fig.delaxes(axs[j])

    if title:
        fig.suptitle(title, fontsize=14, y=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else None)

    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
        logger.info(f"Saved boxplots to {outfile}")

    return fig, axs[:n]


def compute_class_medoids_clustered(
    signals,
    labels,
    n_clusters=10,
    n_medoids=1,
    total_target=1000,
    window_frac=0.10,
    n_jobs=None,
    norm="path",
):
    """
    features: precomputed low-dim features (e.g. PCA of signal features) of shape (N, F)
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    features, _ = extract_interpretable_features(signals, use_catch22=True)

    templates, template_names, medoid_index_map = [], [], {}

    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0:
            medoid_index_map[lab] = []
            continue

        X = features[idxs]
        sigs = [signals[i] for i in idxs]

        # cluster in feature space
        k = min(n_clusters, len(idxs))
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        c = km.fit_predict(X)

        # sample proportional to cluster size
        sel = []
        for ci in range(k):
            cidx = np.where(c == ci)[0]
            n_take = max(1, int(total_target * len(cidx) / len(idxs)))
            pick = random.sample(list(cidx), min(len(cidx), n_take))
            sel.extend(pick)

        sub_signals = [sigs[i] for i in sel]
        sub_idxs = idxs[sel]

        # pairwise DTW within sampled subset
        D = get_pairwise_DTW(
            sub_signals, window_frac=window_frac, n_jobs=n_jobs, norm=norm
        )
        sums = np.sum(D, axis=1)
        order = np.argsort(sums)[:n_medoids]
        chosen = [int(sub_idxs[o]) for o in order]

        medoid_index_map[lab] = chosen
        for k, gidx in enumerate(chosen):
            templates.append(signals[gidx])
            template_names.append(f"medoid_{lab}_{k}")

    return templates, template_names, medoid_index_map


def compute_class_medoids(
    signals,
    labels,
    n_medoids=1,
    window_frac=0.10,
    n_jobs=None,
    norm="path",
    total_target=1000,
    sampling_strategy="random",
    seed=385,
):
    """
    Compute medoid signal(s) per class.

    signals: list of 1D arrays (aligned to labels)
    labels: array-like of same length
    n_medoids: number of medoids to return per class (take lowest sum-of-distances)
    Returns: (templates, template_names, medoid_index_map)
      - templates: list of 1D arrays (the medoid signals)
      - template_names: list of strings naming each template (e.g. "medoid_<class>_0")
      - medoid_index_map: dict class_label -> list of original indices of medoids
    """
    rng = np.random.default_rng(seed)

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    templates = []
    template_names = []
    medoid_index_map = {}

    for lab in unique_labels:
        idxs = np.where(labels == lab)[0].tolist()
        if len(idxs) == 0:
            medoid_index_map[lab] = []
            continue
        # Extract sub-signals and compute pairwise DTW
        sub_signals = [signals[i] for i in idxs]
        # Use path normalization (recommended)
        D = get_pairwise_DTW(
            sub_signals, window_frac=window_frac, n_jobs=n_jobs, norm=norm
        )
        sums = np.sum(D, axis=1)
        order = np.argsort(sums)[:n_medoids]
        chosen = [idxs[int(i)] for i in order]  # map back to global indices
        medoid_index_map[lab] = chosen
        for k, gidx in enumerate(chosen):
            templates.append(signals[gidx])
            template_names.append(f"medoid_{lab}_{k}")
    return templates, template_names, medoid_index_map


def thousands_formatter(x, pos):
    if x == 0:
        return "0"
    elif x >= 1e6:
        return f"{x/1e6:.1f}M" if x % 1e6 != 0 else f"{int(x/1e6)}M"
    elif x >= 1e3:
        return f"{x/1e3:.1f}K" if x % 1e3 != 0 else f"{int(x/1e3)}K"
    else:
        return str(int(x))


def get_peptide_sequences():
    return {
        "ßCAT": "YLDSGIHSGAC",
        "ßCATD": "YLDSDIHSGAC",
        "ßCATW": "YLDSWIHSGAC",
        "ßCATGG": "YLDSGGHSGAC",
        "ßCATL": "YLDSLIHSGAC",
        "ßCATWW": "YLDSWWHSGAC",
        "ßCATWWW": "YLDSWWWHSGAC",
        "BCAR3": "IMDRTPEKLC",
        "ßCAT20": "YLDSGIHSGACKTGKHGEGC",
        "ßCAT30": "YLDSGIHSGACKTGKHGEGCEAVKLQRDLC",
        "ßCAT35": "YLDSGIHSGACKTGKHGEGCEAVKLQRDLGCDLQH",
        "CHGnegG": "YEYEYEGEYEYEC",
        "CHGposD": "RKHGRKWHDKRKC",
        "CHGposL": "RKHGRKWHLKRKC",
        "SZsmall": "GSGAGSSGGSIGGRC",
        "SZlarge": "GFLFPEHTYFFRC",
        "βCATHphil": "YLDSGIHSGAKDKKC",
        "βCATHmod": "YLDSGIHSGALKGQC",
        "βCATHphob": "YLDSGIHSGAKKKAC",
        "βCATins1": "VVVVGYLDSGIHSGAC",
        "βCATins2": "YLDSGIHSGAGVVVVC",
    }


def plot_peptide_medoids_panel(
    medoids_dict,
    std_dict,
    n_dict,
    all_signals_dict,
    class_order=None,
    resamp_len=300,
    colors=None,
    outdir=None,
    figsize=None,
    peptide_sequences=None,
    dtw_limit=None,
):
    if class_order is None:
        class_order = list(medoids_dict.keys())

    n_peptides = len(class_order)

    font_size = 5
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )

    # If figsize explicitly None keep behavior; otherwise use provided figsize
    if figsize is None:
        # use ~0.9" per row, double-column width ~7.2"
        n_rows = int(np.ceil(n_peptides / 2))
        figsize = (6.7, 1.2 * n_rows + 0.6)

    if peptide_sequences is None:
        peptide_sequences = get_peptide_sequences()  # or pass explicitly

    # Handle base names for replicates
    base_names = []
    for lab in class_order:
        if "-rep" in lab or "-run" in lab:
            base = lab.split("-")[0]
        else:
            base = lab
        base_names.append(base)

    if colors is None:
        colors = sns.color_palette("colorblind", n_peptides)

    t = np.linspace(0, 1, resamp_len)
    xticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    # flatten signals safely
    all_y_vals = np.concatenate(
        [np.concatenate(all_signals_dict[p]) for p in class_order]
    )
    lower_percentile = np.percentile(all_y_vals, 1)
    upper_percentile = np.percentile(all_y_vals, 99)
    padding = 0.05 * (upper_percentile - lower_percentile)
    global_min, global_max = lower_percentile - padding, upper_percentile + padding

    n_bins = 30
    y_clipped = all_y_vals[
        (all_y_vals >= lower_percentile) & (all_y_vals <= upper_percentile)
    ]
    bin_edges = np.linspace(y_clipped.min(), y_clipped.max(), n_bins + 1)

    all_counts = [
        np.histogram(np.concatenate(all_signals_dict[p]), bins=bin_edges)[0]
        for p in class_order
    ]
    global_hist_max = max(c.max() for c in all_counts) if all_counts else 1
    hist_xlim = (0, global_hist_max * 1.05)

    n_rows = int(np.ceil(n_peptides / 2))
    n_cols = 4  # two peptides per row, each peptide has main + hist => 4 columns

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=n_rows,
        ncols=n_cols,
        figure=fig,
        width_ratios=[3, 1, 3, 1],
        height_ratios=[1] * n_rows,
        hspace=0.25,
        wspace=0.15,
    )

    LABEL_FONT = font_size
    TITLE_FONT = font_size + 1
    TICK_FONT = font_size
    last_row_has_two = n_peptides % 2 == 0

    # fill left column top->bottom first, then right column top->bottom
    for idx in range(n_peptides):
        # map linear idx into column-major placement
        if idx < n_rows:
            # left column
            row = idx
            col_block = 0
            item_idx = idx
        else:
            # right column
            row = idx - n_rows
            col_block = 2
            item_idx = idx

        # safety (shouldn't be needed, but keep defensive)

        full_label = class_order[item_idx]
        base_name = base_names[item_idx]
        seq = peptide_sequences.get(base_name, "")
        color = colors[idx]

        ax_main = fig.add_subplot(gs[row, col_block])
        medoid = np.array(medoids_dict[full_label])
        std_info = std_dict[full_label]
        N = n_dict[full_label]

        if isinstance(std_info, dict) and std_info.get("type") == "envelope":
            lower = np.array(std_info["lower"])
            upper = np.array(std_info["upper"])
            median = np.array(std_info["median"])
            ax_main.fill_between(t, lower, upper, color=color, alpha=0.25, linewidth=0)

            # optionally plot outer envelope if present
            if std_info.get("outer") is not None:
                outer_low, outer_high = std_info["outer"]
                ax_main.fill_between(
                    t, outer_low, outer_high, color=color, alpha=0.12, linewidth=0
                )
            # plot median
            ax_main.plot(t, medoid, color=color, linewidth=1.2)  # medoid (exemplar)
            # ax_main.plot(t, median, color=color, linewidth=0.9, linestyle='--', alpha=0.9)
            # faintly overlay a few aligned examples
            for ex in std_info.get("aligned_examples", []):
                ax_main.plot(t, ex, color=color, linewidth=0.5, alpha=0.25)
        else:
            # legacy behavior: std_info is 1D array of std
            std = np.array(std_info)
            ax_main.fill_between(t, medoid - std, medoid + std, color=color, alpha=0.25)
            ax_main.plot(t, medoid, color=color, linewidth=1.2)

        if seq:
            # Place name slightly left of center, sequence slightly right
            ax_main.text(
                0.49,
                1.04,
                full_label,
                transform=ax_main.transAxes,
                ha="right",
                va="bottom",
                fontsize=TITLE_FONT,
                fontweight="bold",
                color=color,
            )
            if len(seq) == 11:
                pre = seq[:4]
                center = seq[4]
                post = seq[5:]
                seq_formatted = f"{pre}$\\mathbf{{{center}}}${post}"
            else:
                seq_formatted = seq

            ax_main.text(
                0.51,
                1.04,
                seq_formatted,
                transform=ax_main.transAxes,
                ha="left",
                va="bottom",
                fontsize=TITLE_FONT - 0.5,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="none",
                    edgecolor=color,
                    linewidth=0.7,
                ),
                color="black",
            )
        else:
            ax_main.text(
                0.5,
                1.04,
                full_label,
                transform=ax_main.transAxes,
                ha="center",
                va="bottom",
                fontsize=TITLE_FONT,
                fontweight="bold",
                color=color,
            )

        # N label (top-right, subtle)
        ax_main.text(
            0.99,
            0.95,
            f"N={N}",
            transform=ax_main.transAxes,
            fontsize=TICK_FONT,
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, edgecolor="none"
            ),
        )

        # Styling
        ax_main.spines[["top", "right"]].set_visible(False)
        ax_main.set_ylim(global_min, global_max)

        # Y-label only on the first (left) peptide column
        if col_block == 0:
            ax_main.set_ylabel("Norm. current", fontsize=LABEL_FONT)
            ax_main.tick_params(axis="y", labelsize=TICK_FONT, pad=1)
        else:
            # hide y-axis ticks/labels on the second column to avoid duplication
            ax_main.set_yticklabels([])
            ax_main.tick_params(axis="y", labelsize=TICK_FONT, pad=1)

        # X-axis ticks: only if last row AND last row has two peptides
        show_x = (row == n_rows - 1) and last_row_has_two
        ax_main.set_xticks(xticks)
        if show_x:
            ax_main.set_xlabel("Warped time", fontsize=LABEL_FONT)
            ax_main.tick_params(axis="x", labelsize=TICK_FONT, pad=1)
        else:
            # keep tick positions but hide labels
            ax_main.set_xticklabels([])
            ax_main.tick_params(axis="x", labelsize=TICK_FONT, pad=1)

        ax_hist = fig.add_subplot(gs[row, col_block + 1])
        vals = np.concatenate(all_signals_dict[full_label])
        ax_hist.hist(
            vals, bins=bin_edges, orientation="horizontal", color=color, alpha=0.7
        )

        ax_hist.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        ax_hist.set_xlim(hist_xlim)
        ax_hist.spines[["top", "right", "left"]].set_visible(False)
        ax_hist.tick_params(axis="y", left=False, labelleft=False)
        ax_hist.tick_params(axis="x", labelsize=TICK_FONT - 0.5, pad=1)

        # only show histogram xlabel when we show x on main (i.e., last full row)
        if show_x:
            ax_hist.set_xlabel("Count", fontsize=LABEL_FONT, labelpad=1)
            plt.setp(ax_hist.get_xticklabels(), rotation=30, ha="right")
        else:
            ax_hist.tick_params(axis="x", labelbottom=False)

        ax_hist.set_ylim(global_min, global_max)

    # If odd number of peptides, hide the unused axes in the last row (the rightmost pair)
    if n_peptides % 2 == 1:
        empty_col_block = 2  # second peptide slot in last row
        # hide main
        ax_empty_main = fig.add_subplot(gs[n_rows - 1, empty_col_block])
        ax_empty_main.set_axis_off()
        # hide hist
        ax_empty_hist = fig.add_subplot(gs[n_rows - 1, empty_col_block + 1])
        ax_empty_hist.set_axis_off()

    plt.tight_layout(rect=[0.12, 0.02, 0.99, 0.99])

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, "fig2a_peptide_medoids")
        plt.savefig(out_path + ".png", dpi=600, bbox_inches="tight")
        plt.savefig(out_path + ".svg", bbox_inches="tight")
        logger.info(f"Saved figure to {out_path}")
    plt.close()
    return fig


def warp_signal_to_reference(ref, sig, window_frac=0.1):
    """
    Warp `sig` to the time-axis of `ref` using the DTW path computed by dtw_path().
    For each index i in ref, average all sig[j] that map to i.
    Returns aligned signal of length len(ref).
    """
    path = dtw_path(ref, sig, window_frac=window_frac)
    Lr = len(ref)
    buckets = [[] for _ in range(Lr)]
    for i_ref, j_sig in path:
        # guard indices
        if 0 <= i_ref < Lr and j_sig >= 0:
            buckets[i_ref].append(sig[j_sig])
    aligned = np.array(
        [np.mean(b) if len(b) > 0 else np.nan for b in buckets], dtype=float
    )
    # fill any NaNs (interpolate)
    if np.isnan(aligned).any():
        good = ~np.isnan(aligned)
        if good.sum() >= 2:
            xi = np.flatnonzero(~good)
            xp = np.flatnonzero(good)
            aligned[~good] = np.interp(xi, xp, aligned[good])
        else:
            aligned = np.nan_to_num(aligned)
    return aligned


def warp_signal_to_reference_numba(ref, sig, window_frac=0.10):
    """
    Use the numba dtw path to warp 'sig' onto ref's time axis.
    Returns aligned numpy array of length len(ref).
    """
    ref = np.asarray(ref, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    if ref.size == 0 or sig.size == 0:
        return np.zeros(ref.shape[0], dtype=np.float64)

    path_i, path_j = dtw_path_numba(ref, sig, window_frac)
    # accumulate sums and counts per ref index
    Lr = ref.shape[0]
    sums = np.zeros(Lr, dtype=np.float64)
    counts = np.zeros(Lr, dtype=np.int32)

    # path_i/path_j are same length
    for k in range(path_i.shape[0]):
        ii = int(path_i[k])
        jj = int(path_j[k])
        if ii >= 0 and ii < Lr and jj >= 0 and jj < sig.shape[0]:
            sums[ii] += sig[jj]
            counts[ii] += 1

    # compute averages (use nan where no counts)
    aligned = np.empty(Lr, dtype=np.float64)
    for i in range(Lr):
        if counts[i] > 0:
            aligned[i] = sums[i] / counts[i]
        else:
            aligned[i] = np.nan

    # interpolate NaNs if any
    if np.isnan(aligned).any():
        good = ~np.isnan(aligned)
        if good.sum() >= 2:
            xi = np.flatnonzero(~good)
            xp = np.flatnonzero(good)
            aligned[~good] = np.interp(xi, xp, aligned[good])
        else:
            # fallback: replace nan with zero
            aligned = np.nan_to_num(aligned)

    return aligned


def compute_std_per_class(
    signals,
    labels,
    medoid_indices,
    resamp_len=300,
    method="std",
    band=(25, 75),
    outer_percent=None,
    n_examples=5,
    seed=385,
):
    """
    Compute class medoids (re-checked) and either:
      - 'std' method: compute pointwise std across signals (existing behaviour)
      - 'dtw_align' method: DTW-warp all signals to medoid time axis, then compute median and percentile envelope

    Returns:
      medoids_dict, std_or_envelope_dict, n_dict

    std_or_envelope_dict[lab] is either:
      - 1D numpy array (std), or
      - dict with keys: 'type':'envelope', 'median', 'lower', 'upper', 'aligned_examples' (list of aligned arrays)
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    medoids_dict = {}
    std_dict = {}
    n_dict = {}

    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        n = len(idxs)
        n_dict[lab] = n
        if n == 0:
            medoids_dict[lab] = np.zeros(resamp_len)
            std_dict[lab] = np.zeros(resamp_len)
            continue

        class_signals = [np.asarray(signals[i]) for i in idxs]
        # ensure resampled lengths (if not, resample or raise)
        lengths_ok = all(len(s) == resamp_len for s in class_signals)
        if not lengths_ok:
            # try to resample each to resamp_len using numpy interp
            class_signals = [
                (
                    np.interp(
                        np.linspace(0, 1, resamp_len), np.linspace(0, 1, len(s)), s
                    )
                    if len(s) != resamp_len
                    else s
                )
                for s in class_signals
            ]

        # Determine medoid (use provided medoid index if present)
        if lab in medoid_indices and len(medoid_indices[lab]) > 0:
            medoid_local_idx = medoid_indices[lab][
                0
            ]  # index relative to `signals` input
            medoid = np.asarray(signals[medoid_local_idx])
            # if medoid length differs, resample
            if len(medoid) != resamp_len:
                medoid = np.interp(
                    np.linspace(0, 1, resamp_len),
                    np.linspace(0, 1, len(medoid)),
                    medoid,
                )
        else:
            # fallback: use pointwise median
            arr = np.vstack(class_signals)
            medoid = np.median(arr, axis=0)

        medoids_dict[lab] = medoid

        if method == "std":
            arr = np.vstack(class_signals)
            std = np.std(arr, axis=0)
            std_dict[lab] = std
        elif method == "dtw_align":
            # warp every class signal to medoid axis
            aligned = []
            for s in class_signals:
                a = warp_signal_to_reference_numba(medoid, s, window_frac=0.1)
                aligned.append(a)
            A = np.vstack(aligned)  # (N, L)
            lowp, highp = band
            lower = np.percentile(A, lowp, axis=0)
            upper = np.percentile(A, highp, axis=0)
            median = np.percentile(A, 50, axis=0)
            # optionally wide outer envelope
            outer = None
            if outer_percent is not None:
                outer_low = (100 - outer_percent) / 2.0
                outer_high = 100 - outer_low
                outer = (
                    np.percentile(A, outer_low, axis=0),
                    np.percentile(A, outer_high, axis=0),
                )

            # sample a few aligned examples for faint overlay
            if n_examples is not None and n_examples > 0:
                if n <= n_examples:
                    examples = [A[i] for i in range(n)]
                else:
                    sel = rng.choice(np.arange(n), size=n_examples, replace=False)
                    examples = [A[i] for i in sel]
            else:
                examples = []

            std_dict[lab] = {
                "type": "envelope",
                "median": median,
                "lower": lower,
                "upper": upper,
                "outer": outer,
                "aligned_examples": examples,
            }
        else:
            raise ValueError(f"Unknown method={method}")

    return medoids_dict, std_dict, n_dict


def sample_per_class(y, classes, per_class=100, seed=385):
    rng = np.random.default_rng(seed)
    sampled_idxs = []

    for lbl in classes:
        idxs = np.where(y == lbl)[0]
        if per_class <= 0 or len(idxs) <= per_class:
            sampled = idxs.tolist()
        else:
            sampled = rng.choice(idxs, per_class, replace=False).tolist()
        sampled_idxs.extend(sampled)

    return sampled_idxs


def plot_signal_distributions_supplementary(
    all_signals_dict, class_order, colors=None, outdir=None, figsize=(6, 7)
):
    """
    Supplementary figure: overlaid KDEs of all signal values per peptide.

    Parameters
    ----------
    all_signals_dict : dict
        {class: list of 1D signals}
    class_order : list
        e.g., ["ßCAT", "ßCATD", "ßCATW", "ßCATL"]
    colors : list, optional
        Same as main figure
    outdir : str
    figsize : tuple
    """
    if colors is None:
        colors = sns.color_palette("colorblind", len(class_order))

    # Get global y-limits (current values)
    all_vals_flat = np.concatenate(
        [np.concatenate(all_signals_dict[cls]) for cls in class_order]
    )

    y_min = np.percentile(all_vals_flat, 1)
    y_max = np.percentile(all_vals_flat, 99)

    padding = 0.01 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    # Evaluate KDE on this robust range
    y_grid = np.linspace(y_min, y_max, 500)

    fig, ax = plt.subplots(figsize=figsize)

    # Store legend handles
    legend_elements = []

    for cls, color in zip(class_order, colors):
        vals = np.concatenate(all_signals_dict[cls])
        N = len(all_signals_dict[cls])  # number of events

        kde = gaussian_kde(vals, bw_method="scott")
        density = kde(y_grid)

        # Plot line and fill
        ax.plot(density, y_grid, color=color, linewidth=1.8)
        ax.fill_betweenx(y_grid, 0, density, color=color, alpha=0.15)

        legend_elements.append(
            Line2D([0], [0], color=color, lw=2, label=f"{cls} (N={N})")
        )

    ax.set_xlabel("Density", fontsize=9)
    ax.set_ylabel("Normalized current (pA)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=legend_elements,
        frameon=False,
        fontsize=8,
        loc="upper right",
        handlelength=1.2,
        handletextpad=0.5,
    )

    plt.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, "supp_fig_vertical_kdes")
        plt.savefig(out_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(out_path + ".svg", bbox_inches="tight")
        logger.info(f"Saved vertical KDE supplementary figure to {out_path}")
    plt.close()
    return fig


def plot_median_distributions_supplementary(
    all_signals_dict,
    class_order,
    colors=None,
    outdir=None,
    figsize=(6, 7),
    show_raw_data=False,  # adds jittered points
    clip_percentiles=(0.5, 99.5),
):
    """
    Supplementary figure: vertical violin plots of per-event median current.

    Parameters
    ----------
    all_signals_dict : dict
        {class: list of 1D signals}
    class_order : list
        Peptide order
    colors : list
        Same as main figure
    show_raw_data : bool
        Overlay individual median values (recommended)
    """
    medians_by_class = {}
    n_by_class = {}
    all_medians = []
    for cls in class_order:
        medians = [np.median(sig) for sig in all_signals_dict[cls]]
        medians = np.array(medians)
        medians_by_class[cls] = medians
        n_by_class[cls] = len(medians)
        all_medians.append(medians)

    # Flatten for global percentile calculation
    all_medians_flat = np.concatenate(all_medians)
    lower_clip, upper_clip = np.percentile(all_medians_flat, clip_percentiles)

    # Clip data for visualization (does not affect analysis)
    clipped_medians_by_class = {}
    for cls in class_order:
        clipped = np.clip(medians_by_class[cls], lower_clip, upper_clip)
        clipped_medians_by_class[cls] = clipped

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data lists
    data_for_violin = [clipped_medians_by_class[cls] for cls in class_order]
    data_for_box = [
        medians_by_class[cls] for cls in class_order
    ]  # use original for stats, but we'll clip box too for consistency

    # Use clipped data for both violin and boxplot for visual consistency
    data_for_box = data_for_violin

    # Violin plot
    parts = ax.violinplot(
        data_for_violin,
        positions=range(len(class_order)),
        vert=True,
        widths=0.8,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    # Color violins
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor("none")

    # Boxplot inside violin
    bp = ax.boxplot(
        data_for_box,
        positions=range(len(class_order)),
        widths=0.2,
        patch_artist=True,
        whis=[5, 95],  # whiskers at 5th–95th percentiles
        showfliers=False,  # hide outlier points
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        manage_ticks=False,
    )

    if show_raw_data:
        np.random.seed(42)
        for i, cls in enumerate(class_order):
            medians = medians_by_class[cls]
            x_jitter = np.random.normal(i, 0.04, size=len(medians))
            ax.scatter(x_jitter, medians, color=colors[i], s=4, alpha=0.4, linewidth=0)

    # Set tight y-limits based on clipped range
    y_pad = (upper_clip - lower_clip) * 0.05
    ax.set_ylim(lower_clip - y_pad, upper_clip + y_pad)

    # Styling
    ax.set_xticks(range(len(class_order)))
    ax.set_xticklabels(class_order, fontsize=9)
    ax.set_ylabel("Median current (pA)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)

    # Add N labels below x-axis
    for i, cls in enumerate(class_order):
        ax.text(
            i,
            ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.07,
            f"N={n_by_class[cls]}",
            ha="center",
            va="top",
            fontsize=8,
            color="black",
        )

    plt.tight_layout(pad=0.5)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, "supp_fig_median_distributions")
        plt.savefig(out_path + ".png", dpi=600, bbox_inches="tight")
        plt.savefig(out_path + ".svg", bbox_inches="tight")
        logger.info(f"Saved median distribution supplementary figure to {out_path}")
    plt.close()
    return fig


def plot_std_distributions_supplementary(
    all_signals_dict,
    class_order,
    colors=None,
    outdir=None,
    figsize=(6, 7),
    show_raw_data=False,
    clip_percentiles=(0.5, 99.5),
):
    """
    Supplementary figure: vertical violin plots of per-event standard deviation.
    """
    stds_by_class = {}
    n_by_class = {}
    all_stds = []
    for cls in class_order:
        stds = [np.std(sig) for sig in all_signals_dict[cls]]
        stds = np.array(stds)
        stds_by_class[cls] = stds
        n_by_class[cls] = len(stds)
        all_stds.append(stds)

    all_stds_flat = np.concatenate(all_stds)
    lower_clip, upper_clip = np.percentile(all_stds_flat, clip_percentiles)

    clipped_stds_by_class = {}
    for cls in class_order:
        clipped = np.clip(stds_by_class[cls], lower_clip, upper_clip)
        clipped_stds_by_class[cls] = clipped

    fig, ax = plt.subplots(figsize=figsize)

    data_for_violin = [clipped_stds_by_class[cls] for cls in class_order]
    data_for_box = data_for_violin  # consistent clipping

    # Violin
    parts = ax.violinplot(
        data_for_violin,
        positions=range(len(class_order)),
        vert=True,
        widths=0.8,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor("none")

    # Boxplot
    ax.boxplot(
        data_for_box,
        positions=range(len(class_order)),
        widths=0.2,
        patch_artist=True,
        whis=[5, 95],
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        manage_ticks=False,
    )

    if show_raw_data:
        np.random.seed(42)
        for i, cls in enumerate(class_order):
            stds = stds_by_class[cls]
            x_jitter = np.random.normal(i, 0.04, size=len(stds))
            ax.scatter(x_jitter, stds, color=colors[i], s=4, alpha=0.4, linewidth=0)

    y_pad = (upper_clip - lower_clip) * 0.05
    ax.set_ylim(lower_clip - y_pad, upper_clip + y_pad)

    ax.set_xticks(range(len(class_order)))
    ax.set_xticklabels(class_order, fontsize=9)
    ax.set_ylabel("Standard deviation (pA)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)

    # N labels
    for i, cls in enumerate(class_order):
        ax.text(
            i,
            ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.07,
            f"N={n_by_class[cls]}",
            ha="center",
            va="top",
            fontsize=8,
            color="black",
        )

    plt.tight_layout(pad=0.5)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, "supp_fig_std_distributions")
        plt.savefig(out_path + ".png", dpi=600, bbox_inches="tight")
        plt.savefig(out_path + ".svg", bbox_inches="tight")
        logger.info(f"Saved std distribution supplementary figure to {out_path}")
    plt.close()
    return fig


def plot_median_vs_std_contour_supplementary(
    all_signals_dict,
    class_order,
    colors=None,
    outdir=None,
    figsize=(6, 5),
    alpha=0.6,
    point_size=8,
    n_levels=6,
    alpha_fill=0.9,
):
    """
    Supplementary figure: median vs std per signal, colored by class.
    Uses scatter with optional density contours.
    """
    if colors is None:
        colors = sns.color_palette("colorblind", len(class_order))

    fig, ax = plt.subplots(figsize=figsize)

    # Precompute all data to set consistent axis limits
    all_medians = []
    all_stds = []
    for cls in class_order:
        for sig in all_signals_dict[cls]:
            all_medians.append(np.median(sig))
            all_stds.append(np.std(sig))
    all_medians = np.array(all_medians)
    all_stds = np.array(all_stds)

    clip_pct = (0.1, 99.9)
    xlim = np.percentile(all_medians, clip_pct)
    ylim = np.percentile(all_stds, clip_pct)

    # Plot KDE contours per class
    for i, cls in enumerate(class_order):
        medians = np.array([np.median(sig) for sig in all_signals_dict[cls]])
        stds = np.array([np.std(sig) for sig in all_signals_dict[cls]])

        # Clip to global limits to avoid extreme outliers distorting KDE
        mask = (
            (medians >= xlim[0])
            & (medians <= xlim[1])
            & (stds >= ylim[0])
            & (stds <= ylim[1])
        )
        medians_clipped = medians[mask]
        stds_clipped = stds[mask]

        # KDE contour — filled + lines
        sns.kdeplot(
            x=medians_clipped,
            y=stds_clipped,
            ax=ax,
            color=colors[i],
            fill=False,
            alpha=alpha_fill,
            levels=n_levels,
            linewidths=0.8,
            zorder=2,
        )

    # Axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Labels & styling
    ax.set_xlabel("Median current (pA)", fontsize=9)
    ax.set_ylabel("Standard deviation (pA)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors[i], alpha=alpha_fill, label=cls)
        for i, cls in enumerate(class_order)
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False, loc="upper right")

    plt.tight_layout(pad=0.5)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, "supp_fig_median_vs_std_contour")
        plt.savefig(out_path + ".png", dpi=600, bbox_inches="tight")
        plt.savefig(out_path + ".svg", bbox_inches="tight")
        logger.info(f"Saved median vs std contour supplementary figure to {out_path}")
    plt.close()
    return fig


def select_top_nonredundant_features_greedy(
    features_df, labels, n_features=5, corr_threshold=0.95
):
    features_df = features_df.loc[:, features_df.std() > 1e-6]
    features_df = features_df.fillna(features_df.median())

    X = features_df.values
    y = np.array(labels)

    # ANOVA ranking
    f_vals, _ = f_classif(X, y)
    feature_names = features_df.columns.tolist()
    anova_rank = pd.Series(f_vals, index=feature_names).sort_values(ascending=False)

    selected = []
    corr = features_df.corr().abs()

    for feat in anova_rank.index:
        # Check if highly correlated with any already selected
        redundant = False
        for sel in selected:
            if corr.loc[feat, sel] > corr_threshold:
                redundant = True
                break
        if not redundant:
            selected.append(feat)
        if len(selected) >= n_features:
            break

    return selected, features_df[selected]


def select_top_nonredundant_features(
    features_df, labels, n_features=5, corr_threshold=0.95
):
    """
    Select top n_features that are:
    - Most discriminative (high ANOVA F-value)
    - Not highly correlated with each other
    """
    # Remove constant or near-constant features
    features_df = features_df.loc[:, (features_df.std() > 1e-16)]

    # Handle NaNs (your extractor may produce some)
    features_df = features_df.fillna(features_df.median())

    X = features_df.values
    y = np.array(labels)

    # ANOVA F-value ranking
    f_vals, _ = f_classif(X, y)
    feature_names = features_df.columns.tolist()
    anova_rank = pd.Series(f_vals, index=feature_names).sort_values(ascending=False)

    # Compute correlation matrix
    corr = features_df.corr().abs()

    # Clip correlation values to [0, 1] to avoid numerical issues
    corr = corr.clip(lower=0.0, upper=1.0)

    # Remove redundant features: hierarchical clustering on correlation
    # Convert to distance: 1 - correlation
    dist = 1 - corr
    np.fill_diagonal(dist.values, 0)

    dist = dist.clip(lower=0.0)

    # Proceed with clustering
    try:
        linkage_matrix = linkage(squareform(dist.values), method="average")
    except ValueError as e:
        if "contains negative distances" in str(e):
            raise ValueError(
                "Negative distances in linkage. This usually happens due to "
                "numerical instability in correlation matrix. Try reducing "
                "feature dimensionality or checking for constant features."
            ) from e
        else:
            raise

    clusters = fcluster(linkage_matrix, 1 - corr_threshold, criterion="distance")

    # For each cluster, pick the feature with highest ANOVA score
    selected = []
    for cluster_id in np.unique(clusters):
        cluster_features = [
            f for f, c in zip(feature_names, clusters) if c == cluster_id
        ]
        best_in_cluster = anova_rank[cluster_features].idxmax()
        selected.append(best_in_cluster)

    # Sort selected by ANOVA score and take top n
    selected = [f for f in anova_rank.index if f in selected][:n_features]
    return selected, features_df[selected]


def shorten_feature_name(feature_name):
    """Shorten feature names for better display, especially catch22 features"""
    if not isinstance(feature_name, str):
        return str(feature_name)

    # Handle catch22 features specifically
    prefix = ""
    if feature_name.startswith("catch22_"):
        prefix = "catch22_"
    elif feature_name.startswith("c22_"):
        prefix = "c22_"

    if prefix:
        base_name = feature_name.replace(prefix, "")
        # Standardized short names based on the catch22 documentation table
        replacements = {
            "DN_HistogramMode_5": "mode_5",
            "DN_HistogramMode_10": "mode_10",
            "DN_OutlierInclude_p_001_mdrmd": "outlier_timing_pos",
            "DN_OutlierInclude_n_001_mdrmd": "outlier_timing_neg",
            "first1e_acf_tau": "acf_timescale",
            "firstMin_acf": "acf_first_min",
            "SP_Summaries_welch_rect_area_5_1": "low_freq_power",
            "SP_Summaries_welch_rect_centroid": "centroid_freq",
            "FC_LocalSimple_mean3_stderr": "forecast_error",
            "FC_LocalSimple_mean1_tauresrat": "whiten_timescale",
            "MD_hrv_classic_pnn40": "high_fluctuation",
            "SB_BinaryStats_mean_longstretch1": "stretch_high",
            "SB_BinaryStats_diff_longstretch0": "stretch_decreasing",
            "SB_MotifThree_quantile_hh": "entropy_pairs",
            "CO_HistogramAMI_even_2_5": "ami2",
            "CO_trev_1_num": "trev",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi": "ami_timescale",
            "SB_TransitionMatrix_3ac_sumdiagcov": "transition_variance",
            "PD_PeriodicityWang_th001": "periodicity",
            "CO_Embed2_Dist_tau_d_expfit_meandiff": "embedding_dist",
            "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": "rs_range",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": "dfa",
        }

        # Check for exact matches first
        for full_name, short_name in replacements.items():
            if base_name == full_name:
                return short_name

        # Check for partial matches if no exact match found
        for full_name, short_name in replacements.items():
            if full_name in base_name:
                # Preserve suffixes like _ac, _sc if present
                suffix = base_name.replace(full_name, "")
                if suffix.startswith("_ac") or suffix.startswith("_sc"):
                    return f"{short_name}{suffix[:3]}"
                return short_name

        # If no match found, return a shortened version of the base name
        return base_name[:15] + "..." if len(base_name) > 15 else base_name

    # Handle other feature types
    if feature_name.startswith("dtw_template"):
        return feature_name.replace("dtw_template_", "DTW_")[:12] + "..."

    if feature_name.startswith("pcc_"):
        return feature_name.replace("pcc_", "PCC_")

    # Generic shortening for long names
    if len(feature_name) > 30:
        return feature_name[:25] + "..."

    return feature_name


def plot_feature_distributions_panel(
    features_df,
    labels,
    selected_features,
    class_order,
    colors,
    outdir=None,
    figsize_base=(6.7, 2),
    clip_percentiles=(1, 99),
    stat_test=False,
):
    """
    Figure 2B: Distribution of top 5 features across 4 classes.
    Horizontal layout, violin plots with embedded boxplots (no jitter for large N)..
    """

    font_size = 5
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )

    n_features = len(selected_features)
    if n_features == 0:
        logger.warning(
            "No features selected to plot; skipping feature distribution panel."
        )
        return None

    n_cols = 5
    n_rows = max(1, (n_features + n_cols - 1) // n_cols)  # ceiling division
    n_classes = len(class_order)

    if stat_test:
        # Auto-select strategy to prevent clutter
        if n_classes > 5:
            comp_strategy = "reference"
            ref_class = class_order[0]  # Default to first class
            logger.warning(
                f"{n_classes} classes detected. Using reference-based comparisons "
                f"(vs '{ref_class}') to avoid visual clutter. Set pairwise_ref_class explicitly to control this."
            )
        else:
            comp_strategy = "all_pairs"
            ref_class = None
    else:
        comp_strategy = "none"
        ref_class = None

    if comp_strategy == "all_pairs":
        pairs_to_test = list(combinations(range(n_classes), 2))
    elif comp_strategy == "reference":
        ref_idx = class_order.index(ref_class)
        pairs_to_test = [(ref_idx, i) for i in range(n_classes) if i != ref_idx]
    else:
        pairs_to_test = []

    fig_width = figsize_base[0]
    fig_height = figsize_base[1] * n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height), sharey=False
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, feat in enumerate(selected_features):
        ax = axes[idx]
        data_by_class = []
        all_vals = []
        for cls in class_order:
            mask = np.array(labels) == cls
            vals = features_df.loc[mask, feat].values
            data_by_class.append(vals)
            all_vals.extend(vals)

        all_vals = np.concatenate(data_by_class)

        if np.all(np.isnan(all_vals)):
            ax.text(
                0.5,
                0.5,
                f"{feat}\n(All NaN)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=font_size,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            continue

        finite_vals = all_vals[np.isfinite(all_vals)]
        if finite_vals.size == 0:
            ax.text(
                0.5,
                0.5,
                f"{feat}\n(All invalid)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=font_size,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        lower_clip, upper_clip = np.nanpercentile(all_vals, clip_percentiles)

        clipped_data_by_class = [
            np.clip(vals, lower_clip, upper_clip) for vals in data_by_class
        ]

        clipped_all_vals = np.concatenate(clipped_data_by_class)
        y_min, y_max = np.nanmin(clipped_all_vals), np.nanmax(clipped_all_vals)
        y_range = y_max - y_min
        pad = y_range * 0.05 if y_range > 0 else 0.1
        ax.set_ylim(y_min - pad, y_max + pad)

        parts = ax.violinplot(
            clipped_data_by_class,
            positions=range(len(class_order)),
            vert=True,
            widths=0.8,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )

        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor("none")

        ax.boxplot(
            clipped_data_by_class,
            positions=range(len(class_order)),
            widths=0.2,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
            medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
            manage_ticks=False,
        )

        ax.set_xticks(range(len(class_order)))
        ax.set_xticklabels(class_order, fontsize=font_size, rotation=30, ha="right")

        feat_short = shorten_feature_name(feat)

        ax.set_title(feat_short, fontsize=font_size, pad=6)

        ax.tick_params(axis="y", labelsize=font_size)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["left"].set_linewidth(0.6)

        if stat_test and pairs_to_test:
            clean_data = [vals[np.isfinite(vals)] for vals in data_by_class]

            pvals = []
            valid_pairs = []
            for i, j in pairs_to_test:
                if len(clean_data[i]) < 5 or len(clean_data[j]) < 5:
                    pvals.append(1.0)
                    valid_pairs.append(False)
                    continue

                try:
                    _, p = mannwhitneyu(
                        clean_data[i], clean_data[j], alternative="two-sided"
                    )
                    pvals.append(p)
                    valid_pairs.append(True)
                except Exception as e:
                    print(
                        f"Stat test failed for {feat} ({class_order[i]} vs {class_order[j]}): {str(e)}"
                    )
                    pvals.append(1.0)
                    valid_pairs.append(False)

            if any(valid_pairs):
                pvals_array = np.array(pvals)
                valid_mask = np.array(valid_pairs)

                reject, p_corrected, _, _ = multipletests(
                    pvals_array[valid_mask], alpha=0.05, method="fdr_bh"
                )

                results = [None] * len(pairs_to_test)
                idx_valid = 0
                for k, is_valid in enumerate(valid_pairs):
                    if is_valid:
                        results[k] = (reject[idx_valid], p_corrected[idx_valid])
                        idx_valid += 1

                bar_y_base = ax.get_ylim()[1] * 1.15
                bar_height = y_range * 0.05
                bar_spacing = y_range * 0.08
                max_bar_y = bar_y_base

                for k, (i, j) in enumerate(pairs_to_test):
                    if results[k] is None or not results[k][0]:
                        continue

                    reject_h0, p_val = results[k]
                    if not reject_h0:
                        continue

                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    else:
                        stars = "*"

                    bar_y = bar_y_base + k * bar_spacing
                    max_bar_y = max(max_bar_y, bar_y + bar_height)

                    ax.plot(
                        [i, i, j, j],
                        [bar_y - bar_height / 2, bar_y, bar_y, bar_y - bar_height / 2],
                        lw=0.8,
                        color="black",
                    )

                    mid_x = (i + j) / 2
                    ax.text(
                        mid_x,
                        bar_y + bar_height / 4,
                        stars,
                        ha="center",
                        va="bottom",
                        fontsize=font_size,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
                    )

                current_ylim = ax.get_ylim()
                new_top = max(current_ylim[1] * 1.15, max_bar_y + bar_spacing)
                ax.set_ylim(current_ylim[0], new_top)

    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.8)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        base_name = "fig2b_feature_distributions"
        if stat_test:
            base_name = base_name + "_MUtest"
        out_path = os.path.join(outdir, base_name)
        plt.savefig(out_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(out_path + ".svg", bbox_inches="tight")
        logger.info(f"Saved Figure 2B to {out_path}")
    plt.close()
    return fig


def _compute_point_size(n):
    if n < 1_000:
        return 5.0
    if n < 5_000:
        return 2.5
    if n < 20_000:
        return 1.0
    if n < 100_000:
        return 0.3
    if n < 500_000:
        return 0.1
    return 0.08


def plot_embeddings_panel(
    features_df,
    labels,
    class_order,
    colors,
    outdir=None,
    figsize=(6.7, 2.25),
    seed=385,
    alpha=0.45,
    force_hexbin=False,
):
    X_full = features_df.fillna(features_df.median()).values
    y_full = np.array(labels)
    n_plot = X_full.shape[0]
    s = _compute_point_size(n_plot)
    if n_plot > 100_000:
        alpha = min(alpha, 0.25)
    elif n_plot > 20_000:
        alpha = min(alpha, 0.35)

    rasterize = n_plot > 1_000

    font_size = 5
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )

    X = StandardScaler().fit_transform(features_df.fillna(features_df.median()))
    y = np.array(labels)

    pca = PCA(n_components=2, random_state=seed).fit_transform(X)
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=50,
        early_exaggeration=18,
        learning_rate=200,
    ).fit_transform(X)
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=80, min_dist=0.1)
    umap_emb = reducer.fit_transform(X)

    embeddings = [pca, tsne, umap_emb]
    titles = ["PCA", "t-SNE", "UMAP"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, emb, title in zip(axes, embeddings, titles):
        if force_hexbin and n_plot > 20_000:
            ax.hexbin(emb[:, 0], emb[:, 1], gridsize=200, mincnt=1, linewidths=0.0)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        for i, cls in enumerate(class_order):
            mask = y == cls
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                color=colors[i],
                s=s,
                alpha=alpha,
                rasterized=rasterize,
            )

        ax.set_xlabel(f"{title}1", fontsize=font_size, labelpad=2)
        ax.set_ylabel(f"{title}2", fontsize=font_size, labelpad=2)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)

        ax.grid(False)
        ax.set_facecolor("white")

        ax.margins(0.05)

    plt.tight_layout(pad=0.8)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, "fig2c_embeddings")
        plt.savefig(out_path + ".png", dpi=600, bbox_inches="tight")
        plt.savefig(out_path + ".svg", bbox_inches="tight")
        logger.info(f"Saved Figure 2C to {out_path}")

    plt.close()
    return fig


def plot_dtw_embeddings_panel(
    D_dtw,
    labels,
    class_order,
    colors,
    features_df=None,  # optional: for combined embedding
    outdir=None,
    figsize=(9, 3),
    seed=385,
    embedding_type="dtw",  # "dtw" or "combined"
):
    y = np.array(labels)

    if embedding_type == "dtw":
        mds = MDS(
            n_components=2, dissimilarity="precomputed", random_state=seed, n_jobs=-1
        )
        emb_dtw = mds.fit_transform(D_dtw)

        umap_reducer = umap.UMAP(
            n_components=2,
            metric="precomputed",
            random_state=seed,
            n_neighbors=min(80, len(labels) - 1),
            min_dist=0.1,
            init="random",
        )
        emb_umap = umap_reducer.fit_transform(D_dtw)

        tsne = TSNE(
            n_components=2,
            perplexity=50,
            random_state=seed,
            init=emb_dtw,
            learning_rate=200,
            n_iter=1000,
        ).fit_transform(emb_dtw)

        embeddings = [emb_dtw, tsne, emb_umap]
        titles = ["MDS (DTW)", "t-SNE (DTW)", "UMAP (DTW)"]

    elif embedding_type == "combined" and features_df is not None:
        X_feat = StandardScaler().fit_transform(
            features_df.fillna(features_df.median())
        )

        D_feat = squareform(pdist(X_feat, metric="euclidean"))

        def robust_normalize(D):
            low, high = np.percentile(D, [1, 99])
            D_clip = np.clip(D, low, high)
            return (D_clip - D_clip.min()) / (D_clip.max() - D_clip.min() + 1e-8)

        D_dtw_norm = robust_normalize(D_dtw)
        D_feat_norm = robust_normalize(D_feat)

        # Combine: weighted average (50/50)
        D_combined = 0.5 * D_dtw_norm + 0.5 * D_feat_norm

        # Embed combined distance
        mds = MDS(
            n_components=2, dissimilarity="precomputed", random_state=seed, n_jobs=-1
        )
        emb_dtw = mds.fit_transform(D_combined)

        umap_reducer = umap.UMAP(
            n_components=2,
            metric="precomputed",
            random_state=seed,
            n_neighbors=min(80, len(labels) - 1),
            min_dist=0.1,
            init="random",
        )
        emb_umap = umap_reducer.fit_transform(D_combined)

        tsne = TSNE(
            n_components=2,
            perplexity=50,
            random_state=seed,
            init=emb_dtw,
            learning_rate=200,
            n_iter=1000,
        ).fit_transform(emb_dtw)

        embeddings = [emb_dtw, tsne, emb_umap]
        titles = ["MDS (Combined)", "t-SNE (Combined)", "UMAP (Combined)"]
    else:
        raise ValueError("Invalid embedding_type or missing features_df")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, emb, title in zip(axes, embeddings, titles):
        for i, cls in enumerate(class_order):
            mask = y == cls
            ax.scatter(emb[mask, 0], emb[mask, 1], color=colors[i], s=0.5, alpha=0.5)

        ax.set_xlabel(f"{title.split()[0]}1", fontsize=8, labelpad=2)
        ax.set_ylabel(f"{title.split()[0]}2", fontsize=8, labelpad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)
        ax.grid(False)
        ax.set_facecolor("white")
        ax.margins(0.05)

    plt.tight_layout(pad=0.8)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        prefix = (
            "supp_dtw_embeddings"
            if embedding_type == "dtw"
            else "supp_combined_embeddings"
        )
        out_path = os.path.join(outdir, prefix)
        plt.savefig(out_path + ".png", dpi=600, bbox_inches="tight")
        plt.savefig(out_path + ".svg", bbox_inches="tight")
        logger.info(f"Saved {embedding_type} embedding plot to {out_path}")

    plt.close(fig)
    return fig


def dtw_plots_figure(
    X,
    y,
    selected_classes=None,
    per_class=1000,
    outdir=None,
    resamp_len=300,
    n_jobs=16,
    dtw_features=False,
    seed=385,
    run_rep_separate=False,
    run_ids=None,
    dtw_limit=2000,
):
    if run_rep_separate and run_ids is None:
        raise ValueError("run_ids must be provided when run_rep_separate=True")
    if run_rep_separate and len(run_ids) != len(y):
        raise ValueError("run_ids must have same length as y")

    # choose base classes
    unique_classes = np.unique(y)
    if selected_classes is None:
        classes = unique_classes
    else:
        classes = [c for c in unique_classes if c in set(selected_classes)]
        if not classes:
            raise ValueError("No requested classes found in labels")

    # restrict to candidate indices that belong to chosen base classes
    idx_candidates = [i for i, lbl in enumerate(y) if lbl in classes]

    # When run_rep_separate=True we SAMPLE per (base, run_id) pair so each replicate gets up to per_class samples.
    if run_rep_separate:
        combined_labels = np.array(
            [f"{y[i]}___{run_ids[i]}" for i in idx_candidates], dtype=object
        )
        combined_unique = sorted(np.unique(combined_labels))
        sampled_rel = sample_per_class(
            combined_labels, combined_unique, per_class=per_class, seed=seed
        )
        sampled_idxs = [idx_candidates[i] for i in sampled_rel]
    else:
        base_labels = np.array([y[i] for i in idx_candidates], dtype=object)
        base_unique = sorted(np.unique(base_labels))
        sampled_rel = sample_per_class(
            base_labels, base_unique, per_class=per_class, seed=seed
        )
        sampled_idxs = [idx_candidates[i] for i in sampled_rel]

    # build sampled subset
    X_sub = [X[i] for i in sampled_idxs]
    labels_sub_original = [y[i] for i in sampled_idxs]  # base labels

    logger.info(
        "DTW plots on %d signals from %d base-classes: %s",
        len(X_sub),
        len(classes),
        classes,
    )

    if run_rep_separate:
        run_ids_sub = [run_ids[i] for i in sampled_idxs]
        labels_sub_plot, plot_class_order, plot_colors = (
            create_replicate_labels_and_colors(labels_sub_original, run_ids_sub)
        )

        # optional exclusion (keeps semantics from your original code)
        exclude_label = "ßCATD-rep2"
        if exclude_label in labels_sub_plot:
            keep_mask = [lbl != exclude_label for lbl in labels_sub_plot]
            X_sub = [x for x, keep in zip(X_sub, keep_mask) if keep]
            labels_sub_original = [
                lbl for lbl, keep in zip(labels_sub_original, keep_mask) if keep
            ]
            run_ids_sub = [rid for rid, keep in zip(run_ids_sub, keep_mask) if keep]
            labels_sub_plot, plot_class_order, plot_colors = (
                create_replicate_labels_and_colors(labels_sub_original, run_ids_sub)
            )
            logger.info(
                f"Excluded label '{exclude_label}' - now using {len(X_sub)} signals."
            )
    else:
        labels_sub_plot = labels_sub_original
        plot_class_order = sorted(set(labels_sub_plot))
        plot_colors = get_colors_for_classes_dtw(plot_class_order)

    # Build all_signals_dict grouped by the labels used in plots (replicate-aware when requested)
    labels_for_medoid = labels_sub_plot

    labels_arr = np.array(labels_for_medoid, dtype=object)
    classes_for_dtw = sorted(np.unique(labels_arr))
    sampled_idxs_dtw_relative = sample_per_class(
        labels_arr, classes_for_dtw, per_class=dtw_limit, seed=seed
    )
    # indices relative to X_sub / labels_for_medoid
    X_DTW_limit = [X_sub[i] for i in sampled_idxs_dtw_relative]
    DTW_labels_plot = [labels_for_medoid[i] for i in sampled_idxs_dtw_relative]

    # For convenience, DTW dict and colors
    DTW_plot_class_order = sorted(np.unique(DTW_labels_plot))
    if run_rep_separate:
        # We already have replicate-aware labels, so colors must align with DTW_plot_class_order
        # Use get mapping from previously created plot_class_order/colors if possible
        if (
            plot_class_order
            and plot_colors
            and set(DTW_plot_class_order).issubset(set(plot_class_order))
        ):
            color_map_full = dict(zip(plot_class_order, plot_colors))
            DTW_plot_colors = [color_map_full[c] for c in DTW_plot_class_order]
        else:
            DTW_plot_colors = get_colors_for_classes_dtw(DTW_plot_class_order)
    else:
        DTW_plot_colors = get_colors_for_classes_dtw(DTW_plot_class_order)

    templates, template_names, medoid_index_map = compute_class_medoids(
        X_DTW_limit,
        DTW_labels_plot,
        window_frac=0.10,
        n_jobs=n_jobs,
        norm="path",
    )

    all_signals_dict = {cls: [] for cls in plot_class_order}
    for sig, lbl in zip(X_DTW_limit, DTW_labels_plot):
        all_signals_dict.setdefault(lbl, []).append(sig)

    medoids_dict = {}
    for name, sig in zip(template_names, templates):
        # name is like "medoid_ßCAT-rep1_0" → split by '_' and take the class part
        parts = name.split("_")
        if len(parts) >= 3:
            cls = "_".join(parts[1:-1])  # handles 'ßCAT-rep1' (has hyphen)
        else:
            cls = parts[1]
        medoids_dict[cls] = sig

    medoids_dict, std_dict, n_dict = compute_std_per_class(
        X_DTW_limit,
        DTW_labels_plot,
        medoid_index_map,
        resamp_len=300,
        method="dtw_align",
        band=(25, 75),
        outer_percent=None,
        n_examples=0,
    )

    plot_peptide_medoids_panel(
        medoids_dict,
        std_dict,
        n_dict,
        all_signals_dict=all_signals_dict,
        class_order=plot_class_order,
        resamp_len=resamp_len,
        outdir=outdir,
        colors=plot_colors,
        dtw_limit=dtw_limit,
    )

    # Export Figure 2A data for reproducibility
    try:
        from figure_bundle_io import save_pickle_bundle
        medoids_bundle = {
            "medoids_dict": medoids_dict,
            "std_dict": std_dict,
            "n_dict": n_dict,
            "all_signals_dict": all_signals_dict,
            "class_order": plot_class_order,
            "resamp_len": resamp_len,
            "colors": plot_colors,
            "dtw_limit": dtw_limit,
        }
        save_pickle_bundle(
            os.path.join(outdir, "fig2_medoids_bundle.pkl"), medoids_bundle
        )
        logger.info(f"Exported Figure 2A bundle to {outdir}/fig2_medoids_bundle.pkl")
    except Exception as e:
        logger.warning(f"Failed to export Figure 2A bundle: {e}")

    plot_signal_distributions_supplementary(
        all_signals_dict,
        class_order=plot_class_order,
        colors=plot_colors,
        outdir=outdir,
    )

    plot_median_distributions_supplementary(
        all_signals_dict,
        class_order=plot_class_order,
        colors=plot_colors,
        outdir=outdir,
    )

    plot_std_distributions_supplementary(
        all_signals_dict,
        class_order=plot_class_order,
        colors=plot_colors,
        outdir=outdir,
    )

    plot_median_vs_std_contour_supplementary(
        all_signals_dict,
        class_order=plot_class_order,
        colors=plot_colors,
        outdir=outdir,
    )

    features_df, _ = extract_interpretable_features(
        X_DTW_limit,
        use_catch22=True,
    )

    top_features, _ = select_top_nonredundant_features(
        features_df, DTW_labels_plot, n_features=5, corr_threshold=0.95
    )
    if top_features:
        logging.info(f"Top 5 features: {', '.join(top_features)}")

        plot_feature_distributions_panel(
            features_df,
            DTW_labels_plot,  # labels_sub_plot,
            top_features,
            class_order=plot_class_order,
            colors=plot_colors,
            outdir=outdir,
        )

        plot_feature_distributions_panel(
            features_df,
            DTW_labels_plot,  # labels_sub_plot,
            top_features,
            class_order=plot_class_order,
            colors=plot_colors,
            outdir=outdir,
            stat_test=True,
        )

        # Export Figure 2B data for reproducibility
        try:
            from figure_bundle_io import save_pickle_bundle
            features_bundle = {
                "features_df": features_df,
                "labels": DTW_labels_plot,
                "selected_features": top_features,
                "class_order": plot_class_order,
                "colors": plot_colors,
            }
            save_pickle_bundle(
                os.path.join(outdir, "fig2_features_bundle.pkl"), features_bundle
            )
            logger.info(f"Exported Figure 2B bundle to {outdir}/fig2_features_bundle.pkl")
        except Exception as e:
            logger.warning(f"Failed to export Figure 2B bundle: {e}")
    else:
        logger.warning(
            "No valid features found for Figure 2B; skipping feature distribution plots."
        )

    try:
        top_features, _ = select_top_nonredundant_features(
            features_df, DTW_labels_plot, n_features=40, corr_threshold=0.95
        )
    except ValueError as e:
        if "negative distances" in str(e):
            logger.info("Falling back to greedy feature selection...")
            top_features, _ = select_top_nonredundant_features_greedy(
                features_df, DTW_labels_plot, n_features=40, corr_threshold=0.95
            )
        else:
            raise
    if top_features:
        logging.info(f"Top features for embeddings: {', '.join(top_features)}")

        plot_embeddings_panel(
            features_df[top_features],
            DTW_labels_plot,
            class_order=plot_class_order,
            colors=plot_colors,
            outdir=outdir,
            seed=seed,
        )

        # Export Figure 2C data for reproducibility
        try:
            from figure_bundle_io import save_pickle_bundle
            embeddings_bundle = {
                "features_df": features_df[top_features],
                "labels": DTW_labels_plot,
                "class_order": plot_class_order,
                "colors": plot_colors,
                "seed": seed,
            }
            save_pickle_bundle(
                os.path.join(outdir, "fig2_embeddings_bundle.pkl"), embeddings_bundle
            )
            logger.info(f"Exported Figure 2C bundle to {outdir}/fig2_embeddings_bundle.pkl")
        except Exception as e:
            logger.warning(f"Failed to export Figure 2C bundle: {e}")
    else:
        logger.warning(
            "No valid features found for Figure 2C; skipping embeddings plot."
        )

    if dtw_features:
        sampled_idxs_dtw = sample_per_class(
            y, selected_classes, per_class=500, seed=seed
        )
        X_dtw = [X[i] for i in sampled_idxs_dtw]
        labels_dtw_orig = [y[i] for i in sampled_idxs_dtw]
        if run_rep_separate:
            run_ids_dtw = [run_ids[i] for i in sampled_idxs_dtw]
            labels_dtw_plot, dtw_class_order, dtw_colors = (
                create_replicate_labels_and_colors(
                    labels_dtw_orig,
                    run_ids_dtw,
                    base_class_order=selected_classes,
                )
            )
        else:
            labels_dtw_plot = labels_dtw_orig
            dtw_class_order = sorted(set(labels_dtw_plot))
            dtw_colors = get_colors_for_classes_dtw(dtw_class_order)

        D_dtw = get_pairwise_DTW(X_dtw, window_frac=0.2, n_jobs=n_jobs, norm="path")
        plot_within_between_boxplots(D_dtw, labels_dtw_plot, outdir=outdir)

        plot_dtw_embeddings_panel(
            D_dtw,
            labels_dtw_plot,
            class_order=dtw_class_order,
            colors=dtw_colors,
            outdir=outdir,
            seed=seed,
            embedding_type="dtw",
        )

        # Combined embedding: feature selection still on original labels
        features_dtw, _ = extract_interpretable_features(X_dtw, use_catch22=True)
        try:
            top30_dtw, _ = select_top_nonredundant_features(
                features_dtw, labels_dtw_orig, n_features=30, corr_threshold=0.95
            )
        except ValueError as e:
            if "negative distances" in str(e):
                top30_dtw, _ = select_top_nonredundant_features_greedy(
                    features_dtw, labels_dtw_orig, n_features=30, corr_threshold=0.95
                )
            else:
                raise

        plot_dtw_embeddings_panel(
            D_dtw,
            labels_dtw_plot,
            class_order=dtw_class_order,
            colors=dtw_colors,
            features_df=features_dtw[top30_dtw],
            outdir=outdir,
            seed=seed,
            embedding_type="combined",
        )


def create_replicate_labels_and_colors(original_labels, run_ids, base_class_order=None):
    """
    Assign fixed base colors (ßCAT=colorblind[0], ßCATD=colorblind[1], etc.),
    then create shaded versions for replicates.
    """
    if len(original_labels) != len(run_ids):
        raise ValueError("original_labels and run_ids must have same length")

    from collections import defaultdict

    base_to_run_set = defaultdict(set)
    for lbl, rid in zip(original_labels, run_ids):
        base_to_run_set[lbl].add(rid)

    base_to_run_list = {}
    for base, runs in base_to_run_set.items():
        try:
            sorted_runs = sorted(runs, key=lambda x: int(x) if x.isdigit() else x)
        except:
            sorted_runs = sorted(runs)
        base_to_run_list[base] = sorted_runs

    run_to_rep = {}
    for base, runs in base_to_run_list.items():
        for idx, rid in enumerate(runs, start=1):
            run_to_rep[(base, rid)] = idx

    new_labels = [
        f"{lbl}-rep{run_to_rep[(lbl, rid)]}"
        for lbl, rid in zip(original_labels, run_ids)
    ]
    unique_new = sorted(set(new_labels))

    colorblind10 = sns.color_palette("colorblind", 10)

    fixed_assignments = {
        "ßCAT": 0,
        "ßCATD": 1,
        "ßCATL": 2,
        "ßCATW": 3,
        "ßCATWW": 4,
        "ßCATGG": 5,
        "ßCATWWW": 6,
        "BCAR3": 7,
        "ßCAT20": 8,
        "ßCAT30": 9,
    }

    # Get all unique base classes from original_labels
    base_classes_all = set(original_labels)

    # Build base_color_map using FIXED indices
    base_color_map = {}
    for base in base_classes_all:
        if base in fixed_assignments:
            base_color_map[base] = colorblind10[fixed_assignments[base]]
        else:
            # Fallback: use next available colorblind color or husl
            used_indices = set(fixed_assignments.values())
            available = [i for i in range(10) if i not in used_indices]
            if available:
                idx = available[0]
                base_color_map[base] = colorblind10[idx]
                fixed_assignments[base] = idx  # avoid reuse
            else:
                # Use husl as last resort
                extra = sns.husl_palette(1, s=0.9, l=0.65)[0]
                base_color_map[base] = extra

    base_to_reps = defaultdict(list)
    for lbl in unique_new:
        base = lbl.split("-rep")[0]
        base_to_reps[base].append(lbl)

    for base in base_to_reps:
        base_to_reps[base] = sorted(base_to_reps[base])  # rep1, rep2, ...

    color_map = {}
    for base, reps in base_to_reps.items():
        base_rgb = base_color_map[base]
        n = len(reps)
        if n == 1:
            shades = [base_rgb]
        else:
            h, l, s = colorsys.rgb_to_hls(*base_rgb)
            # Make rep1 = original color, rep2 = lighter, rep3 = darker, etc.
            # Or: linear spread around original lightness
            delta = 0.15
            lightnesses = [
                l + delta * (i - (n - 1) / 2) / max(1, n - 1) for i in range(n)
            ]
            # Clamp to [0.2, 0.95]
            lightnesses = [min(0.95, max(0.2, L)) for L in lightnesses]
            shades = [colorsys.hls_to_rgb(h, L, s) for L in lightnesses]
        for rep, shade in zip(reps, shades):
            color_map[rep] = shade

    plot_class_order = sorted(unique_new)

    # Colors aligned to plot_class_order (so plotting functions can use colors[i] <-> class_order[i])
    colors_for_classes = [color_map[cls] for cls in plot_class_order]

    # Return: per-sample labels, class order, and colors aligned to class order
    return new_labels, plot_class_order, colors_for_classes


def get_colors_for_classes_dtw(class_labels):
    """Generate distinct colors for each class label."""
    colorblind10 = sns.color_palette("colorblind", 10)
    colorblind_hex = [mcolors.to_hex(c) for c in colorblind10]
    fixed_assignments = {
        "ßCAT": 0,
        "ßCATD": 1,
        "ßCATL": 2,
        "ßCATW": 3,
        "ßCATWW": 4,
        "ßCATGG": 5,
        "ßCATWWW": 6,
        "BCAR3": 7,
        "ßCAT20": 8,
        "ßCAT30": 9,
    }

    fixed_colors = {cls: colorblind_hex[idx] for cls, idx in fixed_assignments.items()}

    remaining_classes = [c for c in class_labels if c not in fixed_colors]
    unique_remaining = sorted(
        set(remaining_classes), key=lambda x: hashlib.md5(x.encode()).hexdigest()
    )
    n_extra = len(unique_remaining)
    extra_map = {}

    if n_extra > 0:
        if len(fixed_colors) + n_extra <= 10:
            # Use unused colorblind colors
            used_indices = set(fixed_assignments.values())
            available_indices = [i for i in range(10) if i not in used_indices]
            # Take first n_extra available
            extra_colors = [colorblind_hex[i] for i in available_indices[:n_extra]]
        else:
            extra_colors = sns.husl_palette(n_extra, s=0.9, l=0.65)
        extra_map = dict(zip(unique_remaining, extra_colors))
    color_map = {**fixed_colors, **extra_map}
    return [color_map[cls] for cls in class_labels]


def get_colors_for_classes(class_labels):
    """
    Return a dict mapping each class label -> hex color string.
    - Use colorblind palette for fixed classes (if present).
    - Then use tab20 (20 colors) to cover many classes without repeats.
    - If still more required, fall back to husl/hsv evenly spaced hues.
    - Deterministic ordering for extras via MD5 of class name.
    """
    # normalize labels (strings)
    class_labels = [str(x) for x in class_labels]

    # 1) Base colorblind palette (10 colors)
    cb = sns.color_palette("colorblind", 10)
    cb_hex = [mcolors.to_hex(c) for c in cb]

    # 2) Fixed assignments (keep your preferred mapping)
    fixed_assignments = {
        "ßCAT": 0,
        "ßCATD": 1,
        "ßCATL": 2,
        "ßCATW": 3,
        "ßCATWW": 4,
        "ßCATGG": 5,
        "ßCATWWW": 6,
        "BCAR3": 7,
        "ßCAT20": 8,
        "ßCAT30": 9,
    }
    # clamp indices
    fixed_assignments = {k: (v % len(cb_hex)) for k, v in fixed_assignments.items()}

    fixed_colors = {
        k: cb_hex[v] for k, v in fixed_assignments.items() if k in class_labels
    }

    # 3) Remaining classes (deterministic order)
    remaining = [c for c in class_labels if c not in fixed_colors]
    unique_remaining = sorted(
        set(remaining), key=lambda x: hashlib.md5(x.encode()).hexdigest()
    )
    n_extra = len(unique_remaining)

    extra_map = {}
    if n_extra > 0:
        # Try tab20 first (20 distinct colors)
        tab20 = plt.get_cmap("tab20")  # returns RGBA colormap with 20 colors
        tab20_colors = [mcolors.to_hex(tab20(i)) for i in range(tab20.N)]

        # choose colors from tab20 excluding any used by fixed_colors if they happen to collide
        used_hex = set(fixed_colors.values())
        tab20_available = [c for c in tab20_colors if c not in used_hex]

        if len(tab20_available) >= n_extra:
            extra_colors = tab20_available[:n_extra]
        else:
            # Not enough in tab20 (unlikely for 19 classes), so create evenly spaced HUSL/HSL/HSV colors
            # Use seaborn.husl_palette which yields perceptually spaced hues
            husl = sns.husl_palette(n_extra, s=0.85, l=0.55, hue=0)
            extra_colors = [mcolors.to_hex(c) for c in husl]

            # Still ensure no collision with used_hex (very unlikely) by adjusting lightness slightly
            for i, col in enumerate(extra_colors):
                if col in used_hex:
                    # nudge Hue by a small amount
                    hls = mcolors.rgb_to_hsv(mcolors.to_rgb(col))
                    # rotate hue a bit and re-convert
                    new_h = (hls[0] + 0.07 * (i + 1)) % 1.0
                    rgb = mcolors.hsv_to_rgb((new_h, hls[1], hls[2]))
                    extra_colors[i] = mcolors.to_hex(rgb)

        extra_map = dict(zip(unique_remaining, extra_colors))

    # Final mapping; fallback gray if something missing
    color_map = {}
    for cls in class_labels:
        if cls in fixed_colors:
            color_map[cls] = fixed_colors[cls]
        elif cls in extra_map:
            color_map[cls] = extra_map[cls]
        else:
            color_map[cls] = "#808080"  # neutral gray fallback

    return color_map


def dtw_plots(
    X,
    y,
    selected_classes=None,
    per_class=200,
    outdir=None,
    resamp_len=300,
    n_jobs=16,
    seed=385,
):
    # pick classes
    unique_classes = np.unique(y)
    if selected_classes is None:
        classes = unique_classes
    else:
        classes = [c for c in unique_classes if c in set(selected_classes)]
        if not classes:
            raise ValueError("No requested classes found in labels")

    sampled_idxs = sample_per_class(y, classes, per_class=per_class, seed=seed)

    X_sub = [X[i] for i in sampled_idxs]
    labels_sub = [y[i] for i in sampled_idxs]
    logger.info(
        "DTW plots on %d signals from %d classes: %s", len(X_sub), len(classes), classes
    )

    # visualise embedding and distances
    D = get_pairwise_DTW(X_sub, window_frac=0.2, n_jobs=n_jobs, norm="path")

    # compute separability
    sep = compute_distance_separability(D, labels_sub)
    logger.info(f"Global silhouette: {sep['global_silhouette']}")
    logger.info(f"Per-class mean silhouette: {sep['per_class_silhouette']}")

    emb_file = (
        os.path.join(outdir, "dtw_distance_embedding.png")
        if outdir
        else "dtw_distance_embedding.png"
    )
    emb = visualize_distance_embedding(labels_sub, D, outfile=emb_file, seed=seed)

    tsne_file = (
        os.path.join(outdir, "dtw_distance_tsne.png")
        if outdir
        else "dtw_distance_tsne.png"
    )
    emb_tsne = visualize_tsne_distance(
        labels_sub, D, perplexity=30, n_iter=1000, outfile=tsne_file, seed=seed
    )

    outpref = os.path.join(outdir, "dtw") if outdir else "dtw"
    plot_files = plot_silhouette_results(
        sep, outfile_prefix=outpref, emb=emb, labels=labels_sub
    )

    # boxplots
    box_file = os.path.join(outdir, "dtw") if outdir else "dtw"
    df_boxes = plot_within_between_boxplots(D, labels_sub, outfile_prefix=box_file)

    tree_file = (
        os.path.join(outdir, "dtw_hierarchical_tree")
        if outdir
        else "dtw_hierarchical_tree"
    )
    plot_hierarchical_signals(
        D, X_sub, labels_sub, outfile=tree_file, max_per_class=5, seed=seed
    )

    features_df, feat_names = extract_interpretable_features(X_sub, use_catch22=True)

    #  Boxplots
    box_plot_file = (
        os.path.join(outdir, "feature_boxplots.png")
        if outdir
        else "feature_boxplots.png"
    )
    plot_feature_boxplots(
        features_df, labels_sub, ["min", "median_slope"], outfile=box_plot_file
    )

    # UMAP on features only
    feat_umap_file = (
        os.path.join(outdir, "features_umap.png") if outdir else "features_umap.png"
    )
    emb_feat = umap_on_features(
        features_df, labels_sub, n_neighbors=200, min_dist=0.1, outfile=feat_umap_file
    )

    # UMAP on combined distance (DTW + features)
    combined_umap_file = (
        os.path.join(outdir, "dtw_combined_umap_alpha0.5.png")
        if outdir
        else "dtw_combined_umap_alpha0.5.png"
    )
    emb_comb = umap_on_combined_distance(
        features_df,
        D,
        labels_sub,
        alpha=0.5,
        n_neighbors=200,
        min_dist=0.1,
        outfile=combined_umap_file,
    )

    return emb, D, labels_sub
