from matplotlib.colors import LinearSegmentedColormap, Normalize
import os
import hashlib
import logging
import random
import warnings
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

try:
    import tensorflow as tf
except ImportError:
    tf = None


from utils_classification import extract_interpretable_features

logger = logging.getLogger("nanopore-peptide-classifier")
logger.addHandler(logging.NullHandler())

COLOR_CORRECT = "#1F449C"  # blue
COLOR_INCORRECT = "#F05039"  # red-orange

class ColorMapping(dict):
    def __init__(self, ordered_labels, mapping):
        super().__init__(mapping)
        self._ordered = list(ordered_labels)

    def __iter__(self):
        for lbl in self._ordered:
            yield self[lbl]

    def __getitem__(self, key):
        if isinstance(key, int):
            lbl = self._ordered[key]
            return super().__getitem__(lbl)
        return super().__getitem__(key)

    def as_dict(self):
        return dict(self)

    def ordered_labels(self):
        return list(self._ordered)

    def ordered_colors(self):
        return [self[lbl] for lbl in self._ordered]


def get_colors_for_classes(class_labels):
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

    ordered_unique = []
    for c in class_labels:
        if c not in ordered_unique:
            ordered_unique.append(c)

    fixed_colors = {}
    for cls, idx in fixed_assignments.items():
        if cls in ordered_unique and 0 <= idx < len(colorblind_hex):
            fixed_colors[cls] = colorblind_hex[idx]

    remaining = [c for c in ordered_unique if c not in fixed_colors]
    unique_remaining = sorted(
        dict.fromkeys(remaining), key=lambda x: hashlib.md5(x.encode()).hexdigest()
    )

    n_extra = len(unique_remaining)
    extra_map = {}
    if n_extra > 0:
        used_indices = set(v for k, v in fixed_assignments.items() if k in fixed_colors)
        available_indices = [
            i for i in range(len(colorblind_hex)) if i not in used_indices
        ]

        if (
            len(fixed_colors) + n_extra <= len(colorblind_hex)
            and len(available_indices) >= n_extra
        ):
            extra_colors = [colorblind_hex[i] for i in available_indices[:n_extra]]
        else:
            extra_rgb = sns.husl_palette(n_extra, s=0.9, l=0.65)
            extra_colors = [mcolors.to_hex(c) for c in extra_rgb]

        extra_map = dict(zip(unique_remaining, extra_colors))

    color_map_dict = {**fixed_colors, **extra_map}

    default_cycle = [
        mcolors.to_hex(c) for c in mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    ]
    dc_len = len(default_cycle)
    idx_def = 0
    for lbl in ordered_unique:
        if lbl not in color_map_dict:
            color_map_dict[lbl] = default_cycle[idx_def % dc_len]
            idx_def += 1

    return ColorMapping(ordered_unique, color_map_dict)



def sliding_occlusion_importance(
    signal,
    model,
    feature_kwargs,
    true_class=None,
    window_size=15,
    step=2,
    baseline_value=None,
    resample_len=None,
):
    sig = np.asarray(signal, dtype=float).copy()

    if resample_len is not None and len(sig) != resample_len and len(sig) > 1:
        xs_old = np.linspace(0, 1, len(sig))
        xs_new = np.linspace(0, 1, resample_len)
        sig = np.interp(xs_new, xs_old, sig)

    if baseline_value is None:
        baseline_value = float(np.nanmedian(sig))

    try:
        feats_base_df, _ = extract_interpretable_features([sig], **feature_kwargs)
        if feats_base_df.empty:
            return np.array([]), np.array([]), np.nan
    except Exception as e:
        logger.warning("Feature extraction failed for baseline: %s", e)
        return np.array([]), np.array([]), np.nan

    try:
        proba_base = model.predict_proba(feats_base_df.values)
        if proba_base.ndim == 1:
            proba_base = np.vstack([1 - proba_base, proba_base]).T

        if true_class is None:
            true_class = int(np.argmax(proba_base, axis=1)[0])
        elif isinstance(true_class, str) and hasattr(model, "classes_"):
            try:
                true_class = int(np.where(model.classes_ == true_class)[0][0])
            except:
                true_class = int(np.argmax(proba_base, axis=1)[0])

        baseline_prob = float(proba_base[0, true_class])
    except Exception as e:
        logger.warning("Prediction failed for baseline: %s", e)
        return np.array([]), np.array([]), np.nan

    L = len(sig)
    positions, imps = [], []
    valid_positions = []

    for start in range(0, L, step):
        end = min(L, start + window_size)
        s = sig.copy()
        s[start:end] = baseline_value

        try:
            feats_df, _ = extract_interpretable_features([s], **feature_kwargs)
            if feats_df.empty:
                importance = np.nan
            else:
                # Get prediction for occluded signal
                proba = model.predict_proba(feats_df.values)
                if proba.ndim == 1:
                    proba = np.vstack([1 - proba, proba]).T

                # Calculate importance as change in probability
                prob = float(proba[0, true_class])
                importance = baseline_prob - prob
        except Exception as e:
            logger.debug("Occlusion step failed at %d-%d: %s", start, end, e)
            importance = np.nan

        center = (start + end) / 2.0
        positions.append(center)
        imps.append(importance)
        if not np.isnan(importance):
            valid_positions.append(True)
        else:
            valid_positions.append(False)

    positions = np.array(positions)
    imps = np.array(imps)

    # Handle NaN values using interpolation where possible
    if np.any(np.isnan(imps)):
        valid_mask = ~np.isnan(imps)
        if np.sum(valid_mask) > 1:
            try:
                imps_clean = np.interp(
                    positions, positions[valid_mask], imps[valid_mask]
                )
                imps = imps_clean
            except Exception as e:
                logger.warning("Interpolation failed: %s", e)

    return positions, imps, baseline_prob




def plot_colored_violin_shapstyle(
    ax,
    shap_arr,
    features_df,
    idxs,
    topk,
    cols=None,
    cmap_name="RdBu_r",
    font_size=6,
    show_yticklabels=True,
    violin_width=0.4,
    n_kde_points=100,
    kde_bandwidth=None,
    force_xlim=None,
):
    warnings.filterwarnings(
        "ignore", category=UserWarning
    )

    if cols is None:
        cols = list(features_df.columns)

    idxs = np.asarray(idxs, dtype=int)
    if idxs.size == 0:
        ax.text(0.5, 0.5, "no samples", ha="center")
        ax.set_axis_off()
        return None

    n_feats = len(topk)
    feature_inds = []
    for f in topk:
        if f not in cols:
            raise ValueError(f"feature '{f}' not found in cols/features_df")
        feature_inds.append(cols.index(f))

    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=0.0, vmax=1.0)

    for pos, fname in enumerate(topk):
        fi = feature_inds[pos]
        shap_vals = np.asarray(shap_arr[idxs, fi], dtype=float)
        feat_vals = np.asarray(features_df.iloc[idxs][fname].values, dtype=float)

        finite_shap = np.isfinite(shap_vals)
        shap_vals = shap_vals[finite_shap]
        feat_vals = feat_vals[finite_shap]
        if shap_vals.size == 0:
            continue

        norm_feat_vals = _normalize_feature_values_for_coloring(feat_vals)

        x_min, x_max = float(np.min(shap_vals)), float(np.max(shap_vals))
        x_range = x_max - x_min
        if shap_vals.size < 2 or x_range <= 1e-12:
            if shap_vals.size == 1:
                jitter = np.array([0.0], dtype=float)
            else:
                jitter = np.linspace(-0.15, 0.15, shap_vals.size)
            colors = cmap(norm(norm_feat_vals))
            ax.scatter(
                shap_vals,
                np.full(shap_vals.size, pos, dtype=float) + jitter,
                c=colors,
                s=10,
                alpha=0.8,
                linewidths=0,
            )
            continue

        # Compute KDE for SHAP values
        try:
            kde = gaussian_kde(shap_vals, bw_method=kde_bandwidth)
        except (ValueError, np.linalg.LinAlgError):
            jitter = np.linspace(-0.15, 0.15, shap_vals.size)
            colors = cmap(norm(norm_feat_vals))
            ax.scatter(
                shap_vals,
                np.full(shap_vals.size, pos, dtype=float) + jitter,
                c=colors,
                s=10,
                alpha=0.8,
                linewidths=0,
            )
            continue

        # Range to evaluate KDE over
        x_vals = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, n_kde_points)
        kde_vals = kde(x_vals)
        if not np.all(np.isfinite(kde_vals)) or np.max(kde_vals) <= 0:
            jitter = np.linspace(-0.15, 0.15, shap_vals.size)
            colors = cmap(norm(norm_feat_vals))
            ax.scatter(
                shap_vals,
                np.full(shap_vals.size, pos, dtype=float) + jitter,
                c=colors,
                s=10,
                alpha=0.8,
                linewidths=0,
            )
            continue

        kde_vals = kde_vals / kde_vals.max() * violin_width

        bw = kde.scotts_factor() * shap_vals.std(ddof=1)
        if not np.isfinite(bw) or bw <= 0:
            bw = 0.1 * (x_max - x_min) if (x_max - x_min) > 0 else 1.0

        colors_along = []
        for x0 in x_vals:
            weights = np.exp(-0.5 * ((shap_vals - x0) / bw) ** 2)
            if weights.sum() > 0:
                avg_color = np.sum(norm_feat_vals * weights) / weights.sum()
            else:
                avg_color = 0.5  # fallback gray
            colors_along.append(avg_color)
        colors_along = np.array(colors_along)

        y = pos

        verts_left = np.array([x_vals, y - kde_vals]).T
        verts_right = np.array([x_vals[::-1], (y + kde_vals)[::-1]]).T
        verts = np.vstack([verts_left, verts_right])

        for i in range(len(x_vals) - 1):
            # Polygon coords
            xs = [x_vals[i], x_vals[i + 1], x_vals[i + 1], x_vals[i]]
            ys = [
                y - kde_vals[i],
                y - kde_vals[i + 1],
                y + kde_vals[i + 1],
                y + kde_vals[i],
            ]

            poly_verts = list(zip(xs, ys))
            color_val = (colors_along[i] + colors_along[i + 1]) / 2
            color = cmap(norm(color_val))
            polygon = plt.Polygon(
                poly_verts, facecolor=color, edgecolor="none", linewidth=0
            )
            ax.add_patch(polygon)

    ax.set_ylim(-0.6, n_feats - 1 + 0.6)
    ax.set_yticks(np.arange(n_feats))
    if show_yticklabels:
        ax.set_yticklabels([shorten_feature_name(f) for f in topk], fontsize=font_size)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=font_size)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="x", labelsize=font_size - 1)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if force_xlim is not None:
        ax.set_xlim(force_xlim)
    else:
        all_shap_vals = []
        for f in topk:
            fi = cols.index(f)
            all_shap_vals.append(shap_arr[idxs, fi])
        all_shap_vals = np.asarray(np.concatenate(all_shap_vals), dtype=float)
        all_shap_vals = all_shap_vals[np.isfinite(all_shap_vals)]
        if all_shap_vals.size:
            x_min, x_max = np.min(all_shap_vals), np.max(all_shap_vals)
            x_range = x_max - x_min
            margin = 0.05 * x_range if x_range > 0 else 1.0
            ax.set_xlim(x_min - margin, x_max + margin)

    return sm


def _normalize_feature_values_for_coloring(values):
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, 0.5, dtype=float)
    finite_mask = np.isfinite(vals)
    n_finite = int(np.sum(finite_mask))
    if n_finite <= 1:
        return out

    v = vals[finite_mask]
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty_like(v, dtype=float)
    if n_finite == 2:
        ranks[order] = np.array([0.0, 1.0], dtype=float)
    else:
        ranks[order] = np.linspace(0.0, 1.0, n_finite)
    out[finite_mask] = ranks
    return out


def get_shap_for_class(shap_vals, class_idx, n_total_features, classes=None):
    if isinstance(shap_vals, (list, tuple)):
        arrs = [np.asarray(a) for a in shap_vals]
        if len(arrs) == 0:
            raise RuntimeError("Received empty SHAP list.")
        idx = int(class_idx) if class_idx is not None else 0
        idx = max(0, min(idx, len(arrs) - 1))
        out = np.asarray(arrs[idx])
        if out.ndim != 2:
            raise RuntimeError(f"Expected class SHAP to be 2D, got shape {out.shape}")
        return out

    arr = np.asarray(shap_vals)
    if arr.ndim == 2:
        # (n_samples, n_features) or transposed
        if arr.shape[1] == n_total_features:
            return arr
        if arr.shape[0] == n_total_features:
            return arr.T
        raise RuntimeError(
            f"2D SHAP shape {arr.shape} does not match n_total_features={n_total_features}"
        )

    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP array ndim={arr.ndim}, shape={arr.shape}")

    c_len = len(classes) if classes is not None else None
    idx = int(class_idx) if class_idx is not None else 0

    if arr.shape[1] == n_total_features:
        idx_clamped = max(0, min(idx, arr.shape[2] - 1))
        return arr[:, :, idx_clamped]
    if c_len is not None and arr.shape[0] == c_len and arr.shape[2] == n_total_features:
        idx_clamped = max(0, min(idx, arr.shape[0] - 1))
        return arr[idx_clamped, :, :]
    if arr.shape[2] == n_total_features:
        idx_clamped = max(0, min(idx, arr.shape[1] - 1))
        return arr[:, idx_clamped, :]

    shape = arr.shape
    feat_axes = [i for i, s in enumerate(shape) if s == n_total_features]
    cls_axes = [i for i, s in enumerate(shape) if c_len is not None and s == c_len]
    if not feat_axes:
        raise RuntimeError(
            f"Cannot infer feature axis in SHAP shape {shape} for n_total_features={n_total_features}"
        )
    feat_ax = feat_axes[0]
    if cls_axes:
        cls_ax = cls_axes[0]
    else:
        candidates = [i for i in range(3) if i != feat_ax and idx < shape[i]]
        if not candidates:
            raise RuntimeError(
                f"Cannot infer class axis in SHAP shape {shape} for class_idx={idx}"
            )
        cls_ax = candidates[-1]
    sample_ax = [i for i in range(3) if i not in (feat_ax, cls_ax)][0]

    moved = np.moveaxis(arr, (sample_ax, feat_ax, cls_ax), (0, 1, 2))
    idx_clamped = max(0, min(idx, moved.shape[2] - 1))
    out = moved[:, :, idx_clamped]
    if out.ndim != 2 or out.shape[1] != n_total_features:
        raise RuntimeError(
            f"Failed to coerce SHAP to (n_samples, n_features), got {out.shape} from original {shape}"
        )
    return out


def _resolve_class_index(classes, label):
    """Resolve class index robustly for mixed dtype labels (e.g., numpy scalars/strings)."""
    if classes is None:
        return None
    arr = np.asarray(classes)
    if arr.size == 0:
        return None

    try:
        exact = np.where(arr == label)[0]
        if len(exact) > 0:
            return int(exact[0])
    except Exception:
        pass

    lbl = str(label)
    arr_str = np.asarray([str(x) for x in arr], dtype=object)
    str_eq = np.where(arr_str == lbl)[0]
    if len(str_eq) > 0:
        return int(str_eq[0])

    str_eq_ci = np.where(np.char.lower(arr_str.astype(str)) == lbl.lower())[0]
    if len(str_eq_ci) > 0:
        return int(str_eq_ci[0])
    return None


def _set_interpretability_seed(seed):
    if seed is None:
        return
    s = int(seed)
    np.random.seed(s)
    random.seed(s)
    if tf is not None:
        try:
            tf.random.set_seed(s)
        except Exception:
            pass


def _select_peptides_deterministic(summary, top_n, by_error=True):
    if summary is None or summary.empty:
        return []

    work = summary.reset_index().rename(columns={"index": "peptide"})
    if "peptide" not in work.columns:
        return []

    if by_error:
        sort_cols = ["n_incorrect", "error_rate", "total", "peptide"]
        ascending = [False, False, False, True]
    else:
        sort_cols = ["total", "peptide"]
        ascending = [False, True]

    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return work["peptide"].head(int(top_n)).tolist()


def _safe_run_id_at(run_ids, idx):
    if run_ids is None or idx is None:
        return ""
    try:
        if int(idx) < 0 or int(idx) >= len(run_ids):
            return ""
        val = run_ids[int(idx)]
        if pd.isna(val):
            return ""
        return str(val)
    except Exception:
        return ""


def _write_selected_examples_tsv(output_dir, rows, filename):
    if not rows:
        return
    out_path = os.path.join(output_dir, filename)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    logger.info("Saved interpretability pooled examples TSV: %s", out_path)


def run_interpretability_pipeline(
    model,
    model_name,
    X_train_raw,
    y_train,
    X_val_raw,
    y_val,
    val_metrics,
    output_dir,
    scenario_name,
    specific_peptides=None,
    val_run_ids=None,
    feature_kwargs=None,
    seed=385,
    top_n_features=3,
    top_n_peptides=4,
    occlusion_window=50,
    occlusion_step=20,
    resample_len=300,
    n_jobs=1,
):
    os.makedirs(output_dir, exist_ok=True)
    _set_interpretability_seed(seed)
    if feature_kwargs is None:
        feature_kwargs = {"use_catch22": True}
    
    estimator = getattr(model, "model", model) if hasattr(model, "model") else model
    scaler = getattr(model, "scaler", None)

    y_val_arr = np.asarray(y_val)
    run_id_arr = np.asarray(val_run_ids) if val_run_ids is not None else None
    y_pred = val_metrics.get("y_pred")
    y_pred_arr = np.asarray(y_pred)

    used_wrapper_features = False
    kept_idx = None
    if hasattr(model, "_get_cached_or_extract_features") and hasattr(
        model, "feature_names"
    ):
        try:
            Xf_interp, kept_idx = model._get_cached_or_extract_features(X_val_raw)
            features_df = pd.DataFrame(
                np.asarray(Xf_interp, dtype=float), columns=list(model.feature_names)
            )
            used_wrapper_features = True
            logger.info(
                "[%s] Interpretability features extracted via model wrapper path (n=%d).",
                scenario_name,
                len(features_df),
            )
        except Exception as e:
            logger.warning(
                "[%s] Wrapper feature extraction failed, falling back to direct extraction: %s",
                scenario_name,
                str(e),
            )
            features_df, _ = extract_interpretable_features(X_val_raw, **feature_kwargs)
    else:
        features_df, _ = extract_interpretable_features(X_val_raw, **feature_kwargs)

    if used_wrapper_features and kept_idx is not None:
        kept_idx = np.asarray(kept_idx, dtype=int)
        if kept_idx.size > 0 and (np.max(kept_idx) < len(y_val_arr)):
            y_val_arr = y_val_arr[kept_idx]
            y_pred_arr = y_pred_arr[kept_idx]
            if run_id_arr is not None:
                run_id_arr = run_id_arr[kept_idx]

    if len(features_df) != len(y_val_arr):
        n = min(len(features_df), len(y_val_arr))
        features_df = features_df.iloc[:n].reset_index(drop=True)
        y_val_arr = y_val_arr[:n]
        y_pred_arr = y_pred_arr[:n]
        if run_id_arr is not None:
            run_id_arr = run_id_arr[:n]

    df_feat = features_df.copy()
    df_feat["true_label"] = y_val_arr
    df_feat["pred_label"] = y_pred_arr
    df_feat["correct"] = df_feat["true_label"] == df_feat["pred_label"]
    df_feat.to_csv(
        os.path.join(output_dir, "diagnostic_val_features_and_preds.csv"), index=False
    )

    slope_cols = [
        c
        for c in features_df.columns
        if "slope" in str(c).lower() or "slop" in str(c).lower()
    ]
    if slope_cols:
        slope_rows = []
        for pep in sorted(pd.unique(df_feat["true_label"])):
            sub = df_feat[df_feat["true_label"] == pep]
            if sub.empty:
                continue
            for col in slope_cols:
                vals = (
                    pd.to_numeric(sub[col], errors="coerce")
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if vals.size == 0:
                    continue
                slope_rows.append(
                    {
                        "peptide": str(pep),
                        "feature": str(col),
                        "n": int(vals.size),
                        "q50": float(np.quantile(vals, 0.50)),
                        "q90": float(np.quantile(vals, 0.90)),
                        "q95": float(np.quantile(vals, 0.95)),
                        "q99": float(np.quantile(vals, 0.99)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                    }
                )
        if slope_rows:
            slope_diag_path = os.path.join(
                output_dir, "diagnostic_slope_features_by_class.tsv"
            )
            pd.DataFrame(slope_rows).to_csv(slope_diag_path, sep="\t", index=False)
            logger.info(
                "[%s] Saved slope-feature diagnostics: %s",
                scenario_name,
                slope_diag_path,
            )

    if specific_peptides is not None and len(specific_peptides) > 0:
        # Use user-specified peptides
        logger.info(
            "[%s] Using user-specified peptides: %s", scenario_name, specific_peptides
        )

        available_peptides = sorted(list(pd.unique(df_feat["true_label"])))
        selected_peptides = [
            pep for pep in specific_peptides if pep in available_peptides
        ]

        missing_peptides = [
            pep for pep in specific_peptides if pep not in available_peptides
        ]
        if missing_peptides:
            logger.warning(
                "[%s] The following specified peptides were not found in validation data: %s",
                scenario_name,
                missing_peptides,
            )

        if len(selected_peptides) == 0:
            logger.warning(
                "[%s] No specified peptides found in data. Falling back to automatic selection.",
                scenario_name,
            )
            specific_peptides = None

    if specific_peptides is None or len(selected_peptides) == 0:
        # Automatic selection of top misclassified peptides
        per_class = pd.DataFrame(
            {"peptide": df_feat["true_label"], "correct": df_feat["correct"]}
        )
        summary = per_class.groupby("peptide").agg(
            total=("correct", "size"), n_correct=("correct", "sum")
        )
        summary["n_incorrect"] = summary["total"] - summary["n_correct"]
        summary["error_rate"] = summary["n_incorrect"] / summary["total"].replace(
            0, np.nan
        )
        selected_peptides = _select_peptides_deterministic(
            summary, top_n_peptides, by_error=True
        )

        if len(selected_peptides) == 0 or summary["n_incorrect"].sum() == 0:
            selected_peptides = _select_peptides_deterministic(
                summary, top_n_peptides, by_error=False
            )

        logger.info(
            "[%s] Automatically selected peptides (most misclassified): %s",
            scenario_name,
            selected_peptides,
        )

    # Limit to maximum 4 peptides for the figure layout
    selected_peptides = selected_peptides[:4]
    n_peptides = len(selected_peptides)
    logger.info(
        "[%s] Final peptides for Figure 4: %s", scenario_name, selected_peptides
    )

    cmap = get_colors_for_classes(sorted(list(pd.unique(df_feat["true_label"]))))
    color_map = cmap.as_dict()

    # occlusion examples
    occlusion_examples = []
    for pep in selected_peptides:
        mask = df_feat["true_label"] == pep
        idxs = np.where(mask)[0].tolist()
        chosen = []
        for corr in (False, True):
            sub = [i for i in idxs if df_feat.loc[i, "correct"] == corr]
            if sub:
                chosen.append(sub[0])
            if len(chosen) >= 3:
                break
        if not chosen and idxs:
            chosen = idxs[:3]
        occlusion_examples.extend(chosen[:3])
    occlusion_examples = occlusion_examples[:3]

    occlusion_results = {}
    for i in occlusion_examples:
        sig = X_val_raw[i]
        true_label = str(df_feat.loc[i, "true_label"])
        pred_label = df_feat.loc[i, "pred_label"]
        class_idx = None
        if hasattr(estimator, "classes_"):
            try:
                class_idx = int(np.where(estimator.classes_ == pred_label)[0][0])
            except Exception:
                class_idx = None
        pos, imps, baseprob = sliding_occlusion_importance(
            sig,
            model,
            feature_kwargs,
            true_class=class_idx,
            window_size=occlusion_window,
            step=occlusion_step,
            baseline_value=None,
            resample_len=resample_len,
        )
        occlusion_results[i] = {
            "positions": pos,
            "importance": imps,
            "baseprob": baseprob,
            "true_label": true_label,
            "pred_label": pred_label,
        }

    all_signal_values = []
    for pep in selected_peptides:
        mask = (df_feat["true_label"] == pep) & (~df_feat["correct"])
        idxs = df_feat[mask].index.tolist()
        if idxs:
            idx = idxs[0]
            sig = X_val_raw[idx].copy()
            # Resample to common length
            if len(sig) != resample_len and len(sig) > 1:
                xs_old = np.linspace(0, 1, len(sig))
                xs_new = np.linspace(0, 1, resample_len)
                sig = np.interp(xs_new, xs_old, sig)
            all_signal_values.extend(sig.tolist())

    #  global y-limits
    if all_signal_values:
        global_signal_min = min(all_signal_values)
        global_signal_max = max(all_signal_values)
        padding = (global_signal_max - global_signal_min) * 0.1
        global_signal_min -= padding
        global_signal_max += padding
    else:
        global_signal_min, global_signal_max = 0, 1  # Default fallback

    feature_order = None
    if feature_order is None:
        feature_order = features_df.columns.tolist()
    Xf_for_shap = features_df.copy()
    Xs_for_shap = (
        scaler.transform(Xf_for_shap.values)
        if scaler is not None
        else Xf_for_shap.values
    )
    if model_name in ("XGBoost", "LightGBM", "CatBoost", "featuresLGBM"):
        explainer = shap.TreeExplainer(estimator)
        shap_values_raw = explainer.shap_values(Xs_for_shap)
    else:
        logger.info(f"Unsupported model_name '{model_name}' for SHAP explainability")
        return

    shap_by_class = {}
    shap_summary_by_class = {}
    shap_diag_rows = []

    classes = getattr(estimator, "classes_", None)
    n_total_features = len(feature_order)

    for pep in selected_peptides:
        class_idx = _resolve_class_index(classes, pep)
        if classes is not None and class_idx is None:
            raise RuntimeError(
                f"Could not resolve class index for peptide '{pep}' in estimator classes {list(classes)}"
            )

        # get per-class SHAP 2D array (n_samples, n_features)
        shap_class_all = get_shap_for_class(
            shap_values_raw, class_idx, n_total_features, classes
        )
        shap_class_all = np.nan_to_num(
            np.asarray(shap_class_all, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )

        if shap_class_all.ndim != 2:
            raise RuntimeError(
                f"Could not coerce SHAP values for peptide {pep} to 2D (shape {shap_class_all.shape})"
            )

        if shap_class_all.shape[0] != len(df_feat):
            if shap_class_all.shape == (n_total_features, len(df_feat)):
                shap_class_all = shap_class_all.T
            else:
                raise RuntimeError(
                    f"SHAP sample dimension mismatch for {pep}: got {shap_class_all.shape[0]}, expected {len(df_feat)}"
                )

        if shap_class_all.shape[1] != n_total_features:
            raise RuntimeError(
                f"SHAP feature dimension mismatch for {pep}: got {shap_class_all.shape[1]}, expected {n_total_features}"
            )

        shap_by_class[pep] = shap_class_all

        # Compute mean(|SHAP|) across samples of this peptide (use true_label to select)
        mask = (df_feat["true_label"] == pep).values
        if mask.sum() > 0:
            mean_abs = np.mean(np.abs(shap_class_all[mask, :]), axis=0)
        else:
            # fallback: mean across all samples for that class
            mean_abs = np.mean(np.abs(shap_class_all), axis=0)

        # Create pandas Series in the SAME feature order
        shap_summary_by_class[pep] = pd.Series(
            mean_abs, index=feature_order
        ).sort_values(ascending=False)

        # Export magnitude diagnostics so class-specific anomalies are visible in results
        abs_vals = np.abs(shap_class_all[mask, :] if mask.sum() > 0 else shap_class_all)
        abs_vals = np.asarray(abs_vals, dtype=float).ravel()
        abs_vals = abs_vals[np.isfinite(abs_vals)]
        if abs_vals.size > 0:
            shap_diag_rows.append(
                {
                    "peptide": pep,
                    "n_samples": int(mask.sum()),
                    "shap_abs_q50": float(np.quantile(abs_vals, 0.50)),
                    "shap_abs_q90": float(np.quantile(abs_vals, 0.90)),
                    "shap_abs_q95": float(np.quantile(abs_vals, 0.95)),
                    "shap_abs_q99": float(np.quantile(abs_vals, 0.99)),
                    "shap_abs_max": float(np.max(abs_vals)),
                    "mean_abs_top_feature": (
                        float(shap_summary_by_class[pep].iloc[0])
                        if len(shap_summary_by_class[pep])
                        else np.nan
                    ),
                    "top_feature_name": (
                        str(shap_summary_by_class[pep].index[0])
                        if len(shap_summary_by_class[pep])
                        else ""
                    ),
                }
            )

    if shap_diag_rows:
        shap_diag_path = os.path.join(
            output_dir, "diagnostic_shap_class_magnitudes.tsv"
        )
        pd.DataFrame(shap_diag_rows).to_csv(shap_diag_path, sep="\t", index=False)
        logger.info(
            "[%s] Saved SHAP class diagnostics: %s", scenario_name, shap_diag_path
        )

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
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig_width = 6.7
    fig_height = 9.5 
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        5,
        top_n_peptides,
        height_ratios=[1.0, 1.0, 0.15, 0.7, 0.15],
        hspace=0.45,
        wspace=0.3,
    )
    topk = 10
    top_k = 10

    per_pep_topk = {}
    for pep in selected_peptides:
        summ = shap_summary_by_class.get(pep)
        if summ is None:
            per_pep_topk[pep] = []
        else:
            per_pep_topk[pep] = list(summ.index[:top_k])

    union_topk = []
    for pep in selected_peptides:
        for f in per_pep_topk[pep]:
            if f not in union_topk:
                union_topk.append(f)

    if len(union_topk) == 0:
        combined = pd.concat(
            [shap_summary_by_class[p] for p in shap_summary_by_class.keys()], axis=1
        ).fillna(0)
        union_topk = (
            combined.mean(axis=1).sort_values(ascending=False).index[:top_k].tolist()
        )

    # order the union by mean importance across selected peptides (descending)
    mean_imp = {}
    for f in union_topk:
        vals = [
            float(shap_summary_by_class.get(pep, pd.Series(dtype=float)).get(f, 0.0))
            for pep in selected_peptides
        ]
        mean_imp[f] = float(np.mean(vals))

    global_ordered = sorted(union_topk, key=lambda x: mean_imp[x], reverse=True)

    # cap rows to avoid clutter; ensure we at least show top_k
    max_rows = 10  
    if len(global_ordered) > max_rows:
        final_topk = []
        for f in global_ordered:
            if len(final_topk) >= max_rows:
                break
            final_topk.append(f)
        for pep in selected_peptides:
            for f in per_pep_topk[pep]:
                if f not in final_topk and len(final_topk) < max_rows:
                    final_topk.append(f)
        global_topk_display = final_topk
    else:
        global_topk_display = global_ordered

    global_topk_short = [shorten_feature_name(f) for f in global_topk_display]
    n_rows = len(global_topk_display)

    # PANEL A: barh with aligned rows across all peptide columns
    for col_idx, pep in enumerate(selected_peptides):
        ax = fig.add_subplot(gs[0, col_idx])


        summ = shap_summary_by_class.get(pep)
        if summ is None:
            ax.text(0.5, 0.5, "SHAP missing", ha="center")
            ax.set_axis_off()
            continue

        vals_plot = np.array(
            [float(summ.get(f, 0.0)) for f in global_topk_display], dtype=float
        )

        # horizontal bar plot
        y_pos = np.arange(n_rows)
        ax.barh(y_pos, vals_plot, color="red", align="center")
        # show labels only on left-most column to avoid overlap
        if col_idx == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(global_topk_short, fontsize=font_size)
        else:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([""] * n_rows)

        ax.invert_yaxis()
        xpad = np.max(vals_plot) * 0.02 if np.max(vals_plot) > 0 else 0.001
        for i, v in enumerate(vals_plot):
            ax.text(
                v + xpad, i, f"{v:.3f}", va="center", fontsize=font_size, color="red"
            )

        ax.set_xlabel("mean |SHAP| value", fontsize=font_size)
        ax.set_title(pep, fontsize=font_size + 1)

        if col_idx == 0:
            ax.text(
                -0.25,
                1.15,
                "a",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )

    # PANEL B: Combined beeswarm per peptide
    sm_list = []
    for col_idx, pep in enumerate(selected_peptides):
        ax = fig.add_subplot(gs[1, col_idx])
        if pep not in shap_by_class:
            ax.text(0.5, 0.5, "no SHAP", ha="center")
            ax.set_axis_off()
            continue

        shap_arr = shap_by_class[pep]
        mask = df_feat["true_label"] == pep
        idxs = np.where(mask)[0].tolist()
        if len(idxs) == 0:
            ax.text(0.5, 0.5, "no samples", ha="center")
            ax.set_axis_off()
            continue

        # For beeswarm we use the same rows (global_topk_display)
        # show y tick labels only on leftmost column
        show_yticklabels = col_idx == 0

        sm = plot_colored_violin_shapstyle(
            ax=ax,
            shap_arr=shap_arr,
            features_df=features_df,
            idxs=idxs,
            topk=global_topk_display,
            cols=feature_order,
            cmap_name="coolwarm",
            font_size=font_size,
            show_yticklabels=show_yticklabels,
            violin_width=0.35,
        )

        if sm is not None:
            sm_list.append(sm)

        if col_idx == 0:
            ax.text(
                -0.25,
                1.15,
                "b",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )

    # shared colorbar for panel B 
    sm_for_cbar = sm_list[0] if sm_list else None
    if sm_for_cbar is not None:
        cbar_ax = fig.add_subplot(gs[2, :])
        cbar = plt.colorbar(sm_for_cbar, cax=cbar_ax, orientation="horizontal")
        pos = cbar_ax.get_position()
        cbar_ax.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height * 0.25])
        cbar.set_ticks([])
        cbar_ax.text(0, -0.8, "low", ha="left", va="center", fontsize=font_size)
        cbar_ax.text(
            1,
            -0.8,
            "high",
            ha="right",
            va="center",
            fontsize=font_size,
            transform=cbar_ax.transAxes,
        )
        cbar.set_label("Feature value (normalized)", fontsize=font_size)

    # PANEL C: Occlusion analysis
    highlight_frac = 0.45
    norm_mode = "local" 
    occlusion_window = 50
    occlusion_step = 15
    min_confidence = 0.75

    estimator = getattr(model, "rf", model)
    scaler = getattr(model, "scaler", None)

    example_map = {} 
    for pep in selected_peptides:
        idx, prob = select_best_example_idx_for_peptide(
            pep,
            df_feat,
            X_val_raw,
            model,
            feature_kwargs,
            min_confidence=min_confidence,
            scaler=scaler,
            estimator=estimator,
        )
        if idx is None:
            any_idxs = df_feat[df_feat["true_label"] == pep].index.tolist()
            idx = any_idxs[0] if any_idxs else None
            prob = None
        example_map[pep] = (idx, prob)

    pooled_rows = []
    for pep in selected_peptides:
        idx_val, prob_val = example_map.get(pep, (None, None))
        if idx_val is None:
            continue
        pooled_rows.append(
            {
                "scenario": str(scenario_name),
                "model_name": str(model_name),
                "subset": "panel_c_main",
                "peptide": str(pep),
                "example_idx": int(idx_val),
                "run_id": _safe_run_id_at(run_id_arr, idx_val),
                "confidence": float(prob_val) if prob_val is not None else np.nan,
                "true_label": str(df_feat.loc[idx_val, "true_label"]),
                "pred_label": str(df_feat.loc[idx_val, "pred_label"]),
                "correct": bool(df_feat.loc[idx_val, "correct"]),
            }
        )

    collected_vals = []
    for pep, (idx, _) in example_map.items():
        if idx is None:
            continue
        sig = X_val_raw[idx].copy()
        if len(sig) != resample_len and len(sig) > 1:
            xs_old = np.linspace(0, 1, len(sig))
            xs_new = np.linspace(0, 1, resample_len)
            sig = np.interp(xs_new, xs_old, sig)
        collected_vals.extend(sig.tolist())

    if collected_vals:
        global_signal_min = float(np.min(collected_vals))
        global_signal_max = float(np.max(collected_vals))
        pad = (global_signal_max - global_signal_min) * 0.05 
        global_signal_min -= pad
        global_signal_max += pad
    else:
        global_signal_min, global_signal_max = 0.0, 1.0

    all_imps = []
    panel_c_precomputed = {}
    for pep, (idx, _) in example_map.items():
        if idx is None:
            continue
        sig = X_val_raw[idx]
        # Resample signal first (match plot_occlusion_per_peptide behaviour) and
        # compute baseline from the resampled signal so occlusion traces match.
        if len(sig) != resample_len and len(sig) > 1:
            xs_old = np.linspace(0, 1, len(sig))
            xs_new = np.linspace(0, 1, resample_len)
            sig = np.interp(xs_new, xs_old, sig)

        # Get the predicted class index for this example to match plot_occlusion_per_peptide logic
        pred_label = df_feat.loc[idx, "pred_label"]
        class_idx = None
        if hasattr(estimator, "classes_"):
            try:
                class_idx = int(np.where(estimator.classes_ == pred_label)[0][0])
            except Exception:
                class_idx = None

        # sliding_occlusion_importance expects the signal at the requested length
        # so pass resample_len=None to avoid double-resampling (we already did it).
        pos_tmp, imps_tmp, base_prob_tmp = sliding_occlusion_importance(
            sig,
            model,
            feature_kwargs,
            true_class=class_idx,
            window_size=occlusion_window,
            step=occlusion_step,
            baseline_value=np.median(sig),
            resample_len=None,
        )
        panel_c_precomputed[str(pep)] = {
            "example_idx": int(idx),
            "positions": np.asarray(pos_tmp),
            "importance": np.asarray(imps_tmp),
            "baseprob": base_prob_tmp,
        }
        if len(imps_tmp) > 0:
            all_imps.extend(np.abs(imps_tmp))

    global_vmax = float(np.max(all_imps)) if all_imps else 1.0
    global_vmax = max(global_vmax, 1e-12)

    cmap_shared = LinearSegmentedColormap.from_list(
        "white_to_red_shared",
        [(0.0, "white"), (0.75, "white"), (1.0, (0.95, 0.68, 0.68))],
        N=256,
    )

    # Plot columns (Panel C) using the selected example indices
    axes_C = []
    for col_idx, pep in enumerate(selected_peptides):
        ax = fig.add_subplot(gs[3, col_idx])
        idx_for_plot, prob_for_plot = example_map.get(pep, (None, None))
        precomp = panel_c_precomputed.get(str(pep), {})
        plot_occlusion_per_peptide(
            ax,
            df_feat,
            X_val_raw,
            model,
            feature_kwargs,
            pep,
            color_map,
            resample_len,
            fontsize=font_size,
            global_y_min=global_signal_min,
            global_y_max=global_signal_max,
            global_max_importance=global_vmax,
            occlusion_window=occlusion_window,
            occlusion_step=occlusion_step,
            is_first_column=(col_idx == 0),
            highlight_frac=highlight_frac,
            min_confidence=min_confidence,
            norm_mode=norm_mode,
            cmap_shared=cmap_shared,
            example_idx=idx_for_plot,
            example_confidence=prob_for_plot,
            precomputed_positions=precomp.get("positions"),
            precomputed_importance=precomp.get("importance"),
            precomputed_base_prob=precomp.get("baseprob"),
        )
        axes_C.append(ax)
        if col_idx == 0:
            ax.text(
                -0.25,
                1.15,
                "c",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )

    # PANEL D: Shared occlusion colorbar 
    cbar_ax = fig.add_subplot(gs[4, :])
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_shared)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height * 0.25])
    cbar.set_ticks([])
    cbar_ax.text(0, -0.8, "low", ha="left", va="center", fontsize=font_size)
    cbar_ax.text(
        1,
        -0.8,
        "high",
        ha="right",
        va="center",
        fontsize=font_size,
        transform=cbar_ax.transAxes,
    )
    cbar.set_label(
        "Relative temporal importance (|ΔP|)",
        fontsize=font_size,
        rotation=0,
        labelpad=2,
    )
    cbar.ax.tick_params(labelsize=font_size)

    fig_path_png = os.path.join(output_dir, "Figure4_interpretability.png")
    fig_path_svg = os.path.join(output_dir, "Figure4_interpretability.svg")

    fig.savefig(fig_path_png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(fig_path_svg, dpi=300, bbox_inches="tight", pad_inches=0.05)
    logger.info("[%s] Saved Figure 4 to %s", scenario_name, fig_path_png)
    plt.close(fig)

    # additional figure 4C examples 
    def _score_example_confidence(sample_idx):
        try:
            sig_local = X_val_raw[sample_idx].copy()
            feats_df_local, _ = extract_interpretable_features(
                [sig_local], **feature_kwargs
            )
            if feats_df_local.empty:
                return None
            X_local = feats_df_local.values
            if scaler is not None:
                X_local = scaler.transform(X_local)
            proba_local = estimator.predict_proba(X_local)
            if np.ndim(proba_local) == 1:
                proba_local = np.asarray(proba_local).reshape(1, -1)
            if proba_local.shape[0] == 0:
                return None
            return float(np.max(proba_local[0]))
        except Exception:
            return None

    extra_rows = min(4, len(selected_peptides))
    extra_cols = 4
    extra_peptides = selected_peptides[:extra_rows]

    example_grid = {}
    for pep in extra_peptides:
        idxs_correct = df_feat[
            (df_feat["true_label"] == pep) & (df_feat["correct"])
        ].index.tolist()
        idx_pool = (
            idxs_correct
            if idxs_correct
            else df_feat[df_feat["true_label"] == pep].index.tolist()
        )
        scored = []
        for idx in idx_pool:
            p = _score_example_confidence(idx)
            if p is not None:
                scored.append((int(idx), float(p)))
        if not scored and idx_pool:
            scored = [(int(idx_pool[0]), None)]
        scored = sorted(scored, key=lambda x: (-(x[1] if x[1] is not None else -1.0)))
        example_grid[pep] = scored[:extra_cols]

    plotted_examples = []
    for pep in extra_peptides:
        for idx_val, p_val in example_grid.get(pep, []):
            plotted_examples.append((pep, idx_val, p_val))

    if plotted_examples:
        all_signal_vals_more = []
        all_imps_more = []
        for _, idx_val, _ in plotted_examples:
            sig = X_val_raw[idx_val].copy()
            if len(sig) != resample_len and len(sig) > 1:
                xs_old = np.linspace(0, 1, len(sig))
                xs_new = np.linspace(0, 1, resample_len)
                sig = np.interp(xs_new, xs_old, sig)
            all_signal_vals_more.extend(sig.tolist())

            _, imps_tmp, _ = sliding_occlusion_importance(
                sig,
                model,
                feature_kwargs,
                true_class=None,
                window_size=occlusion_window,
                step=occlusion_step,
                baseline_value=np.median(sig),
                resample_len=resample_len,
            )
            if len(imps_tmp) > 0:
                all_imps_more.extend(np.abs(np.asarray(imps_tmp, dtype=float)).tolist())

        if all_signal_vals_more:
            y_min_more = float(np.min(all_signal_vals_more))
            y_max_more = float(np.max(all_signal_vals_more))
            y_pad_more = (y_max_more - y_min_more) * 0.05
            y_min_more -= y_pad_more
            y_max_more += y_pad_more
        else:
            y_min_more, y_max_more = 0.0, 1.0

        vmax_more = float(np.max(all_imps_more)) if all_imps_more else 1.0
        vmax_more = max(vmax_more, 1e-12)
        cmap_shared_more = LinearSegmentedColormap.from_list(
            "white_to_red_shared_more",
            [(0.0, "white"), (0.75, "white"), (1.0, (0.95, 0.68, 0.68))],
            N=256,
        )

        fig_more_h = max(5.0, 1.85 * extra_rows + 0.9)
        fig_more = plt.figure(figsize=(6.7, fig_more_h))
        gs_more = fig_more.add_gridspec(
            extra_rows + 1,
            extra_cols,
            height_ratios=[1.0] * extra_rows + [0.12],
            hspace=0.55,
            wspace=0.35,
        )

        for r, pep in enumerate(extra_peptides):
            entries = example_grid.get(pep, [])
            for c in range(extra_cols):
                ax_more = fig_more.add_subplot(gs_more[r, c])
                if c >= len(entries):
                    ax_more.axis("off")
                    continue
                idx_plot, p_plot = entries[c]
                plot_occlusion_per_peptide(
                    ax_more,
                    df_feat,
                    X_val_raw,
                    model,
                    feature_kwargs,
                    pep,
                    color_map,
                    resample_len,
                    fontsize=font_size,
                    global_max_importance=vmax_more,
                    global_y_min=y_min_more,
                    global_y_max=y_max_more,
                    is_first_column=(c == 0),
                    occlusion_window=occlusion_window,
                    occlusion_step=occlusion_step,
                    highlight_frac=highlight_frac,
                    min_confidence=min_confidence,
                    norm_mode="local",
                    cmap_shared=cmap_shared_more,
                    example_idx=idx_plot,
                    example_confidence=p_plot,
                )

        cbar_ax_more = fig_more.add_subplot(gs_more[extra_rows, :])
        sm_more = ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_shared_more
        )
        sm_more.set_array([])
        cbar_more = fig_more.colorbar(
            sm_more, cax=cbar_ax_more, orientation="horizontal"
        )
        pos_more = cbar_ax_more.get_position()
        cbar_ax_more.set_position(
            [pos_more.x0, pos_more.y0 + 0.01, pos_more.width, pos_more.height * 0.25]
        )
        cbar_more.set_ticks([])
        cbar_ax_more.text(0, -0.8, "low", ha="left", va="center", fontsize=font_size)
        cbar_ax_more.text(
            1,
            -0.8,
            "high",
            ha="right",
            va="center",
            fontsize=font_size,
            transform=cbar_ax_more.transAxes,
        )
        cbar_more.set_label(
            "Relative temporal importance (per plot)",
            fontsize=font_size,
            rotation=0,
            labelpad=2,
        )

        fig_more_png = os.path.join(output_dir, "Figure4_Cmoreexamples.png")
        fig_more_svg = os.path.join(output_dir, "Figure4_Cmoreexamples.svg")
        fig_more.savefig(fig_more_png, dpi=600, bbox_inches="tight", pad_inches=0.05)
        fig_more.savefig(fig_more_svg, dpi=300, bbox_inches="tight", pad_inches=0.05)
        logger.info(
            "[%s] Saved additional Figure 4C grid to %s", scenario_name, fig_more_png
        )
        plt.close(fig_more)

        for pep, idx_val, p_val in plotted_examples:
            pooled_rows.append(
                {
                    "scenario": str(scenario_name),
                    "model_name": str(model_name),
                    "subset": "panel_c_more",
                    "peptide": str(pep),
                    "example_idx": int(idx_val),
                    "run_id": _safe_run_id_at(run_id_arr, idx_val),
                    "confidence": float(p_val) if p_val is not None else np.nan,
                    "true_label": str(df_feat.loc[idx_val, "true_label"]),
                    "pred_label": str(df_feat.loc[idx_val, "pred_label"]),
                    "correct": bool(df_feat.loc[idx_val, "correct"]),
                }
            )

    _write_selected_examples_tsv(
        output_dir, pooled_rows, "interpretability_pooled_examples.tsv"
    )

    # Export interpretability bundle for Figure4 replay
    import pickle
    bundle = {
        "shap_by_class": shap_by_class,
        "shap_summary_by_class": shap_summary_by_class,
        "occlusion_results": occlusion_results,
        "X_val_raw": X_val_raw,
        "df_feat": df_feat,
        "selected_peptides": selected_peptides,
        "color_map": color_map,
        "y_val_arr": y_val_arr,
        "y_pred_arr": y_pred_arr,
        "feature_order": feature_order,
        "model_name": model_name,
        "scenario_name": scenario_name,
        "resample_len": resample_len,
        "global_signal_min": global_signal_min,
        "global_signal_max": global_signal_max,
        "example_map": example_map,
        "global_vmax": global_vmax,
        "cmap_shared": cmap_shared,
        "font_size": font_size,
        "top_k": topk,
        "occlusion_window": occlusion_window,
        "occlusion_step": occlusion_step,
        "highlight_frac": highlight_frac,
        "norm_mode": norm_mode,
        "min_confidence": min_confidence,
        "global_topk_display": global_topk_display,
        "per_pep_topk": per_pep_topk,
        "panel_c_precomputed": panel_c_precomputed,
    }
    bundle_pkl = os.path.join(output_dir, "fig4_interpretability_bundle.pkl")
    with open(bundle_pkl, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("[%s] Saved Figure 4 interpretability bundle: %s", scenario_name, bundle_pkl)

    return


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


def plot_occlusion_per_peptide(
    ax,
    df_feat,
    signals,
    model,
    feature_kwargs,
    peptide,
    color_map,
    resample_len=300,
    fontsize=7,
    global_max_importance=None,
    global_y_min=None,
    global_y_max=None,
    is_first_column=True,
    occlusion_window=30,
    occlusion_step=5,
    highlight_frac=0.8,
    min_confidence=0.3,
    norm_mode="local",
    cmap_shared=None,
    example_idx=None,
    example_confidence=None,
    precomputed_positions=None,
    precomputed_importance=None,
    precomputed_base_prob=None,
):
    estimator = getattr(model, "rf", model)
    scaler = getattr(model, "scaler", None)
    est_classes = getattr(estimator, "classes_", None)

    def _find_class_index(classes_arr, label):
        """Robust lookup of label in classes_ (handles str/int/float/bytes)."""
        if classes_arr is None:
            return None
        idx = np.where(classes_arr == label)[0]
        if idx.size:
            return int(idx[0])
        s = str(label)
        for i, c in enumerate(classes_arr):
            if str(c) == s:
                return int(i)
        for cast in (int, float):
            try:
                labc = cast(label)
                idx = np.where(classes_arr == labc)[0]
                if idx.size:
                    return int(idx[0])
            except Exception:
                pass
        return None

    idx = None
    max_prob = None
    if example_idx is not None:
        idx_candidate = int(example_idx)
        if 0 <= idx_candidate < len(df_feat) and str(
            df_feat.loc[idx_candidate, "true_label"]
        ) == str(peptide):
            idx = idx_candidate
            max_prob = example_confidence
        else:
            logger.warning(
                "Provided example_idx %s does not belong to peptide %s. Falling back.",
                example_idx,
                peptide,
            )

    if idx is None:
        mask = (df_feat["true_label"] == peptide) & (df_feat["correct"])
        idxs = df_feat[mask].index.tolist()
        if not idxs:
            ax.text(
                0.5,
                0.5,
                "No correct classifications",
                ha="center",
                va="center",
                fontsize=fontsize,
            )
            ax.set_axis_off()
            return

        best_idx = None
        best_prob = -1.0
        for idx_try in idxs:
            try:
                sig_try = signals[idx_try].copy()
                feats_df, _ = extract_interpretable_features(
                    [sig_try], **feature_kwargs
                )
                if feats_df.empty:
                    continue
                X = feats_df.values
                if scaler is not None:
                    try:
                        X = scaler.transform(X)
                    except Exception:
                        continue
                proba = estimator.predict_proba(X)
                if proba.ndim == 1:
                    proba = proba.reshape(1, -1)

                pred_label_try = df_feat.loc[idx_try, "pred_label"]
                class_idx_try = _find_class_index(est_classes, pred_label_try)
                if class_idx_try is None:
                    prob_try = float(np.max(proba[0]))
                elif class_idx_try < proba.shape[1]:
                    prob_try = float(proba[0, class_idx_try])
                else:
                    prob_try = float(np.max(proba[0]))

                if prob_try > best_prob and prob_try >= min_confidence:
                    best_prob = prob_try
                    best_idx = idx_try
            except Exception:
                continue

        if best_idx is None:
            fallback_idx = None
            fallback_prob = -1.0
            for idx_try in idxs:
                try:
                    sig_try = signals[idx_try].copy()
                    feats_df, _ = extract_interpretable_features(
                        [sig_try], **feature_kwargs
                    )
                    if feats_df.empty:
                        continue
                    X = feats_df.values
                    if scaler is not None:
                        X = scaler.transform(X)
                    proba = estimator.predict_proba(X)
                    if proba.ndim == 1:
                        proba = proba.reshape(1, -1)
                    p = float(np.max(proba[0]))
                    if p > fallback_prob:
                        fallback_prob = p
                        fallback_idx = idx_try
                except Exception:
                    continue
            if fallback_idx is not None:
                idx = fallback_idx
                max_prob = float(fallback_prob)
            else:
                idx = int(idxs[0])
                max_prob = None
        else:
            idx = int(best_idx)
            max_prob = float(best_prob)

    sig = signals[idx].copy()
    true_label = df_feat.loc[idx, "true_label"]
    pred_label = df_feat.loc[idx, "pred_label"]

    if len(sig) != resample_len and len(sig) > 1:
        xs_old = np.linspace(0, 1, len(sig))
        xs_new = np.linspace(0, 1, resample_len)
        sig = np.interp(xs_new, xs_old, sig)

    class_idx = None
    if hasattr(model, "classes_"):
        try:
            class_idx = int(np.where(model.classes_ == pred_label)[0][0])
        except Exception:
            class_idx = None

    if (
        precomputed_positions is not None
        and precomputed_importance is not None
    ):
        pos = np.asarray(precomputed_positions)
        imps = np.asarray(precomputed_importance)
        base_prob = precomputed_base_prob
    else:
        pos, imps, base_prob = sliding_occlusion_importance(
            sig,
            model,
            feature_kwargs,
            true_class=class_idx,
            window_size=occlusion_window,
            step=occlusion_step,
            baseline_value=np.median(sig),
            resample_len=None,
        )

    if len(pos) == 0 or np.all(np.isnan(imps)):
        ax.text(
            0.5, 0.5, "Occlusion failed", ha="center", va="center", fontsize=fontsize
        )
        ax.set_axis_off()
        return

    abs_imps = np.abs(np.nan_to_num(imps, nan=0.0))
    local_vmax = float(np.max(abs_imps)) if abs_imps.size else 0.0

    if norm_mode == "global" and global_max_importance is not None:
        vmax_used = float(global_max_importance)
    else:
        vmax_used = max(local_vmax, 1e-12)

    cutoff = highlight_frac * vmax_used

    if cmap_shared is not None:
        cmap_custom = cmap_shared
    else:
        cmap_custom = LinearSegmentedColormap.from_list(
            "white_to_red_top",
            [(0.0, "white"), (0.75, "white"), (1.0, (0.95, 0.68, 0.68))],
            N=256,
        )
    norm = Normalize(vmin=0.0, vmax=vmax_used)

    time_points = np.arange(len(sig))
    ax.plot(time_points, sig, color="black", linewidth=1.0, alpha=0.9, zorder=5)
    if global_y_min is not None and global_y_max is not None:
        ax.set_ylim(global_y_min, global_y_max)

    positions = np.asarray(pos)
    if len(positions) > 1:
        diffs = np.diff(positions)
        left_bounds = np.empty_like(positions, dtype=float)
        right_bounds = np.empty_like(positions, dtype=float)
        for i in range(len(positions)):
            if i == 0:
                left = positions[i] - diffs[0] / 2.0
            else:
                left = positions[i] - diffs[i - 1] / 2.0
            if i == len(positions) - 1:
                right = positions[i] + diffs[-1] / 2.0
            else:
                right = positions[i] + diffs[i] / 2.0
            left_bounds[i] = max(0, left)
            right_bounds[i] = min(len(sig), right)
    else:
        half = max(1.0, resample_len * 0.01)
        left_bounds = positions - half
        right_bounds = positions + half
        left_bounds = np.clip(left_bounds, 0, len(sig))
        right_bounds = np.clip(right_bounds, 0, len(sig))

    cur_start = None
    cur_max = 0.0
    cur_end = None
    for i in range(len(positions)):
        a = abs_imps[i]
        left = left_bounds[i]
        right = right_bounds[i]
        if a >= cutoff:
            if cur_start is None:
                cur_start = left
                cur_max = a
                cur_end = right
            else:
                cur_end = right
                cur_max = max(cur_max, a)
        elif cur_start is not None:
            frac = (cur_max - cutoff) / max(1e-12, (vmax_used - cutoff))
            frac = float(np.clip(frac, 0.0, 1.0))
            alpha = float(np.clip(0.35 + 0.55 * frac, 0.0, 1.0))
            color = cmap_custom(norm(cur_max))
            ax.axvspan(
                cur_start,
                cur_end,
                ymin=0,
                ymax=1,
                color=color,
                alpha=alpha,
                linewidth=0,
                zorder=2,
            )
            cur_start = None
            cur_max = 0.0
            cur_end = None
    if cur_start is not None:
        frac = (cur_max - cutoff) / max(1e-12, (vmax_used - cutoff))
        frac = float(np.clip(frac, 0.0, 1.0))
        alpha = float(np.clip(0.35 + 0.55 * frac, 0.0, 1.0))
        color = cmap_custom(norm(cur_max))
        ax.axvspan(
            cur_start,
            cur_end,
            ymin=0,
            ymax=1,
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=2,
        )

    max_prob_str = (
        f"{max_prob:.3f}"
        if (max_prob is not None and isinstance(max_prob, (int, float)))
        else str(max_prob)
    )
    ax.set_title(f"{peptide} (#{idx})", fontsize=fontsize, pad=3)
    if is_first_column:
        ax.set_ylabel("Normalized Current", fontsize=fontsize)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    ax.set_xlabel("Signal index", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.grid(axis="x", alpha=0.12, linewidth=0.4)
    ax._occlusion_cmap = cmap_custom
    ax._occlusion_norm = norm
    ax._occlusion_vmax = vmax_used
    ax._occlusion_cutoff = cutoff


def select_best_example_idx_for_peptide(
    peptide,
    df_feat,
    signals,
    model,
    feature_kwargs,
    min_confidence=0.3,
    scaler=None,
    estimator=None,
):
    mask = (df_feat["true_label"] == peptide) & (df_feat["correct"])
    idxs = df_feat[mask].index.tolist()
    if len(idxs) == 0:
        return None, None

    est = estimator if estimator is not None else getattr(model, "rf", model)
    scl = scaler if scaler is not None else getattr(model, "scaler", None)
    est_classes = getattr(est, "classes_", None)

    def _find_class_index(classes_arr, label):
        if classes_arr is None:
            return None
        idx = np.where(classes_arr == label)[0]
        if idx.size:
            return int(idx[0])
        s = str(label)
        for i, c in enumerate(classes_arr):
            if str(c) == s:
                return int(i)
        for cast in (int, float):
            try:
                labc = cast(label)
                idx = np.where(classes_arr == labc)[0]
                if idx.size:
                    return int(idx[0])
            except Exception:
                pass
        return None

    best_idx = None
    max_prob = -1.0

    for idx in idxs:
        try:
            sig = signals[idx].copy()
            feats_df, _ = extract_interpretable_features([sig], **feature_kwargs)
            if feats_df.empty:
                continue
            X = feats_df.values
            if scl is not None:
                try:
                    X = scl.transform(X)
                except Exception:
                    # if scaling fails, skip this sample
                    continue

            proba = est.predict_proba(X)
            if proba.ndim == 1:
                proba = proba.reshape(1, -1)

            pred_label = df_feat.loc[idx, "pred_label"]
            class_idx = _find_class_index(est_classes, pred_label)

            if class_idx is None:
                try:
                    pred_from_model = est.predict(X)[0]
                    class_idx = _find_class_index(est_classes, pred_from_model)
                except Exception:
                    class_idx = None

            if class_idx is not None and class_idx < proba.shape[1]:
                prob = float(proba[0, class_idx])
            else:
                prob = float(np.max(proba[0]))

            if prob > max_prob and prob >= min_confidence:
                max_prob = prob
                best_idx = idx
        except Exception:
            continue

    if best_idx is None:
        fallback_best = None
        fallback_best_prob = -1.0
        for idx in idxs:
            try:
                sig = signals[idx].copy()
                feats_df, _ = extract_interpretable_features([sig], **feature_kwargs)
                if feats_df.empty:
                    continue
                X = feats_df.values
                if scl is not None:
                    X = scl.transform(X)
                proba = est.predict_proba(X)
                if proba.ndim == 1:
                    proba = proba.reshape(1, -1)
                p = float(np.max(proba[0]))
                if p > fallback_best_prob:
                    fallback_best_prob = p
                    fallback_best = idx
            except Exception:
                continue
        if fallback_best is not None:
            return fallback_best, float(fallback_best_prob)
        else:
            return None, None

    return best_idx, float(max_prob)


def _get_inception_keras_model(model):
    """Extract the underlying Keras model from various InceptionTime model types."""
    # Check for NativeInceptionTimeClassifier (has model_ attribute)
    if hasattr(model, "model_") and hasattr(model.model_, "predict"):
        return model.model_

    if hasattr(model, "_model"):
        return model._model

    # Direct keras model
    if hasattr(model, "predict") and hasattr(model, "trainable_variables"):
        return model

    if hasattr(model, "fit_dict") and "model" in model.fit_dict:
        return model.fit_dict["model"]

    return None


def compute_integrated_gradients_saliency(
    signal,
    model,
    class_idx,
    baseline_value=None,
    n_steps=50,
    resample_len=None,
):
    if tf is None:
        logger.warning("TensorFlow not available for integrated gradients.")
        return np.array([]), np.array([])

    try:
        keras_model = _get_inception_keras_model(model)
        if keras_model is None:
            logger.warning("Could not extract Keras model from InceptionTime model.")
            return np.array([]), np.array([])

        sig = np.asarray(signal, dtype=np.float32).copy()

        if resample_len is not None and len(sig) != resample_len and len(sig) > 1:
            xs_old = np.linspace(0, 1, len(sig))
            xs_new = np.linspace(0, 1, resample_len)
            sig = np.interp(xs_new, xs_old, sig)

        if baseline_value is None:
            baseline_value = 0.0

        baseline = np.full_like(sig, baseline_value, dtype=np.float32)

        alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
        interpolated_signals = np.array(
            [baseline + alpha * (sig - baseline) for alpha in alphas], dtype=np.float32
        )

        interpolated_signals = interpolated_signals[..., np.newaxis]

        # Compute gradients using GradientTape
        with tf.GradientTape() as tape:
            inputs = tf.Variable(interpolated_signals, trainable=True, dtype=tf.float32)
            logits = keras_model(inputs, training=False)

            # Get the output for the target class
            target_logits = logits[:, class_idx]

        grads = tape.gradient(target_logits, inputs)

        if grads is None:
            logger.warning("Gradient computation returned None for class %d", class_idx)
            return np.array([]), np.array([])

        grads_np = np.abs(grads.numpy())  # shape: (n_steps, seq_len, 1)

        # Average across interpolation steps
        saliency = np.mean(grads_np, axis=(0, 2))  # shape: (seq_len,)

        positions = np.arange(len(sig), dtype=np.float32)

        return positions, saliency

    except Exception as e:
        logger.warning("Integrated gradients computation failed: %s", str(e))
        return np.array([]), np.array([])


def plot_inception_saliency_per_peptide(
    ax,
    df_feat,
    signals,
    model,
    peptide,
    color_map,
    resample_len=300,
    fontsize=7,
    global_max_saliency=None,
    global_y_min=None,
    global_y_max=None,
    is_first_column=True,
    highlight_frac=0.8,
    min_confidence=0.3,
    norm_mode="local",
    cmap_shared=None,
    example_idx=None,
    example_confidence=None,
):
    # Find correctly classified examples
    mask = (df_feat["true_label"] == peptide) & (df_feat["correct"])
    idxs = df_feat[mask].index.tolist()
    if not idxs:
        ax.text(
            0.5,
            0.5,
            "No correct classifications",
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        ax.set_axis_off()
        return

    # Get class index
    classes = getattr(model, "classes_", None)
    class_idx = 0
    if classes is not None:
        try:
            class_idx = int(np.where(classes == peptide)[0][0])
        except Exception:
            class_idx = 0

    if example_idx is not None:
        idx = example_idx
        max_prob = example_confidence
    else:
        best_idx = None
        max_prob = -1.0
        for cand_idx in idxs:
            try:
                sig = signals[cand_idx].copy()

                keras_model = _get_inception_keras_model(model)
                if keras_model is None:
                    continue

                sig_prep = np.asarray(sig, dtype=np.float32)
                if len(sig_prep) != resample_len and len(sig_prep) > 1:
                    xs_old = np.linspace(0, 1, len(sig_prep))
                    xs_new = np.linspace(0, 1, resample_len)
                    sig_prep = np.interp(xs_new, xs_old, sig_prep)

                sig_prep = sig_prep[np.newaxis, :, np.newaxis]
                proba = keras_model(
                    tf.constant(sig_prep, dtype=tf.float32), training=False
                ).numpy()
                prob = float(proba[0, class_idx])

                if prob > max_prob and prob >= min_confidence:
                    max_prob = prob
                    best_idx = cand_idx
            except Exception as e:
                logger.debug("Error evaluating candidate %s: %s", cand_idx, str(e))
                continue

        if best_idx is None:
            best_idx = idxs[0] if idxs else None
            max_prob = None

        idx = best_idx

    if idx is None:
        ax.text(
            0.5, 0.5, "No example found", ha="center", va="center", fontsize=fontsize
        )
        ax.set_axis_off()
        return

    sig = signals[idx].copy()
    true_label = df_feat.loc[idx, "true_label"]

    if len(sig) != resample_len and len(sig) > 1:
        xs_old = np.linspace(0, 1, len(sig))
        xs_new = np.linspace(0, 1, resample_len)
        sig = np.interp(xs_new, xs_old, sig)

    positions, saliency = compute_integrated_gradients_saliency(
        sig,
        model,
        class_idx,
        baseline_value=np.median(sig),
        n_steps=50,
        resample_len=resample_len,
    )

    if len(positions) == 0 or len(saliency) == 0:
        ax.text(
            0.5, 0.5, "Saliency failed", ha="center", va="center", fontsize=fontsize
        )
        ax.set_axis_off()
        return

    # Normalize saliency
    abs_saliency = np.abs(np.nan_to_num(saliency, nan=0.0))
    local_vmax = float(np.max(abs_saliency)) if abs_saliency.size else 0.0

    if norm_mode == "global" and global_max_saliency is not None:
        vmax_used = float(global_max_saliency)
    else:
        vmax_used = max(local_vmax, 1e-12)

    cutoff = highlight_frac * vmax_used

    if cmap_shared is not None:
        cmap_custom = cmap_shared
    else:
        cmap_custom = LinearSegmentedColormap.from_list(
            "white_to_red_ig",
            [(0.0, "white"), (0.75, "white"), (1.0, (0.95, 0.68, 0.68))],
            N=256,
        )
    norm = Normalize(vmin=0.0, vmax=vmax_used)

    time_points = np.arange(len(sig))
    ax.plot(time_points, sig, color="black", linewidth=1.0, alpha=0.9, zorder=5)
    if global_y_min is not None and global_y_max is not None:
        ax.set_ylim(global_y_min, global_y_max)

    # Compute bounds for saliency bars
    if len(positions) > 1:
        diffs = np.diff(positions)
        left_bounds = np.empty_like(positions, dtype=float)
        right_bounds = np.empty_like(positions, dtype=float)
        for i in range(len(positions)):
            if i == 0:
                left = positions[i] - diffs[0] / 2.0
            else:
                left = positions[i] - diffs[i - 1] / 2.0
            if i == len(positions) - 1:
                right = positions[i] + diffs[-1] / 2.0
            else:
                right = positions[i] + diffs[i] / 2.0
            left_bounds[i] = max(0, left)
            right_bounds[i] = min(len(sig), right)
    else:
        half = max(1.0, resample_len * 0.01)
        left_bounds = positions - half
        right_bounds = positions + half
        left_bounds = np.clip(left_bounds, 0, len(sig))
        right_bounds = np.clip(right_bounds, 0, len(sig))

    cur_start = None
    cur_max = 0.0
    cur_end = None
    for i in range(len(positions)):
        a = abs_saliency[i]
        left = left_bounds[i]
        right = right_bounds[i]
        if a >= cutoff:
            if cur_start is None:
                cur_start = left
                cur_max = a
                cur_end = right
            else:
                cur_end = right
                cur_max = max(cur_max, a)
        elif cur_start is not None:
            frac = (cur_max - cutoff) / max(1e-12, (vmax_used - cutoff))
            frac = float(np.clip(frac, 0.0, 1.0))
            alpha = float(np.clip(0.35 + 0.55 * frac, 0.0, 1.0))
            color = cmap_custom(norm(cur_max))
            ax.axvspan(
                cur_start,
                cur_end,
                ymin=0,
                ymax=1,
                color=color,
                alpha=alpha,
                linewidth=0,
                zorder=2,
            )
            cur_start = None
            cur_max = 0.0
            cur_end = None
    if cur_start is not None:
        frac = (cur_max - cutoff) / max(1e-12, (vmax_used - cutoff))
        frac = float(np.clip(frac, 0.0, 1.0))
        alpha = float(np.clip(0.35 + 0.55 * frac, 0.0, 1.0))
        color = cmap_custom(norm(cur_max))
        ax.axvspan(
            cur_start,
            cur_end,
            ymin=0,
            ymax=1,
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=2,
        )

    max_prob_str = (
        f"{max_prob:.3f}"
        if (max_prob is not None and isinstance(max_prob, (int, float)))
        else str(max_prob)
    )
    ax.set_title(f"{peptide} (#{idx})", fontsize=fontsize, pad=3)
    if is_first_column:
        ax.set_ylabel("Normalized Current", fontsize=fontsize)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    ax.set_xlabel("Signal index", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.grid(axis="x", alpha=0.12, linewidth=0.4)

    # Store colormap for colorbar
    ax._saliency_cmap = cmap_custom
    ax._saliency_norm = norm
    ax._saliency_vmax = vmax_used


def run_inception_interpretability_pipeline(
    model,
    model_name,
    X_val_raw,
    y_val,
    y_pred,
    output_dir,
    scenario_name,
    specific_peptides=None,
    val_run_ids=None,
    seed=385,
    top_n_peptides=4,
    resample_len=300,
    example_map=None,
):
    os.makedirs(output_dir, exist_ok=True)
    _set_interpretability_seed(seed)

    y_val_arr = np.asarray(y_val)
    run_id_arr = np.asarray(val_run_ids) if val_run_ids is not None else None
    y_pred_arr = np.asarray(y_pred)

    df_feat = pd.DataFrame(
        {
            "true_label": y_val_arr,
            "pred_label": y_pred_arr,
        }
    )
    df_feat["correct"] = df_feat["true_label"] == df_feat["pred_label"]

    if specific_peptides is not None and len(specific_peptides) > 0:
        available_peptides = sorted(list(pd.unique(df_feat["true_label"])))
        selected_peptides = [
            pep for pep in specific_peptides if pep in available_peptides
        ]
        if not selected_peptides:
            logger.warning(
                "[%s] No specified peptides found. Using auto-selection.", scenario_name
            )
            specific_peptides = None

    if specific_peptides is None or len(selected_peptides) == 0:
        # Auto-select: most misclassified peptides
        per_class = pd.DataFrame(
            {"peptide": df_feat["true_label"], "correct": df_feat["correct"]}
        )
        summary = per_class.groupby("peptide").agg(
            total=("correct", "size"), n_correct=("correct", "sum")
        )
        summary["n_incorrect"] = summary["total"] - summary["n_correct"]
        summary["error_rate"] = summary["n_incorrect"] / summary["total"].replace(
            0, np.nan
        )
        selected_peptides = _select_peptides_deterministic(
            summary, top_n_peptides, by_error=True
        )
        if not selected_peptides or summary["n_incorrect"].sum() == 0:
            selected_peptides = _select_peptides_deterministic(
                summary, top_n_peptides, by_error=False
            )
        logger.info(
            "[%s] Auto-selected peptides for Figure 4-IG: %s",
            scenario_name,
            selected_peptides,
        )

    selected_peptides = selected_peptides[:4]
    n_peptides = len(selected_peptides)

    cmap = get_colors_for_classes(sorted(list(pd.unique(df_feat["true_label"]))))
    color_map = cmap.as_dict()

    if example_map is None:
        example_map = {}
        for pep in selected_peptides:
            idx_list = df_feat[df_feat["true_label"] == pep].index.tolist()
            if idx_list:
                example_map[pep] = (idx_list[0], None)

    pooled_rows = []
    for pep in selected_peptides:
        idx_val, prob_val = example_map.get(pep, (None, None))
        if idx_val is None:
            continue
        pooled_rows.append(
            {
                "scenario": str(scenario_name),
                "model_name": str(model_name),
                "subset": "panel_ig_main",
                "peptide": str(pep),
                "example_idx": int(idx_val),
                "run_id": _safe_run_id_at(run_id_arr, idx_val),
                "confidence": float(prob_val) if prob_val is not None else np.nan,
                "true_label": str(df_feat.loc[idx_val, "true_label"]),
                "pred_label": str(df_feat.loc[idx_val, "pred_label"]),
                "correct": bool(df_feat.loc[idx_val, "correct"]),
            }
        )

    # global y-limits
    collected_vals = []
    for pep, (idx, _) in example_map.items():
        if idx is None:
            continue
        sig = X_val_raw[idx].copy()
        if len(sig) != resample_len and len(sig) > 1:
            xs_old = np.linspace(0, 1, len(sig))
            xs_new = np.linspace(0, 1, resample_len)
            sig = np.interp(xs_new, xs_old, sig)
        collected_vals.extend(sig.tolist())

    if collected_vals:
        global_signal_min = float(np.min(collected_vals))
        global_signal_max = float(np.max(collected_vals))
        pad = (global_signal_max - global_signal_min) * 0.05
        global_signal_min -= pad
        global_signal_max += pad
    else:
        global_signal_min, global_signal_max = 0.0, 1.0

    # Compute global saliency max
    all_saliencies = []
    for pep, (idx, _) in example_map.items():
        if idx is None:
            continue
        sig = X_val_raw[idx]
        class_idx = 0
        if hasattr(model, "classes_"):
            try:
                class_idx = int(np.where(model.classes_ == pep)[0][0])
            except Exception:
                class_idx = 0

        _, sal_tmp = compute_integrated_gradients_saliency(
            sig,
            model,
            class_idx,
            baseline_value=np.median(sig),
            n_steps=50,
            resample_len=resample_len,
        )
        if len(sal_tmp) > 0:
            all_saliencies.extend(np.abs(sal_tmp))

    global_vmax = float(np.max(all_saliencies)) if all_saliencies else 1.0
    global_vmax = max(global_vmax, 1e-12)

    highlight_frac = 0.45
    cmap_shared = LinearSegmentedColormap.from_list(
        "white_to_red_ig_shared",
        [(0.0, "white"), (0.75, "white"), (1.0, (0.95, 0.68, 0.68))],
        N=256,
    )

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
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig_width = 6.7
    fig_height = 9.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        5,
        max(1, n_peptides),
        height_ratios=[1.0, 1.0, 0.15, 0.7, 0.15],
        hspace=0.45,
        wspace=0.3,
    )

    for col_idx, pep in enumerate(selected_peptides):
        ax = fig.add_subplot(gs[3, col_idx])
        idx_for_plot, prob_for_plot = example_map.get(pep, (None, None))
        plot_inception_saliency_per_peptide(
            ax,
            df_feat,
            X_val_raw,
            model,
            pep,
            color_map,
            resample_len,
            fontsize=font_size,
            global_y_min=global_signal_min,
            global_y_max=global_signal_max,
            global_max_saliency=global_vmax,
            is_first_column=(col_idx == 0),
            highlight_frac=highlight_frac,
            min_confidence=0.3,
            norm_mode="local",
            cmap_shared=cmap_shared,
            example_idx=idx_for_plot,
            example_confidence=prob_for_plot,
        )
        if col_idx == 0:
            ax.text(
                -0.25,
                1.15,
                "a",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )
    for empty_col in range(n_peptides, max(1, n_peptides)):
        fig.add_subplot(gs[3, empty_col]).axis("off")

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[4, :])
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_shared)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height * 0.25])
    cbar.set_ticks([])
    cbar_ax.text(0, -0.8, "low", ha="left", va="center", fontsize=font_size)
    cbar_ax.text(
        1,
        -0.8,
        "high",
        ha="right",
        va="center",
        fontsize=font_size,
        transform=cbar_ax.transAxes,
    )
    cbar.set_label(
        "Relative temporal importance (Integrated Gradients)",
        fontsize=font_size,
        rotation=0,
        labelpad=2,
    )
    cbar.ax.tick_params(labelsize=font_size)

    fig_path_png = os.path.join(output_dir, f"Figure4_InceptionTime_IG_saliency.png")
    fig_path_svg = os.path.join(output_dir, f"Figure4_InceptionTime_IG_saliency.svg")

    fig.savefig(fig_path_png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(fig_path_svg, dpi=300, bbox_inches="tight", pad_inches=0.05)
    logger.info(
        "[%s] Saved InceptionTime saliency figure to %s", scenario_name, fig_path_png
    )
    plt.close(fig)

    # additional InceptionTime saliency examples
    def _score_inception_example_confidence(sample_idx, peptide_label):
        try:
            keras_model = _get_inception_keras_model(model)
            if keras_model is None:
                return None
            class_idx_local = 0
            classes_local = getattr(model, "classes_", None)
            if classes_local is not None:
                try:
                    class_idx_local = int(
                        np.where(classes_local == peptide_label)[0][0]
                    )
                except Exception:
                    class_idx_local = 0

            sig_local = np.asarray(X_val_raw[sample_idx], dtype=np.float32).copy()
            if len(sig_local) != resample_len and len(sig_local) > 1:
                xs_old = np.linspace(0, 1, len(sig_local))
                xs_new = np.linspace(0, 1, resample_len)
                sig_local = np.interp(xs_new, xs_old, sig_local)

            sig_local = sig_local[np.newaxis, :, np.newaxis]
            proba_local = keras_model(
                tf.constant(sig_local, dtype=tf.float32), training=False
            ).numpy()
            if np.ndim(proba_local) != 2 or proba_local.shape[0] == 0:
                return None
            if class_idx_local < proba_local.shape[1]:
                return float(proba_local[0, class_idx_local])
            return float(np.max(proba_local[0]))
        except Exception:
            return None

    extra_rows = min(4, len(selected_peptides))
    extra_cols = 4
    extra_peptides = selected_peptides[:extra_rows]

    example_grid = {}
    for pep in extra_peptides:
        idxs_correct = df_feat[
            (df_feat["true_label"] == pep) & (df_feat["correct"])
        ].index.tolist()
        idx_pool = (
            idxs_correct
            if idxs_correct
            else df_feat[df_feat["true_label"] == pep].index.tolist()
        )
        scored = []
        for idx in idx_pool:
            p = _score_inception_example_confidence(int(idx), pep)
            if p is not None:
                scored.append((int(idx), float(p)))
        if not scored and idx_pool:
            scored = [(int(idx_pool[0]), None)]
        scored = sorted(
            scored, key=lambda x: (-(x[1] if x[1] is not None else -1.0), x[0])
        )
        example_grid[pep] = scored[:extra_cols]

    plotted_examples = []
    for pep in extra_peptides:
        for idx_val, p_val in example_grid.get(pep, []):
            plotted_examples.append((pep, idx_val, p_val))

    if plotted_examples:
        all_signal_vals_more = []
        all_saliencies_more = []
        for pep, idx_val, _ in plotted_examples:
            sig = np.asarray(X_val_raw[idx_val], dtype=float).copy()
            if len(sig) != resample_len and len(sig) > 1:
                xs_old = np.linspace(0, 1, len(sig))
                xs_new = np.linspace(0, 1, resample_len)
                sig = np.interp(xs_new, xs_old, sig)
            all_signal_vals_more.extend(sig.tolist())

            class_idx_more = 0
            if hasattr(model, "classes_"):
                try:
                    class_idx_more = int(np.where(model.classes_ == pep)[0][0])
                except Exception:
                    class_idx_more = 0
            _, sal_tmp = compute_integrated_gradients_saliency(
                sig,
                model,
                class_idx_more,
                baseline_value=np.median(sig),
                n_steps=50,
                resample_len=resample_len,
            )
            if len(sal_tmp) > 0:
                all_saliencies_more.extend(
                    np.abs(np.asarray(sal_tmp, dtype=float)).tolist()
                )

        if all_signal_vals_more:
            y_min_more = float(np.min(all_signal_vals_more))
            y_max_more = float(np.max(all_signal_vals_more))
            y_pad_more = (y_max_more - y_min_more) * 0.05
            y_min_more -= y_pad_more
            y_max_more += y_pad_more
        else:
            y_min_more, y_max_more = 0.0, 1.0

        vmax_more = float(np.max(all_saliencies_more)) if all_saliencies_more else 1.0
        vmax_more = max(vmax_more, 1e-12)
        cmap_shared_more = LinearSegmentedColormap.from_list(
            "white_to_red_ig_shared_more",
            [(0.0, "white"), (0.75, "white"), (1.0, (0.95, 0.68, 0.68))],
            N=256,
        )

        fig_more_h = max(5.0, 1.85 * extra_rows + 0.9)
        fig_more = plt.figure(figsize=(6.7, fig_more_h))
        gs_more = fig_more.add_gridspec(
            extra_rows + 1,
            extra_cols,
            height_ratios=[1.0] * extra_rows + [0.12],
            hspace=0.55,
            wspace=0.35,
        )

        for r, pep in enumerate(extra_peptides):
            entries = example_grid.get(pep, [])
            for c in range(extra_cols):
                ax_more = fig_more.add_subplot(gs_more[r, c])
                if c >= len(entries):
                    ax_more.axis("off")
                    continue
                idx_plot, p_plot = entries[c]
                plot_inception_saliency_per_peptide(
                    ax_more,
                    df_feat,
                    X_val_raw,
                    model,
                    pep,
                    color_map,
                    resample_len,
                    fontsize=font_size,
                    global_max_saliency=vmax_more,
                    global_y_min=y_min_more,
                    global_y_max=y_max_more,
                    is_first_column=(c == 0),
                    highlight_frac=highlight_frac,
                    min_confidence=0.3,
                    norm_mode="local",
                    cmap_shared=cmap_shared_more,
                    example_idx=idx_plot,
                    example_confidence=p_plot,
                )

        cbar_ax_more = fig_more.add_subplot(gs_more[extra_rows, :])
        sm_more = ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_shared_more
        )
        sm_more.set_array([])
        cbar_more = fig_more.colorbar(
            sm_more, cax=cbar_ax_more, orientation="horizontal"
        )
        pos_more = cbar_ax_more.get_position()
        cbar_ax_more.set_position(
            [pos_more.x0, pos_more.y0 + 0.01, pos_more.width, pos_more.height * 0.25]
        )
        cbar_more.set_ticks([])
        cbar_ax_more.text(0, -0.8, "low", ha="left", va="center", fontsize=font_size)
        cbar_ax_more.text(
            1,
            -0.8,
            "high",
            ha="right",
            va="center",
            fontsize=font_size,
            transform=cbar_ax_more.transAxes,
        )
        cbar_more.set_label(
            "Relative temporal importance (per plot, Integrated Gradients)",
            fontsize=font_size,
            rotation=0,
            labelpad=2,
        )

        fig_more_png = os.path.join(
            output_dir, "Figure4_InceptionTime_IG_moreexamples.png"
        )
        fig_more_svg = os.path.join(
            output_dir, "Figure4_InceptionTime_IG_moreexamples.svg"
        )
        fig_more.savefig(fig_more_png, dpi=600, bbox_inches="tight", pad_inches=0.05)
        fig_more.savefig(fig_more_svg, dpi=300, bbox_inches="tight", pad_inches=0.05)
        logger.info(
            "[%s] Saved additional InceptionTime saliency grid to %s",
            scenario_name,
            fig_more_png,
        )
        plt.close(fig_more)

        for pep, idx_val, p_val in plotted_examples:
            pooled_rows.append(
                {
                    "scenario": str(scenario_name),
                    "model_name": str(model_name),
                    "subset": "panel_ig_more",
                    "peptide": str(pep),
                    "example_idx": int(idx_val),
                    "run_id": _safe_run_id_at(run_id_arr, idx_val),
                    "confidence": float(p_val) if p_val is not None else np.nan,
                    "true_label": str(df_feat.loc[idx_val, "true_label"]),
                    "pred_label": str(df_feat.loc[idx_val, "pred_label"]),
                    "correct": bool(df_feat.loc[idx_val, "correct"]),
                }
            )

    _write_selected_examples_tsv(
        output_dir, pooled_rows, "interpretability_pooled_examples.tsv"
    )

    return {
        "figure_png": fig_path_png,
        "figure_svg": fig_path_svg,
        "peptides": selected_peptides,
        "n_peptides": n_peptides,
        "global_vmax": global_vmax,
        "example_map": example_map,
    }


def replay_figure_4_from_bundle(bundle, output_dir):
    """
    Recreate Figure 4 from a precomputed interpretability bundle.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack bundle
    shap_by_class = bundle["shap_by_class"]
    shap_summary_by_class = bundle["shap_summary_by_class"]
    X_val_raw = bundle["X_val_raw"]
    df_feat = bundle["df_feat"]
    selected_peptides = bundle["selected_peptides"]
    color_map = bundle["color_map"]
    example_map = bundle["example_map"]
    global_signal_min = bundle["global_signal_min"]
    global_signal_max = bundle["global_signal_max"]
    global_vmax = bundle.get("global_vmax", 1.0)
    cmap_shared = bundle.get("cmap_shared", LinearSegmentedColormap.from_list(
        "white_to_red_shared",
        [(0.0, "white"), (0.75, "white"), (1.0, (0.95, 0.68, 0.68))],
        N=256,
    ))
    font_size = bundle.get("font_size", 5)
    top_k = bundle.get("top_k", 10)
    occlusion_window = bundle.get("occlusion_window", 50)
    occlusion_step = bundle.get("occlusion_step", 15)
    highlight_frac = bundle.get("highlight_frac", 0.45)
    norm_mode = bundle.get("norm_mode", "local")
    resample_len = bundle.get("resample_len", 300)
    global_topk_display = bundle.get("global_topk_display", [])
    panel_c_precomputed = bundle.get("panel_c_precomputed", {})
    
    # Get feature order from bundle or from summary
    feature_order = bundle.get("feature_order", None)
    if feature_order is None:
        feature_order = list(shap_summary_by_class[selected_peptides[0]].index)
    
    features_df = df_feat[[c for c in df_feat.columns if c not in ["true_label", "pred_label", "correct"]]]
    
    # Set matplotlib rcParams to match workflow exactly
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
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    
    # Create figure
    fig_width = 6.7
    fig_height = 9.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        5,
        len(selected_peptides),
        height_ratios=[1.0, 1.0, 0.15, 0.7, 0.15],
        hspace=0.45,
        wspace=0.3,
    )
    
    # Compute global_topk_display if needed
    if not global_topk_display:
        per_pep_topk = {}
        for pep in selected_peptides:
            per_pep_topk[pep] = shap_summary_by_class[pep].head(top_k).index.tolist()
        
        union_topk = []
        for pep in selected_peptides:
            union_topk.extend(per_pep_topk[pep])
        
        if len(union_topk) == 0:
            logger.warning("No top features found, using all features")
            global_topk_display = feature_order[:top_k]
        else:
            # Order by mean importance
            mean_imp = {}
            for f in set(union_topk):
                vals = []
                for pep in selected_peptides:
                    if pep in shap_summary_by_class and f in shap_summary_by_class[pep].index:
                        vals.append(shap_summary_by_class[pep][f])
                mean_imp[f] = float(np.mean(vals)) if vals else 0.0
            
            global_ordered = sorted(set(union_topk), key=lambda x: mean_imp.get(x, 0.0), reverse=True)
            max_rows = 10
            if len(global_ordered) > max_rows:
                global_topk_display = global_ordered[:max_rows]
            else:
                global_topk_display = global_ordered
    
    global_topk_short = [shorten_feature_name(f) for f in global_topk_display]
    n_rows = len(global_topk_display)
    
    # PANEL A: barh with aligned rows across all peptide columns
    for col_idx, pep in enumerate(selected_peptides):
        ax = fig.add_subplot(gs[0, col_idx])
        
        summ = shap_summary_by_class.get(pep)
        if summ is None:
            ax.text(0.5, 0.5, "SHAP missing", ha="center")
            ax.set_axis_off()
            continue
        
        vals_plot = np.array(
            [float(summ.get(f, 0.0)) for f in global_topk_display], dtype=float
        )
        
        # horizontal bar plot
        y_pos = np.arange(n_rows)
        ax.barh(y_pos, vals_plot, color="red", align="center")
        # show labels only on left-most column to avoid overlap
        if col_idx == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(global_topk_short, fontsize=font_size)
        else:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([""] * n_rows)
        
        ax.invert_yaxis()
        xpad = np.max(vals_plot) * 0.02 if np.max(vals_plot) > 0 else 0.001
        for i, v in enumerate(vals_plot):
            ax.text(
                v + xpad, i, f"{v:.3f}", va="center", fontsize=font_size, color="red"
            )
        
        ax.set_xlabel("mean |SHAP| value", fontsize=font_size)
        ax.set_title(pep, fontsize=font_size + 1)
        
        if col_idx == 0:
            ax.text(
                -0.25,
                1.15,
                "a",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )
    
    # PANEL B: Combined beeswarm per peptide
    sm_list = []
    for col_idx, pep in enumerate(selected_peptides):
        ax = fig.add_subplot(gs[1, col_idx])
        if pep not in shap_by_class:
            ax.text(0.5, 0.5, "no SHAP", ha="center")
            ax.set_axis_off()
            continue
        
        shap_arr = shap_by_class[pep]
        mask = df_feat["true_label"] == pep
        idxs = np.where(mask)[0].tolist()
        if len(idxs) == 0:
            ax.text(0.5, 0.5, "no samples", ha="center")
            ax.set_axis_off()
            continue
        
        # For beeswarm we use the same rows (global_topk_display)
        # show y tick labels only on leftmost column
        show_yticklabels = col_idx == 0
        
        sm = plot_colored_violin_shapstyle(
            ax=ax,
            shap_arr=shap_arr,
            features_df=features_df,
            idxs=idxs,
            topk=global_topk_display,
            cols=feature_order,
            cmap_name="coolwarm",
            font_size=font_size,
            show_yticklabels=show_yticklabels,
            violin_width=0.35,
        )
        
        if sm is not None:
            sm_list.append(sm)
        
        if col_idx == 0:
            ax.text(
                -0.25,
                1.15,
                "b",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )
    
    # shared colorbar for panel B
    sm_for_cbar = sm_list[0] if sm_list else None
    if sm_for_cbar is not None:
        cbar_ax = fig.add_subplot(gs[2, :])
        cbar = plt.colorbar(sm_for_cbar, cax=cbar_ax, orientation="horizontal")
        pos = cbar_ax.get_position()
        cbar_ax.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height * 0.25])
        cbar.set_ticks([])
        cbar_ax.text(0, -0.8, "low", ha="left", va="center", fontsize=font_size)
        cbar_ax.text(1, -0.8, "high", ha="right", va="center", fontsize=font_size, transform=cbar_ax.transAxes)
        cbar.set_label("Feature value (normalized)", fontsize=font_size)
    
    # PANEL C: Occlusion analysis
    # Use _plot_occlusion_precomputed_for_replay which is the exact copy of workflow logic
    axes_C = []
    for col_idx, pep in enumerate(selected_peptides):
        ax = fig.add_subplot(gs[3, col_idx])
        idx_for_plot, prob_for_plot = example_map.get(pep, (None, None))
        
        if idx_for_plot is None or not (0 <= int(idx_for_plot) < len(X_val_raw)):
            ax.text(0.5, 0.5, "No occlusion example", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            axes_C.append(ax)
            continue
        
        idx_for_plot = int(idx_for_plot)
        sig = X_val_raw[idx_for_plot].copy()
        
        if len(sig) != resample_len and len(sig) > 1:
            xs_old = np.linspace(0, 1, len(sig))
            xs_new = np.linspace(0, 1, resample_len)
            sig = np.interp(xs_new, xs_old, sig)
        
        # Get precomputed occlusion data
        occl_entry = panel_c_precomputed.get(pep)
        if occl_entry is None:
            # Empty plot if no data
            _plot_occlusion_precomputed_for_replay(
                ax=ax,
                signal=sig,
                positions=np.array([]),
                importances=np.array([]),
                peptide=pep,
                idx=idx_for_plot,
                fontsize=font_size,
                cmap_shared=cmap_shared,
                global_y_min=global_signal_min,
                global_y_max=global_signal_max,
                global_vmax=global_vmax,
                highlight_frac=highlight_frac,
                norm_mode=norm_mode,
                is_first_column=(col_idx == 0),
            )
        else:
            _plot_occlusion_precomputed_for_replay(
                ax=ax,
                signal=sig,
                positions=occl_entry.get("positions", []),
                importances=occl_entry.get("importance", []),
                peptide=pep,
                idx=idx_for_plot,
                fontsize=font_size,
                cmap_shared=cmap_shared,
                global_y_min=global_signal_min,
                global_y_max=global_signal_max,
                global_vmax=global_vmax,
                highlight_frac=highlight_frac,
                norm_mode=norm_mode,
                is_first_column=(col_idx == 0),
            )
        
        axes_C.append(ax)
        if col_idx == 0:
            ax.text(
                -0.25,
                1.15,
                "c",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )
    
    # PANEL D: Shared occlusion colorbar
    cbar_ax = fig.add_subplot(gs[4, :])
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_shared)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height * 0.25])
    cbar.set_ticks([])
    cbar_ax.text(0, -0.8, "low", ha="left", va="center", fontsize=font_size)
    cbar_ax.text(
        1,
        -0.8,
        "high",
        ha="right",
        va="center",
        fontsize=font_size,
        transform=cbar_ax.transAxes,
    )
    cbar.set_label(
        "Relative temporal importance (|ΔP|)",
        fontsize=font_size,
        rotation=0,
        labelpad=2,
    )
    cbar.ax.tick_params(labelsize=font_size)
    
    # Save figure
    fig_path_png = os.path.join(output_dir, "Figure4_interpretability.png")
    fig_path_svg = os.path.join(output_dir, "Figure4_interpretability.svg")
    
    fig.savefig(fig_path_png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(fig_path_svg, dpi=300, bbox_inches="tight", pad_inches=0.05)
    logger.info("Saved Figure 4 replay to %s", fig_path_png)
    plt.close(fig)
    
    return fig_path_png, fig_path_svg


def _plot_occlusion_precomputed_for_replay(
    ax,
    signal,
    positions,
    importances,
    peptide,
    idx,
    fontsize,
    cmap_shared,
    global_y_min,
    global_y_max,
    global_vmax,
    highlight_frac,
    norm_mode,
    is_first_column,
):
    sig = np.asarray(signal, dtype=float)
    pos = np.asarray(positions, dtype=float)
    imps = np.asarray(importances, dtype=float)
    
    time_points = np.arange(len(sig))
    ax.plot(time_points, sig, color="black", linewidth=1.0, alpha=0.9, zorder=5)
    if global_y_min is not None and global_y_max is not None:
        ax.set_ylim(global_y_min, global_y_max)
    
    abs_imps = np.abs(np.nan_to_num(imps, nan=0.0))
    local_vmax = float(np.max(abs_imps)) if abs_imps.size else 0.0
    
    if norm_mode == "global" and global_vmax is not None:
        vmax_used = float(global_vmax)
    else:
        vmax_used = max(local_vmax, 1e-12)
    
    cutoff = highlight_frac * vmax_used
    norm = Normalize(vmin=0.0, vmax=vmax_used)
    
    if len(pos) > 0:
        if len(pos) > 1:
            diffs = np.diff(pos)
            left_bounds = np.empty_like(pos, dtype=float)
            right_bounds = np.empty_like(pos, dtype=float)
            for i in range(len(pos)):
                if i == 0:
                    left = pos[i] - diffs[0] / 2.0
                else:
                    left = pos[i] - diffs[i - 1] / 2.0
                if i == len(pos) - 1:
                    right = pos[i] + diffs[-1] / 2.0
                else:
                    right = pos[i] + diffs[i] / 2.0
                left_bounds[i] = max(0, left)
                right_bounds[i] = min(len(sig), right)
        else:
            half = max(1.0, len(sig) * 0.01)
            left_bounds = pos - half
            right_bounds = pos + half
            left_bounds = np.clip(left_bounds, 0, len(sig))
            right_bounds = np.clip(right_bounds, 0, len(sig))
        
        cur_start = None
        cur_max = 0.0
        cur_end = None
        for i in range(len(pos)):
            a = abs_imps[i]
            left = left_bounds[i]
            right = right_bounds[i]
            if a >= cutoff:
                if cur_start is None:
                    cur_start = left
                    cur_max = a
                    cur_end = right
                else:
                    cur_end = right
                    cur_max = max(cur_max, a)
            elif cur_start is not None:
                frac = (cur_max - cutoff) / max(1e-12, (vmax_used - cutoff))
                frac = float(np.clip(frac, 0.0, 1.0))
                alpha = float(np.clip(0.35 + 0.55 * frac, 0.0, 1.0))
                color = cmap_shared(norm(cur_max))
                ax.axvspan(
                    cur_start,
                    cur_end,
                    ymin=0,
                    ymax=1,
                    color=color,
                    alpha=alpha,
                    linewidth=0,
                    zorder=2,
                )
                cur_start = None
                cur_end = None
                cur_max = 0.0
        if cur_start is not None:
            frac = (cur_max - cutoff) / max(1e-12, (vmax_used - cutoff))
            frac = float(np.clip(frac, 0.0, 1.0))
            alpha = float(np.clip(0.35 + 0.55 * frac, 0.0, 1.0))
            color = cmap_shared(norm(cur_max))
            ax.axvspan(
                cur_start,
                cur_end,
                ymin=0,
                ymax=1,
                color=color,
                alpha=alpha,
                linewidth=0,
                zorder=2,
            )
    
    ax.set_title(f"{peptide} (#{idx})", fontsize=fontsize, pad=3)
    if is_first_column:
        ax.set_ylabel("Normalized Current", fontsize=fontsize)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    ax.set_xlabel("Signal index", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.grid(axis="x", alpha=0.12, linewidth=0.4)
