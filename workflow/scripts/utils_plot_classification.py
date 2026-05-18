#!/usr/bin/env python
import csv
import hashlib
import logging
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

logger = logging.getLogger("nanopore-peptide-classifier")
logger.addHandler(logging.NullHandler())


def normalize_by_row(cm):
    cm = np.asarray(cm, dtype=float)
    rowsum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(cm, rowsum, where=(rowsum != 0))
    out[np.isnan(out)] = 0.0
    return out


def _diag_annotations(percent_mat, true_labels, pred_labels, threshold=10):
    if percent_mat.size == 0:
        return np.array([[]], dtype=object)
    annot = np.full(percent_mat.shape, "", dtype=object)
    for r in range(percent_mat.shape[0]):
        row = percent_mat[r].astype(float)
        if pred_labels.size:
            matches = np.where(pred_labels == true_labels[r])[0]
            if matches.size:
                annot[r, matches[0]] = f"{int(round(row[matches[0]]))}"
        topk = min(2, row.size)
        for idx in np.argsort(row)[-topk:][::-1]:
            annot[r, idx] = f"{int(round(row[idx]))}"
        for idx in np.where(row > threshold)[0]:
            annot[r, idx] = f"{int(round(row[idx]))}"
    return annot


def _rectangular_confusion(y_true, y_pred):
    if y_true.size == 0:
        return np.array([[]], dtype=int), np.array([]), np.array([])
    rows = np.unique(y_true)
    cols = np.unique(y_pred)
    cm = np.zeros((len(rows), len(cols)), dtype=int)
    for i, r in enumerate(rows):
        mask = y_true == r
        for j, c in enumerate(cols):
            cm[i, j] = int(np.sum(mask & (y_pred == c)))
    return cm, rows, cols


def plot_confusion_matrices(
    output_path,
    val_metrics,
    test_metrics,
    normalize="row",
    outfile_name="confusion_matrices",
    scenario_name=None,
):
    os.makedirs(output_path, exist_ok=True)

    # Export metrics for reproducibility  
    if scenario_name:
        try:
            from figure_bundle_io import save_pickle_bundle
            bundle = {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "scenario_name": scenario_name,
            }
            # Create filename matching Figure3.py expectations
            scenario_lower = scenario_name.lower().replace("ß", "ss")
            if scenario_name == "ßCAT_single_variants":
                bundle_filename = "fig3_bcat_single_val_test_metrics.pkl"
            elif scenario_name == "ALL_classes":
                bundle_filename = "fig3_all_classes_val_test_metrics.pkl"
            else:
                # For other scenarios, just use a safe filename
                bundle_filename = f"fig3_{scenario_lower}_val_test_metrics.pkl"
            
            bundle_path = os.path.join(output_path, bundle_filename)
            save_pickle_bundle(bundle_path, bundle)
            logger.info(f"Exported Figure 3 metrics bundle to {bundle_path}")
        except Exception as e:
            logger.warning(f"Failed to export Figure 3 metrics bundle: {e}")

    def safe_get(m, k, default):
        return m.get(k, default) if isinstance(m, dict) else default

    val_acc = safe_get(val_metrics, "Accuracy (w/o unknown)", 0.0)
    val_acc_exc = safe_get(val_metrics, "Accuracy (w/o unknown & w/o crude)", 0.0)
    val_acc_bal = safe_get(val_metrics, "Balanced Accuracy (w/o unknown)", 0.0)
    test_acc = safe_get(test_metrics, "Accuracy (w/o unknown)", 0.0)
    test_acc_exc = safe_get(test_metrics, "Accuracy (w/o unknown & w/o crude)", 0.0)
    test_acc_bal = safe_get(test_metrics, "Balanced Accuracy (w/o unknown)", 0.0)
    val_top2 = safe_get(val_metrics, "Top-2 (w/o unknown)", 0.0)
    val_top3 = safe_get(val_metrics, "Top-3 (w/o unknown)", 0.0)
    test_top2 = safe_get(test_metrics, "Top-2 (w/o unknown)", 0.0)
    test_top3 = safe_get(test_metrics, "Top-3 (w/o unknown)", 0.0)
    val_prec = safe_get(val_metrics, "Weighted Precision", 0.0)
    test_prec = safe_get(test_metrics, "Weighted Precision", 0.0)

    val_cm = np.asarray(
        safe_get(val_metrics, "Confusion Matrix (all)", np.array([[]])), dtype=int
    )
    val_labels = np.asarray(np.unique(safe_get(val_metrics, "y_true", np.array([]))))

    test_known_cm = np.array([[]])
    test_row_labels = np.array([])
    test_col_labels = np.array([])

    if isinstance(test_metrics, dict):
        y_all = np.asarray(test_metrics.get("y_true", []))
        preds_all = np.asarray(test_metrics.get("y_pred", []))
        mask_unknown = (
            np.asarray(
                test_metrics.get("mask_unknown", np.zeros_like(y_all, dtype=bool))
            )
            if y_all.size
            else np.array([])
        )

        if y_all.size and preds_all.size:
            keep = ~mask_unknown
            y_t, y_p = y_all[keep], preds_all[keep]
            test_known_cm, test_row_labels, test_col_labels = _rectangular_confusion(
                y_t, y_p
            )
        if (
            test_known_cm.size == 0
        ) and "Confusion Matrix (w/o unknown)" in test_metrics:
            test_known_cm = np.asarray(
                test_metrics["Confusion Matrix (w/o unknown)"], dtype=int
            )
            test_row_labels = np.asarray(
                test_metrics.get("Labels (confusion_matrix_known)", [])
            )
            test_col_labels = np.asarray(
                test_metrics.get("Labels (confusion_matrix_known)", [])
            )

    val_percent = normalize_by_row(val_cm) * 100 if val_cm.size else np.array([[]])
    test_percent = (
        normalize_by_row(test_known_cm) * 100 if test_known_cm.size else np.array([[]])
    )

    val_diag = _diag_annotations(val_percent, val_labels, val_labels)
    test_diag = _diag_annotations(test_percent, test_row_labels, test_col_labels)

    val_plot = normalize_by_row(val_cm) if normalize == "row" else val_cm.astype(float)
    test_plot = (
        normalize_by_row(test_known_cm)
        if normalize == "row" and test_known_cm.size
        else (test_known_cm.astype(float) if test_known_cm.size else np.array([[]]))
    )

    val_title = (
        f"Validation\nBalanced Prec: {val_prec:.2%}\n"
        f"Balanced Recall: {val_acc_bal:.2%}\nTop-2 Recall: {val_top2:.2%}\nTop-3 Recall: {val_top3:.2%}"
    )
    test_title = (
        f"Test (known-only)\nBalanced Prec: {test_prec:.2%}\n"
        f"Balanced Recall: {test_acc_bal:.2%}\nTop-2 Recall: {test_top2:.2%}\nTop-3 Recall: {test_top3:.2%}"
    )

    def get_figsize(n_labels):
        if n_labels <= 4:
            return (2, 1.5)
        elif n_labels <= 9:
            return (2.5, 2) 
        else:
            return (3, 2.5)

    val_annot_full = (
        val_diag
        if (
            hasattr(val_diag, "shape")
            and val_diag.size > 0
            and val_diag.shape == val_plot.shape
        )
        else None
    )
    test_annot_full = (
        test_diag
        if (
            hasattr(test_diag, "shape")
            and test_diag.size > 0
            and test_diag.shape == test_plot.shape
        )
        else None
    )

    datasets = [
        (
            "validation",
            val_plot,
            val_cm,
            val_percent,
            val_labels,
            val_labels,
            val_annot_full,
            val_title,
        ),
        (
            "test",
            test_plot,
            test_known_cm,
            test_percent,
            test_row_labels,
            test_col_labels,
            test_annot_full,
            test_title,
        ),
    ]

    figures = [
        ("A_counts", True, None),  
        ("B_diagacc", False, "diagacc"),
        ("C_colors", False, "colors"),
    ]

    for suffix, show_counts, plot_type in figures:
        for (
            name,
            plot_data,
            cm_raw,
            percent_data,
            y_labels,
            x_labels,
            diag_annot,
            title_text,
        ) in datasets:
            if not (plot_data.size and x_labels.size and y_labels.size):
                continue  # skip if no data

            figsize = get_figsize(len(x_labels))

            with plt.rc_context(
                {
                    "font.size": 5,
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Arial", "DejaVu Sans"],
                    "axes.labelsize": 5,
                    "axes.titlesize": 5,
                    "xtick.labelsize": 5,
                    "ytick.labelsize": 5,
                    "legend.fontsize": 5,
                    "axes.linewidth": 0.4,
                    "xtick.major.width": 0.4,
                    "ytick.major.width": 0.4,
                    "xtick.major.size": 2,
                    "ytick.major.size": 2,
                }
            ):
                fig, ax = plt.subplots(1, 1, figsize=figsize)

           
            if show_counts:
                
                color_data = plot_data
                annot_data = cm_raw
                fmt_str = "d"
                vmin, vmax = (0, 1) if normalize == "row" else (None, None)
                cbar_label = None
            else:
                if plot_type == "diagacc":
                    color_data = percent_data
                    annot_data = diag_annot
                    fmt_str = ""
                    vmin, vmax = 0, 100
                    cbar_label = ""
                else:  # "colors"
                    color_data = plot_data
                    annot_data = False  
                    fmt_str = ""
                    vmin, vmax = (0, 1) if normalize == "row" else (None, None)
                    cbar_label = ""
            hm = sns.heatmap(
                color_data,
                annot=annot_data,
                fmt=fmt_str,
                cmap="Blues",
                ax=ax,
                xticklabels=x_labels,
                yticklabels=y_labels,
                vmin=vmin,
                vmax=vmax,
                annot_kws={"size": 5},
                cbar_kws={"label": cbar_label} if cbar_label else {},
            )
            if hasattr(hm, "collections") and hm.collections:
                cbar = hm.collections[0].colorbar
                if cbar:
                    cbar.ax.tick_params(
                        labelsize=5,  
                        length=1,  
                        width=0.3,
                        pad=2,
                    )
                    if cbar_label:
                        cbar.ax.yaxis.label.set_fontsize(5)

            ax.set_xlabel("Predicted", fontsize=5, labelpad=2)
            ax.set_ylabel("True Label", fontsize=5, labelpad=2)

            ax.tick_params(
                axis="both",
                which="both",
                left=False, 
                bottom=False,  
                labelsize=5,
            )

            # Rotate labels if many classes
            if len(x_labels) > 3:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                plt.setp(ax.get_yticklabels(), rotation=0)
            else:
                plt.setp(ax.get_xticklabels(), rotation=0)
                plt.setp(ax.get_yticklabels(), rotation=0)

            plt.tight_layout()
            for ext in ("png", "svg"):
                plt.savefig(
                    os.path.join(output_path, f"{outfile_name}_{suffix}_{name}.{ext}"),
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.close(fig)

    # unknowns
    plot_unknown_confusion(
        test_metrics,
        output_path,
        outfile_name="confusion_matrices_unknown",
    )
    plot_unknown_mixtures(test_metrics, output_path, outfile_name=outfile_name)
    plot_unknown_predicted_only(test_metrics, output_path, top_k=5, show_all=False)
    
    # Export unknown metrics for Figure 3 reproducibility (for ALL_classes scenario)
    if scenario_name == "ALL_classes":
        try:
            from figure_bundle_io import save_pickle_bundle
            unknown_bundle = {
                "test_metrics": test_metrics,
                "scenario_name": scenario_name,
            }
            unknown_bundle_path = os.path.join(output_path, "fig3_unknown_test_metrics.pkl")
            save_pickle_bundle(unknown_bundle_path, unknown_bundle)
            logger.info(f"Exported Figure 3 unknown metrics bundle to {unknown_bundle_path}")
        except Exception as e:
            logger.warning(f"Failed to export Figure 3 unknown metrics bundle: {e}")

    # confidence-accuracy curve
    plot_confidence_accuracy_curve(val_metrics, output_path, dataset_name="valid")
    plot_confidence_accuracy_curve(test_metrics, output_path, dataset_name="test")

    plot_pr_compare_same_colors(
        val_metrics,
        test_metrics,
        labels_a_name="Validation",
        labels_b_name="Test",
        output_path=output_path,
        all_possible_labels=val_labels,
        top_k_legend=6,
        label_location="below",  # or "inside"
    )


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

    fixed_colors = {cls: colorblind_hex[idx] for cls, idx in fixed_assignments.items()}

    remaining_classes = [c for c in class_labels if c not in fixed_colors]
    unique_remaining = sorted(
        set(remaining_classes), key=lambda x: hashlib.md5(x.encode()).hexdigest()
    )

    n_extra = len(unique_remaining)
    extra_map = {}
    if n_extra > 0:
        if len(fixed_colors) + n_extra <= 10:
            used_indices = set(fixed_assignments.values())
            available_indices = [i for i in range(10) if i not in used_indices]
            extra_colors = [colorblind_hex[i] for i in available_indices[:n_extra]]
        else:
            extra_colors = sns.husl_palette(n_extra, s=0.9, l=0.65)
        extra_map = dict(zip(unique_remaining, extra_colors))

    color_map = {**fixed_colors, **extra_map}
    return [color_map[cls] for cls in class_labels]


def plot_pr_compare_same_colors(
    metrics_a,
    metrics_b,
    labels_a_name,
    labels_b_name,
    output_path,
    all_possible_labels,
    top_k_legend=6,
    percent=True,
    tick_step=10,
    figsize=(5.5, 1.75),
    dpi=600,
    label_location="below",
    font_family="sans-serif",
    font_size=10,
):
    os.makedirs(output_path, exist_ok=True)

    def _unpack(metrics):
        y_all = np.asarray(metrics.get("y_true", []))
        probs = np.asarray(metrics.get("Probabilities", None))
        if probs is None:
            raise ValueError("Metrics must include 'Probabilities' (N x C array).")
        mask_unknown = np.asarray(
            metrics.get("mask_unknown", np.zeros_like(y_all, dtype=bool))
        )
        keep = ~mask_unknown
        y_known = y_all[keep]
        probs_known = probs[keep]
        cols_labels = metrics.get("Labels (confusion_matrix_all)", None)
        if cols_labels is None or len(cols_labels) != probs_known.shape[1]:
            cols_labels = all_possible_labels
        cols_labels = [str(x) for x in cols_labels]
        return y_known, probs_known, cols_labels

    y_a, probs_a, cols_a = _unpack(metrics_a)
    y_b, probs_b, cols_b = _unpack(metrics_b)

    canonical = [str(x) for x in all_possible_labels]
    C_total = len(canonical)

    color_list = get_colors_for_classes(canonical)
    color_map = dict(zip(canonical, color_list))

    def _select_present(y_known, probs_known, cols_labels):
        present = set([str(x) for x in y_known])
        cols_idx = [i for i, lab in enumerate(cols_labels) if str(lab) in present]
        labels_sel = [cols_labels[i] for i in cols_idx]
        probs_sel = probs_known[:, cols_idx]
        return labels_sel, probs_sel

    labels_a, probs_a_sel = _select_present(y_a, probs_a, cols_a)
    labels_b, probs_b_sel = _select_present(y_b, probs_b, cols_b)

    def _build_bin(y_known, labels_sel):
        N = len(y_known)
        C = len(labels_sel)
        mat = np.zeros((N, C), dtype=int)
        y_known_str = np.array([str(x) for x in y_known])
        for j, lab in enumerate(labels_sel):
            mat[:, j] = (y_known_str == str(lab)).astype(int)
        return mat

    ybin_a = _build_bin(y_a, labels_a)
    ybin_b = _build_bin(y_b, labels_b)

    def _compute_prs(ybin, probs_sel, labels_sel):
        per_class = []
        ap_map = {}
        for j, lab in enumerate(labels_sel):
            npos = int(ybin[:, j].sum())
            if npos == 0:
                continue
            prec, rec, thr = precision_recall_curve(ybin[:, j], probs_sel[:, j])
            ap = average_precision_score(ybin[:, j], probs_sel[:, j])
            per_class.append(
                {
                    "label": str(lab),
                    "precision": prec,
                    "recall": rec,
                    "ap": ap,
                    "n_pos": npos,
                }
            )
            ap_map[str(lab)] = float(ap)
        return per_class, ap_map

    per_a, ap_map_a = _compute_prs(ybin_a, probs_a_sel, labels_a)
    per_b, ap_map_b = _compute_prs(ybin_b, probs_b_sel, labels_b)

    combined_ap = {
        lab: max(ap_map_a.get(lab, 0.0), ap_map_b.get(lab, 0.0))
        for lab in set(list(ap_map_a.keys()) + list(ap_map_b.keys()))
    }
    labels_present_union = sorted(
        combined_ap.keys(), key=lambda x: -combined_ap.get(x, 0.0)
    )

    fig_width = max(figsize[0] * 1.6, 6.0)  # scale width, but keep at least 6.0
    fig_height = figsize[1]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        sharey=True,
        sharex=False,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.18},
    )

    def _set_label_fonts(ax):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(font_family)
            lbl.set_fontsize(max(8, font_size - 1))

    def _plot_panel(ax, per_list, dataset_name):
        for c in per_list:
            lab = c["label"]
            color = color_map.get(lab, (0.2, 0.2, 0.2))
            rec = c["recall"] * (100.0 if percent else 1.0)
            prec = c["precision"] * (100.0 if percent else 1.0)
            ax.plot(rec, prec, linewidth=1.0, alpha=1, color=color, zorder=1)
        ax.set_title(f"{dataset_name}", fontsize=font_size + 1, fontfamily=font_family)
        ax.set_xlabel(
            "Recall (%)" if percent else "Recall",
            fontfamily=font_family,
            fontsize=font_size,
        )
        ax.set_ylabel(
            "Precision (%)" if percent else "Precision",
            fontfamily=font_family,
            fontsize=font_size,
        )
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 100 if percent else 1.0)
        ax.set_ylim(0, 100 if percent else 1.0)
        if percent:
            ax.xaxis.set_major_locator(MultipleLocator(tick_step))
            ax.yaxis.set_major_locator(MultipleLocator(tick_step))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        _set_label_fonts(ax)

    _plot_panel(axes[0], per_a, labels_a_name)
    _plot_panel(axes[1], per_b, labels_b_name)

    axes[1].set_ylabel("")  # remove right panel y-label to avoid repetition

    peptides_shown = [
        lab
        for lab in canonical
        if lab in set(list(ap_map_a.keys()) + list(ap_map_b.keys()))
    ]
    n = len(peptides_shown)

    proxy_handles = []
    proxy_labels = []
    for lab in peptides_shown:
        val_ap = ap_map_a.get(lab, None)
        test_ap = ap_map_b.get(lab, None)
        val_str = f"{val_ap:.2f}" if val_ap is not None else "-"
        test_str = f"{test_ap:.2f}" if test_ap is not None else "-"
        label_text = f"{lab} {val_str}/{test_str}"
        proxy_handles.append(Line2D([0], [0], color=color_map.get(lab), lw=4))
        proxy_labels.append(label_text)

    if n <= 8:
        cols = 3
    else:
        cols = 4

    fig.legend(
        proxy_handles,
        proxy_labels,
        loc="lower center",
        ncol=cols,
        bbox_to_anchor=(0.5, -0.75),
        frameon=False,
        fontsize=max(6, font_size - 2),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    base = os.path.join(output_path, f"PR_compare_{labels_a_name}_vs_{labels_b_name}")
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)
    return {"png": base + ".png", "svg": base + ".svg"}


def compute_macro_pr_curve(per_class, n_points=500, recall_grid=None):
    if recall_grid is None:
        recall_grid = np.linspace(0.0, 1.0, n_points)

    valid = [
        c
        for c in per_class
        if ("precision" in c and "recall" in c and len(c["recall"]) > 0)
    ]
    if len(valid) == 0:
        return recall_grid, np.zeros_like(recall_grid), 0.0

    interp_precisions = []
    ap_vals = []
    for c in valid:
        prec = np.asarray(c["precision"])
        rec = np.asarray(c["recall"])
        p_interp = np.interp(recall_grid, rec, prec, left=prec[0], right=prec[-1])
        interp_precisions.append(p_interp)
        ap_vals.append(float(c.get("ap", 0.0)))

    # Average across classes (macro)
    precision_macro = np.nanmean(np.vstack(interp_precisions), axis=0)
    macro_ap = float(np.mean(ap_vals)) if len(ap_vals) > 0 else 0.0

    return recall_grid, precision_macro, macro_ap


def plot_precision_recall_publication(
    metrics,
    output_path,
    dataset_name="test",
    top_k_legend=6,
    tick_step=10,
    figsize=(12, 7.5),  
    dpi=600,
    percent=True,
    save_pdf=False,
    show_per_class_lines=True,
    font_family="sans-serif",
    font_size=10,
):
    if not isinstance(metrics, dict):
        logger.warning("metrics must be a dict-like object")

    y_all = np.asarray(metrics.get("y_true", []))
    probs = metrics.get("Probabilities", None)
    if probs is None:
        logger.warning("metrics must contain 'Probabilities' (N x C array)")

    probs = np.asarray(probs)
    mask_unknown = np.asarray(
        metrics.get("mask_unknown", np.zeros_like(y_all, dtype=bool))
    )
    keep = ~mask_unknown
    y_known = np.asarray(y_all)[keep]
    probs_known = probs[keep]

    if y_known.size == 0 or probs_known.size == 0:
        logger.warning("No known samples available after applying mask_unknown.")

    labels_cols = np.asarray(metrics.get("Labels (confusion_matrix_all)", []))
    if labels_cols is None or len(labels_cols) != probs_known.shape[1]:
        # fallback: try 'y_pred' unique order or use indices
        y_pred = np.asarray(metrics.get("y_pred", []))
        if y_pred is not None and len(np.unique(y_pred)) == probs_known.shape[1]:
            labels_cols = np.unique(y_pred)
        else:
            labels_cols = np.arange(probs_known.shape[1])
    labels_cols = [str(x) for x in labels_cols]

    present_labels = set([str(x) for x in y_known])
    cols_to_keep = [i for i, lab in enumerate(labels_cols) if lab in present_labels]
    if len(cols_to_keep) == 0:
        logger.warning("No probability columns match classes present in y_true.")

    probs_sel = probs_known[:, cols_to_keep]
    class_labels = [labels_cols[i] for i in cols_to_keep]
    C = probs_sel.shape[1]
    N = probs_sel.shape[0]

    y_bin = np.zeros((N, C), dtype=int)
    y_known_str = np.array([str(x) for x in y_known])
    for j, lbl in enumerate(class_labels):
        y_bin[:, j] = (y_known_str == lbl).astype(int)

    per_class = []
    for j, lbl in enumerate(class_labels):
        n_pos = int(y_bin[:, j].sum())
        if n_pos == 0:
            continue
        prec, rec, thr = precision_recall_curve(y_bin[:, j], probs_sel[:, j])
        ap = average_precision_score(y_bin[:, j], probs_sel[:, j])
        per_class.append(
            {"label": lbl, "precision": prec, "recall": rec, "ap": ap, "n_pos": n_pos}
        )

    if len(per_class) == 0:
        logger.warning(
            "No valid per-class PR curves computed (probabilities degenerate?)."
        )
        return 0

    p_micro, r_micro, _ = precision_recall_curve(y_bin.ravel(), probs_sel.ravel())
    ap_micro = average_precision_score(y_bin, probs_sel, average="micro")
    ap_macro = np.mean([c["ap"] for c in per_class])

    per_class_sorted = sorted(per_class, key=lambda x: x["ap"], reverse=True)
    top_for_legend = per_class_sorted[:top_k_legend]
    other_for_plot = per_class_sorted[top_k_legend:]

    plt.rcParams.update({"font.size": font_size, "font.family": font_family})
    fig, ax = plt.subplots(figsize=figsize)

    if show_per_class_lines:
        for c in other_for_plot:
            rec = c["recall"] * (100.0 if percent else 1.0)
            prec = c["precision"] * (100.0 if percent else 1.0)
            ax.plot(rec, prec, linewidth=0.9, alpha=0.18, zorder=1, label="_nolegend_")
    handles = []
    labels = []
    for c in top_for_legend:
        rec = c["recall"] * (100.0 if percent else 1.0)
        prec = c["precision"] * (100.0 if percent else 1.0)
        (ln,) = ax.plot(rec, prec, linewidth=1.6, alpha=0.95, zorder=3)
        handles.append(ln)
        labels.append(f"{c['label']} (AP={c['ap']:.2f}, n={c['n_pos']})")

    rec_grid, prec_macro, macro_ap = compute_macro_pr_curve(per_class, n_points=500)
    scale = 100.0 if percent else 1.0
    (ln_macro_line,) = ax.plot(
        rec_grid * scale,
        prec_macro * scale,
        linestyle="--",
        linewidth=2.0,
        color="C1",
        alpha=0.95,
        zorder=30,
        label=f"macro (AP={macro_ap:.3f})",
    )

    if (p_micro is not None) and (r_micro is not None):
        (ln_micro_line,) = ax.plot(
            r_micro * scale,
            p_micro * scale,
            linewidth=3.0,
            color="black",
            alpha=1.0,
            zorder=40,
            label=f"micro (AP={ap_micro:.3f})",
        )
        handles.append(ln_micro_line)
        labels.append(f"micro (AP={ap_micro:.3f})")
    else:
        handles.insert(0, ln_macro_line)
        labels.insert(0, f"macro (AP={macro_ap:.3f})")

    proxy = Line2D([0], [0], linewidth=0, color="w")
    handles.append(proxy)
    labels.append(f"macro AP={ap_macro:.3f} | classes={len(per_class)} | samples={N}")

    ax.set_title(f"{dataset_name.title()}: Precision–Recall", fontsize=font_size + 1)
    ax.set_xlabel("Recall (%)" if percent else "Recall", fontsize=font_size)
    ax.set_ylabel("Precision (%)" if percent else "Precision", fontsize=font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25)

    if percent:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    else:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)

    leg = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=max(7, font_size - 1),
        frameon=False,
    )

    text_str = f"micro AP = {ap_micro:.3f}\nmacro AP = {ap_macro:.3f}\nclasses plotted = {len(per_class)}\nsamples = {N}"
    ax.text(
        0.98,
        0.02,
        text_str,
        transform=ax.transAxes,
        fontsize=max(7, font_size - 1),
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
    )

    plt.tight_layout(rect=[0, 0, 0.78, 1.0])  # leave room on the right for legend

    os.makedirs(output_path, exist_ok=True)
    base = os.path.join(output_path, f"Precision_Recall_{dataset_name}")
    png_path = base + ".png"
    svg_path = base + ".svg"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    tsv_path = os.path.join(output_path, f"precision_recall_micro_{dataset_name}.tsv")
    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["recall", "precision"])
        for r, p in zip(r_micro, p_micro):
            writer.writerow([f"{r:.6f}", f"{p:.6f}"])

    summary = {
        "png": png_path,
        "svg": svg_path,
        "tsv_micro": tsv_path,
        "micro_ap": float(ap_micro),
        "macro_ap": float(ap_macro),
        "n_classes_plotted": len(per_class),
        "n_samples": int(N),
        "top_legend": [c["label"] for c in top_for_legend],
    }
    return summary


def plot_recall_precision_curve(
    metrics,
    output_path,
    dataset_name="test",
    tick_step_x=10,
    tick_step_y=10,
    show_per_class=True,
    percent=True,
):
    if not isinstance(metrics, dict):
        return

    y_all = np.asarray(metrics.get("y_true", []))
    probs = metrics.get("Probabilities")
    if probs is None:
        logger.warning("Skipping PR plot: missing per-class probabilities.")
        return
    probs = np.asarray(probs)

    mask_unknown = np.asarray(
        metrics.get("mask_unknown", np.zeros_like(y_all, dtype=bool))
    )
    keep = ~mask_unknown
    y_known = y_all[keep]
    probs_known = probs[keep]

    if y_known.size == 0 or probs_known.size == 0:
        logger.warning("No known samples to plot PR curve.")
        return

    # Determine class labels in the same order as columns of probs_known
    pred_labels = np.asarray(metrics.get("Labels (confusion_matrix_all)", []))
    if len(pred_labels) != probs_known.shape[1]:
        unique_preds = np.unique(np.asarray(metrics.get("y_pred", [])))
        if len(unique_preds) == probs_known.shape[1]:
            pred_labels = unique_preds
        else:
            # fallback: take sorted unique of y_known and hope it matches
            pred_labels = np.unique(y_known)
            if len(pred_labels) != probs_known.shape[1]:
                # final fallback: columns -> 0..C-1
                pred_labels = np.arange(probs_known.shape[1])
                logger.warning(
                    "Could not reliably align class labels to probability columns; using indices."
                )

    # Sanity check: probabilities not degenerate
    max_probs = np.max(probs_known, axis=1)
    if np.std(max_probs) < 1e-6:
        logger.warning(
            "Probabilities appear degenerate — PR curves may be uninformative."
        )

    n_classes = probs_known.shape[1]

    y_bin = np.zeros((y_known.shape[0], n_classes), dtype=int)
    class_list = list(pred_labels)
    for j, lbl in enumerate(class_list):
        y_bin[:, j] = (np.array([str(x) for x in y_known]) == str(lbl)).astype(int)

    per_class_curves = {}
    for j, lbl in enumerate(class_list):
        try:
            p, r, thr = precision_recall_curve(y_bin[:, j], probs_known[:, j])
            ap = average_precision_score(y_bin[:, j], probs_known[:, j])
        except Exception as e:
            logger.warning(f"Failed PR for class {lbl}: {e}")
            continue
        per_class_curves[str(lbl)] = {
            "precision": p,
            "recall": r,
            "thresholds": thr,
            "ap": ap,
        }

    p_micro, r_micro, thr_micro = precision_recall_curve(
        y_bin.ravel(), probs_known.ravel()
    )
    ap_micro = average_precision_score(y_bin, probs_known, average="micro")

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    if show_per_class:
        for lbl, curve in per_class_curves.items():
            rec = curve["recall"] * (100.0 if percent else 1.0)
            prec = curve["precision"] * (100.0 if percent else 1.0)
            ax.plot(
                rec,
                prec,
                linewidth=1,
                alpha=0.25,
                label=f"{lbl} (AP={curve['ap']:.3f})",
            )

    if p_micro is not None:
        ax.plot(
            r_micro * (100.0 if percent else 1.0),
            p_micro * (100.0 if percent else 1.0),
            linewidth=2.5,
            label=f"micro (AP={ap_micro:.3f})",
            color="black",
        )

    xlabel = "Recall (%)" if percent else "Recall"
    ylabel = "Precision (%)" if percent else "Precision"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset_name.title()}: Precision–Recall")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Axis limits and ticks
    if percent:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step_x))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step_y))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    else:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_locator(
            MultipleLocator(tick_step_x / 100.0 if percent else 0.1)
        )
        ax.yaxis.set_major_locator(
            MultipleLocator(tick_step_y / 100.0 if percent else 0.1)
        )

    ax.legend(loc="lower left", fontsize="small", ncol=1)
    for ext in ("png", "svg"):
        plt.savefig(
            os.path.join(output_path, f"Precision_Recall_{dataset_name}.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()

    if p_micro is not None:
        tsv_path = os.path.join(
            output_path, f"precision_recall_micro_{dataset_name}.tsv"
        )
        with open(tsv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["recall", "precision"])
            for r, p in zip(r_micro, p_micro):
                writer.writerow([f"{r:.6f}", f"{p:.6f}"])


def plot_confidence_accuracy_curve(
    test_metrics, output_path, dataset_name="test", bins=100, figsize=(2, 1.5)
):
    if not isinstance(test_metrics, dict):
        return

    y_all = np.asarray(test_metrics.get("y_true", []))
    preds_all = np.asarray(test_metrics.get("y_pred", []))
    probs = test_metrics.get("Probabilities")
    mask_unknown = np.asarray(
        test_metrics.get("mask_unknown", np.zeros_like(y_all, dtype=bool))
    )

    if not (y_all.size and preds_all.size and probs is not None):
        logger.warning(
            "Skipping confidence-accuracy plot: missing probabilities or labels."
        )
        return

    keep = ~mask_unknown
    y_known = y_all[keep]
    probs_known = np.asarray(probs)[keep]

    if y_known.size == 0 or probs_known.size == 0:
        return

    max_probs = np.max(probs_known, axis=1)

    prob_std = np.std(max_probs)
    prob_min, prob_max = max_probs.min(), max_probs.max()
    if prob_std < 1e-4 or prob_min > 0.99:
        logger.warning(
            f"Skipping confidence plot for {dataset_name}: degenerate probabilities "
            f"(min={prob_min:.4f}, max={prob_max:.4f}, std={prob_std:.6f})."
        )
        return

    pred_labels = np.asarray(test_metrics.get("Labels (confusion_matrix_all)", []))
    if len(pred_labels) != probs_known.shape[1]:
        unique_preds = np.unique(preds_all)
        if len(unique_preds) == probs_known.shape[1]:
            pred_labels = unique_preds
        else:
            logger.warning("Cannot align probabilities to labels for confidence plot.")
            return

    top1_pred_indices = np.argmax(probs_known, axis=1)
    top1_preds = pred_labels[top1_pred_indices]
    y_known_str = np.array([str(lbl) for lbl in y_known])
    top1_preds_str = np.array([str(lbl) for lbl in top1_preds])
    correct = y_known_str == top1_preds_str

    N = correct.size
    if N == 0:
        return

    if bins is None:
        order_desc = np.argsort(-max_probs)
        correct_sorted = correct[order_desc].astype(float)
        cumsum_correct = np.cumsum(correct_sorted) 
        ks = np.arange(1, N + 1)  
        coverage = ks / float(N)  
        accuracy = cumsum_correct / ks  
        cov_plot = coverage * 100.0  
        acc_plot = accuracy * 100.0  

        if len(cov_plot) > 100:
            step = max(1, len(cov_plot) // 100)  # keep ~100 points max
            plot_idx = np.arange(0, len(cov_plot), step)
            cov_plot = cov_plot[plot_idx]
            acc_plot = acc_plot[plot_idx]

    # binned approach
    else:
        n_bins = int(bins)
        if n_bins < 1:
            n_bins = 1
        edges = np.quantile(max_probs, np.linspace(0, 1, n_bins + 1))
        bin_indices = np.digitize(max_probs, edges, right=False) - 1

        bin_acc = np.full(n_bins, np.nan)
        bin_count = np.zeros(n_bins, dtype=int)
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                bin_acc[i] = correct[mask].mean()
                bin_count[i] = mask.sum()

        # Cumulative from high to low confidence
        rev_acc = bin_acc[::-1]
        rev_count = bin_count[::-1]
        cumsum_correct = np.nancumsum(
            np.where(np.isnan(rev_acc), 0.0, rev_acc * rev_count)
        )
        cumsum_total = np.nancumsum(rev_count)

        with np.errstate(divide="ignore", invalid="ignore"):
            acc_smooth = cumsum_correct / cumsum_total
            coverage_smooth = cumsum_total / (
                cumsum_total[-1] if cumsum_total[-1] > 0 else 1
            )

        # Reverse to ascending coverag
        acc_smooth = acc_smooth[::-1]
        coverage_smooth = coverage_smooth[::-1]
        valid = np.isfinite(acc_smooth) & (coverage_smooth >= 0)

        if np.any(valid):
            cov_plot = coverage_smooth[valid] * 100.0
            acc_plot = acc_smooth[valid] * 100.0
        else:
            cov_plot = np.array([100.0])
            acc_plot = np.array([correct.mean() * 100.0 if correct.size else 0.0])

    plt.rcParams.update(
        {
            "font.size": 5,  
            "axes.labelsize": 5,  
            "xtick.labelsize": 5,  
            "ytick.labelsize": 5,  
            "legend.fontsize": 5,
            "axes.linewidth": 0.5,  
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
        }
    )
    plt.figure(figsize=figsize)
    plt.plot(cov_plot, acc_plot, linewidth=1.5)
    plt.xlabel("Coverage (%)")
    plt.ylabel("Accuracy (%)")
    plt.margins(x=0, y=0)
    max_acc = np.max(acc_plot) if acc_plot.size > 0 else 0.0
    plt.ylim(0, min(100.0, max_acc + 2.0))  

    # Add random baseline
    unique_classes = np.unique(y_known_str)
    n_classes = len(unique_classes)
    if n_classes > 1:
        random_acc_pct = 100.0 / n_classes
        plt.axhline(
            y=random_acc_pct,
            color="0.5",
            linestyle="--",
            linewidth=1,
            zorder=1,
            label="RandomClassifier",
        )

    plt.legend(frameon=False, loc="upper right")

    ax = plt.gca()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    tick_step_x = 10
    tick_step_y = 10
    ax.xaxis.set_major_locator(MultipleLocator(tick_step_x))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step_y))

    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

    for ext in ("png", "svg"):
        plt.savefig(
            os.path.join(output_path, f"Coverage_accuracy_{dataset_name}.{ext}"),
            dpi=600,
            bbox_inches="tight",
        )
    plt.close()

    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.50]
    log_rows = []

    for th in thresholds:
        mask_th = max_probs >= th
        if np.any(mask_th):
            cov = mask_th.mean()
            acc = correct[mask_th].mean()
            weighted_f1 = float(
                f1_score(
                    y_known_str[mask_th],
                    top1_preds_str[mask_th],
                    average="weighted",
                    zero_division=0,
                )
            )
        else:
            cov, acc, weighted_f1 = 0.0, np.nan, np.nan
        log_rows.append((th, cov, acc, weighted_f1))

    logger.debug(f"\n{dataset_name.title()} Confidence-Accuracy Trade-offs:")
    logger.debug("Confidence\tCoverage\tAccuracy")
    for th, cov, acc, _weighted_f1 in log_rows:
        cov_pct = cov * 100.0
        acc_pct = (acc * 100.0) if not np.isnan(acc) else float("nan")
        logger.debug(f"{th:.0%}\t\t{cov_pct:.1f}%\t\t{acc_pct:.1f}%")

    tsv_path = os.path.join(output_path, f"coverage_accuracy_{dataset_name}.tsv")
    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["confidence_threshold", "coverage", "accuracy", "weighted_f1"])
        for th, cov, acc, weighted_f1 in log_rows:
            writer.writerow(
                [
                    f"{th:.3f}",
                    f"{cov:.6f}",
                    f"{acc:.6f}" if not np.isnan(acc) else "nan",
                    f"{weighted_f1:.6f}" if not np.isnan(weighted_f1) else "nan",
                ]
            )


def plot_unknown_confusion(
    test_metrics,
    output_path,
    outfile_name="confusion_matrices_unknown",
    normalize="row",
    figsize=(6.5, 4.5),
):
    os.makedirs(output_path, exist_ok=True)
    if not isinstance(test_metrics, dict):
        return

    y_all = np.asarray(test_metrics.get("y_true", []))
    p_all = np.asarray(test_metrics.get("y_pred", []))
    mask_unknown = (
        np.asarray(test_metrics.get("mask_unknown", np.zeros_like(y_all, dtype=bool)))
        if y_all.size
        else np.array([])
    )

    if not (y_all.size and p_all.size and mask_unknown.any()):
        return

    idx = np.where(mask_unknown)[0]
    y_u, p_u = y_all[idx], p_all[idx]

    true_labels = np.unique(y_u)
    pred_labels = np.unique(p_u)

    cm = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for i, t in enumerate(true_labels):
        mask_t = y_u == t
        for j, c in enumerate(pred_labels):
            cm[i, j] = int(np.sum(mask_t & (p_u == c)))

    def get_mixture_components(true_label):
        s = str(true_label).lower()
        if s.startswith("unknown2"):
            return {"ßCAT", "ßCATD", "ßCATL", "ßCATW"}  # from mixes[0]
        elif s.startswith("unknown3"):
            return {"BCAR3", "SZsmall", "βCATHmod", "βCATins2"}  # from mixes[2]
        else:
            return set()  

    orig_row_components = [get_mixture_components(lbl) for lbl in true_labels]

    total_samples = len(y_u)
    correct_predictions = 0
    for i, true_lbl in enumerate(y_u):
        pred_lbl = p_u[i]
        comps = get_mixture_components(true_lbl)
        if str(pred_lbl) in comps:
            correct_predictions += 1
    overall_acc = correct_predictions / total_samples if total_samples > 0 else 0.0

    row_totals = cm.sum(axis=1)
    recall_per_row = []
    for i, comps in enumerate(orig_row_components):
        if row_totals[i] == 0:
            recall_per_row.append(0.0)
        else:
            tp = 0
            for j, pred in enumerate(pred_labels):
                if str(pred) in comps:
                    tp += cm[i, j]
            recall_per_row.append(tp / row_totals[i])

    col_totals = cm.sum(axis=0)
    precision_per_col = []
    for j, pred in enumerate(pred_labels):
        if col_totals[j] == 0:
            precision_per_col.append(0.0)
        else:
            tp = 0
            for i, comps in enumerate(orig_row_components):
                if str(pred) in comps:
                    tp += cm[i, j]
            precision_per_col.append(tp / col_totals[j])

    def map_unknown_label(lbl):
        s = str(lbl)
        s_low = s.lower()
        if s_low.startswith("unknown2"):
            rep = 2 if "rep" in s_low else 1
            return f"Mixture 1 - Rep {rep}"
        if s_low.startswith("unknown3"):
            return "Mixture 2"
        return s

    disp_labels = [map_unknown_label(t) for t in true_labels]
    unique_disp, inv = np.unique(disp_labels, return_inverse=True)
    if len(unique_disp) != len(disp_labels):
        agg_cm = np.zeros((len(unique_disp), cm.shape[1]), dtype=int)
        agg_recall_num = np.zeros(len(unique_disp))
        agg_recall_den = np.zeros(len(unique_disp))
        for orig_i, disp_i in enumerate(inv):
            agg_cm[disp_i] += cm[orig_i]
            agg_recall_num[disp_i] += recall_per_row[orig_i] * row_totals[orig_i]
            agg_recall_den[disp_i] += row_totals[orig_i]
        cm = agg_cm
        disp_labels = list(unique_disp)
        recall_per_row = [
            (agg_recall_num[i] / agg_recall_den[i]) if agg_recall_den[i] > 0 else 0.0
            for i in range(len(unique_disp))
        ]
        row_totals = cm.sum(axis=1)

        new_row_components = []
        for disp_i in range(len(unique_disp)):
            union_comps = set()
            for orig_i, mapped in enumerate(inv):
                if mapped == disp_i:
                    union_comps |= orig_row_components[orig_i]
            new_row_components.append(union_comps)
        row_components = new_row_components
    else:
        row_components = orig_row_components

    cm_percent = normalize_by_row(cm) * 100.0
    diag_annots = _diag_annotations(
        cm_percent, np.asarray(disp_labels), pred_labels, threshold=10
    )
    plot_matrix = normalize_by_row(cm) if normalize == "row" else cm.astype(float)

    height_factor = 0.6
    fig_w = figsize[0] if figsize is not None else 6.5
    fig_h = figsize[1] if figsize is not None else 4.5

    def _draw_mixture_rectangles_per_cell(ax, row_components_list, pred_labels_list):
    
        for i, comps in enumerate(row_components_list):
            if not comps:
                continue
            for j, p in enumerate(pred_labels_list):
                if str(p) in comps:
                    rect = Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                        clip_on=False,
                    )
                    ax.add_patch(rect)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    sns.heatmap(
        cm_percent,
        annot=diag_annots,
        fmt="",
        cmap="Blues",
        ax=ax,
        xticklabels=pred_labels,
        yticklabels=disp_labels,
        vmin=0,
        vmax=100,
    )

    _draw_mixture_rectangles_per_cell(ax, row_components, pred_labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True mixture")
    ax.set_title(f"Peptide mixtures")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(
            os.path.join(output_path, f"{outfile_name}_B_diagacc.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    for suffix, data, annot, fmt, title_base in [
        ("A_counts", plot_matrix, cm, "d", "Peptide mixtures - Counts"),
        ("C_colors", plot_matrix, False, "", "Peptide mixtures - colors only"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        sns.heatmap(
            data,
            annot=annot,
            fmt=fmt,
            cmap="Blues",
            ax=ax,
            xticklabels=pred_labels,
            yticklabels=disp_labels,
            vmin=0,
            vmax=(1 if normalize == "row" else None),
            cbar=True,
        )

        _draw_mixture_rectangles_per_cell(ax, row_components, pred_labels)

        ax.set(
            xlabel="Predicted label",
            ylabel="True mixture",
            title=f"{title_base} (Acc={overall_acc:.1%})",
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        for ext in ("png", "svg"):
            plt.savefig(
                os.path.join(output_path, f"{outfile_name}_{suffix}.{ext}"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


def order_unknown_keys(keys, max_groups=3):
    if not keys:
        return [None] * max_groups

    norm_keys = [(k, str(k).lower()) for k in keys]

    def pop_matching(pattern):
        for idx, (orig, low) in enumerate(norm_keys):
            if re.search(pattern, low):
                return norm_keys.pop(idx)[0]
        return None

    ordered = []
    ordered_candidates = [
        r"unknown2\b|unknown2[^a-z0-9]|^unknown2$|unknown2$",  # unknown2
        r"unknown2.*rep|unknown2rep|replicate.*2|rep\s*2",  # unknown2Rep
        r"unknown3\b|unknown3",  # unknown3
    ]

    for pat in ordered_candidates:
        k = pop_matching(pat)
        if k is not None:
            ordered.append(k)

    for orig, low in norm_keys:
        if re.search(r"unknown|mix|mixt|mixture|1:1|unk", low):
            ordered.append(orig)

    # remove duplicates while preserving order
    seen = set()
    final = []
    for k in ordered:
        if k is None:
            continue
        if k not in seen:
            final.append(k)
            seen.add(k)

    # pad/truncate to max_groups
    if len(final) < max_groups:
        final += [None] * (max_groups - len(final))
    else:
        final = final[:max_groups]

    return final


def plot_unknown_predicted_only(
    test_metrics,
    output_path,
    outfile_name="confusion_matrices_unknown_mixtures_onlyPredictedPortions",
    mixes=None,
    top_k=5,
    show_all=False,
    confidence_line=0.05,
    figsize=(6.7, 1.75),
):
    os.makedirs(output_path, exist_ok=True)

    mixes = mixes or [
        ["ßCAT", "ßCATD", "ßCATL", "ßCATW"],
        ["ßCAT", "ßCATD", "ßCATL", "ßCATW"],
        ["BCAR3", "SZsmall", "βCATHmod", "βCATins2"],
    ]

    if not isinstance(test_metrics, dict):
        return

    y_all = np.asarray(test_metrics.get("y_true", []))
    cand = (
        np.array(
            [
                bool(
                    re.search(
                        r"mix|mixt|mixture|1:1|unknown|unk|mix_|mix-|\d:\d",
                        str(s),
                        flags=re.IGNORECASE,
                    )
                )
                for s in y_all
            ]
        )
        if y_all.size
        else np.array([])
    )
    cand = cand | np.asarray(
        test_metrics.get("mask_unknown", np.zeros_like(y_all, dtype=bool))
    )

    groups = {}
    for i, lbl in enumerate(y_all):
        if cand[i]:
            groups.setdefault(str(lbl), []).append(i)

    if not groups and y_all.size:
        idxs = np.where(
            np.asarray(
                test_metrics.get("mask_unknown", np.zeros_like(y_all, dtype=bool))
            )
        )[0]
        if idxs.size:
            groups["unknown_all"] = idxs.tolist()

    unknown_keys = [
        k
        for k in groups.keys()
        if re.search(r"unknown|mix|mixt|mixture|1:1|unk", str(k), flags=re.IGNORECASE)
    ]
    try:
        keys = order_unknown_keys(unknown_keys, max_groups=3)
    except Exception:
        keys = unknown_keys[:3]

    panel_titles = [
        "Mixture 1 (replicate 1): " + ":".join(mixes[0]),
        "Mixture 1 (replicate 2): " + ":".join(mixes[1]),
        "Mixture 2: " + ":".join(mixes[2]),
    ]

    def _norm_label_simple(s):
        if s is None:
            return ""
        s = str(s).strip().lower()
        s = re.sub(r"[\s\-]+", "", s)
        return s

    def compute_pred_fractions(idxs):
        if len(idxs) == 0:
            return {}
        preds = np.asarray(test_metrics.get("y_pred", []))[idxs]
        preds = [str(p) for p in preds]
        unique, counts = np.unique(preds, return_counts=True)
        total = float(len(preds))
        fracs = {u: (c / total) for u, c in zip(unique, counts)}
        return fracs

    fig, axs = plt.subplots(1, 3, figsize=figsize)
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

    in_color = "#1f77b4"  # blue for components that are actually in the mixture
    out_color = "lightgrey"  # grey for components not in mixture
    out_hatch = "///"

    for ax, key, title, mix in zip(axs, keys, panel_titles, mixes):
        if key is None:
            ax.axis("off")
            continue

        idxs = groups.get(key, [])
        fracs = compute_pred_fractions(idxs)  # dict label->fraction

        if not fracs:
            ax.text(
                0.5, 0.5, "No predictions", ha="center", va="center", fontsize=font_size
            )
            ax.set_axis_off()
            continue

        sorted_items = sorted(fracs.items(), key=lambda kv: kv[1], reverse=True)
        if show_all:
            chosen = sorted_items
        else:
            chosen = sorted_items[:top_k]

        labels = [k for k, v in chosen]
        values = [v for k, v in chosen]

        x = np.arange(len(labels))
        bar_width = 0.8

        mix_norm = set([_norm_label_simple(m) for m in (mix or [])])
        colors = []
        hatches = []
        edgecolors = []
        for lbl in labels:
            lbl_norm = _norm_label_simple(lbl)
            if lbl_norm in mix_norm:
                colors.append(in_color)
                hatches.append(None)
                edgecolors.append("black")
            else:
                colors.append(out_color)
                hatches.append(out_hatch)
                edgecolors.append("gray")
        rects = []
        for xi, val, c, h, ec in zip(x, values, colors, hatches, edgecolors):
            if h:
                r = ax.bar(
                    xi,
                    val,
                    bar_width,
                    facecolor=c,
                    edgecolor=ec,
                    linewidth=0.6,
                    hatch=h,
                )
            else:
                r = ax.bar(xi, val, bar_width, facecolor=c, edgecolor=ec, linewidth=0.6)
            rects.append(r[0])

        # Annotate percent above bars
        for r in rects:
            h = r.get_height()
            ax.text(
                r.get_x() + r.get_width() / 2.0,
                h + 0.01,
                f"{h:.1%}",
                ha="center",
                va="bottom",
                fontsize=font_size,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0, 0.85)
        ax.set_title(title)
        ax.set_xlabel("Peptide", fontsize=font_size)

        # Confidence horizontal line (e.g., 10%)
        ax.axhline(confidence_line, color="gray", linestyle="--", linewidth=0.6)
        
        # Legend: in-mixture vs out-of-mixture
        legend_handles = [
            Patch(
                facecolor=in_color, edgecolor="black", label="Predicted & in mixture"
            ),
            Patch(
                facecolor=out_color,
                edgecolor="gray",
                hatch=out_hatch,
                label="Predicted but not in mixture",
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "svg"):
        plt.savefig(
            os.path.join(output_path, f"{outfile_name}_predicted_only.{ext}"), dpi=300
        )
    plt.close(fig)


def plot_unknown_mixtures(test_metrics, output_path, outfile_name="test"):
    os.makedirs(output_path, exist_ok=True)
    mixes = [
        ["ßCAT", "ßCATD", "ßCATL", "ßCATW"],
        ["ßCAT", "ßCATD", "ßCATL", "ßCATW"],
        ["BCAR3", "SZsmall", "βCATHmod", "βCATins2"],
    ]

    if not isinstance(test_metrics, dict):
        return

    y_all = np.asarray(test_metrics.get("y_true", []))
    probs = test_metrics.get("Probabilities")
    labels_for_probs = (
        np.asarray(test_metrics.get("Labels (confusion_matrix_all)"))
        if test_metrics.get("Labels (confusion_matrix_all)") is not None
        else None
    )
    cand = (
        np.array(
            [
                bool(
                    re.search(
                        r"mix|mixt|mixture|1:1|unknown|unk|mix_|mix-|\d:\d",
                        str(s),
                        flags=re.IGNORECASE,
                    )
                )
                for s in y_all
            ]
        )
        if y_all.size
        else np.array([])
    )
    cand = cand | np.asarray(
        test_metrics.get("mask_unknown", np.zeros_like(y_all, dtype=bool))
    )

    groups = {}
    for i, lbl in enumerate(y_all):
        if cand[i]:
            groups.setdefault(str(lbl), []).append(i)

    if not groups and y_all.size:
        idxs = np.where(
            np.asarray(
                test_metrics.get("mask_unknown", np.zeros_like(y_all, dtype=bool))
            )
        )[0]
        if idxs.size:
            groups["unknown_all"] = idxs.tolist()

    unknown_keys = [
        k
        for k in groups.keys()
        if re.search(r"unknown|mix|mixt|mixture|1:1|unk", str(k), flags=re.IGNORECASE)
    ]
    keys = order_unknown_keys(unknown_keys, max_groups=3)

    panel_titles = [
        "Mixture 1 (replicate 1): " + ":".join(mixes[0]) + " = 1:1:1:1",
        "Mixture 1 (replicate 2): " + ":".join(mixes[1]) + " = 1:1:1:1",
        "Mixture 2: " + ":".join(mixes[2]) + " = 1:1:1:1",
    ]

    def _norm_label_simple(s):
        if s is None:
            return ""
        s = str(s).strip().lower()
        s = re.sub(r"[\s\-]+", "", s)  # remove whitespace and hyphens
        return s

    def compute_group_pred(idxs, comps):
        comps = comps or mixes[0]
        comps_norm = [_norm_label_simple(c) for c in comps]
        comps_display = comps  # keep original labels for plotting

        if len(idxs) == 0:
            true = [1.0 / len(comps_display)] * len(comps_display)
            pred = [0.0] * len(comps_display)
            return comps_display, true, pred

        preds = np.asarray(test_metrics.get("y_pred", []))[idxs]
        preds_norm = np.array([_norm_label_simple(p) for p in preds])

        total = float(len(preds)) if len(preds) else 1.0

        counts = []
        for c_norm in comps_norm:
            counts.append(float((preds_norm == c_norm).sum()))

        counted = sum(counts)
        other_count = total - counted
        other_frac = other_count / total

        # Convert to fractions relative to total (matching confusion matrix row fractions)
        pred = [c / total for c in counts]
        true = [1.0 / len(comps_display)] * len(comps_display)

        return comps_display, true, pred  # keep signature identical to earlier

    panels = []
    for i, key in enumerate(keys[:3]):
        if key is None:
            panels.append((None, None, None))
            continue
        comps, true_p, pred_p = compute_group_pred(groups.get(key, []), mixes[i])
        panels.append((comps, true_p, pred_p))

    fig, axs = plt.subplots(1, 3, figsize=(6.7, 2.25))
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

    for ax, (comps, true_p, pred_p), title in zip(axs, panels, panel_titles):
        if comps is None:
            ax.axis("off")
            continue
        x = np.arange(len(comps))
        w = 0.35
        ax.bar(
            x - w / 2,
            true_p,
            w,
            label="Theoretical",
            facecolor="white",
            edgecolor="gray",
            linewidth=0.8,
            hatch="///",
        )
        rects = ax.bar(x + w / 2, pred_p, w, label="Predicted", edgecolor="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(comps, rotation=45, ha="right")
        ax.set_ylim(0, 0.85)
        for r in rects:
            h = r.get_height()
            ax.text(
                r.get_x() + r.get_width() / 2.0,
                h + 0.01,
                f"{h:.1%}",
                ha="center",
                va="bottom",
                fontsize=font_size,
            )
        ax.set_title(title)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=font_size)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Peptide", fontsize=font_size)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "svg"):
        plt.savefig(
            os.path.join(
                output_path, f"{outfile_name}_unknown_mixtures_proportions.{ext}"
            ),
            dpi=300,
        )
    plt.close(fig)
