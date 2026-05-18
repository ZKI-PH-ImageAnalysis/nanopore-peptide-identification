import hashlib
import logging
import math
import os
import re
import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("physchem_plot")

from utils_dtw_alignment import sample_per_class, select_top_nonredundant_features
from utils_classification import extract_interpretable_features
from utils_interpretability import shorten_feature_name

def get_peptide_sequences():
    return {
        "ßCAT": "YLDSGIHSGAC",
        "ßCATD": "YLDSDIHSGAC",
        "ßCATW":"YLDSWIHSGAC",
        "ßCATGG":"YLDSGGHSGAC",
        "ßCATL":"YLDSLIHSGAC",
        "ßCATWW":"YLDSWWHSGAC",
        "ßCATWWW":"YLDSWWWHSGAC",
        "BCAR3":"IMDRTPEKLC",
        "ßCAT20":"YLDSGIHSGACKTGKHGEGC",
        "ßCAT30":"YLDSGIHSGACKTGKHGEGCEAVKLQRDLC",
        "ßCAT35":"YLDSGIHSGACKTGKHGEGCEAVKLQRDLGCDLQH",
        "CHGnegG":"YEYEYEGEYEYEC",
        "CHGposD":"RKHGRKWHDKRKC",
        "CHGposL":"RKHGRKWHLKRKC",
        "SZsmall":"GSGAGSSGGSIGGRC",
        "SZlarge":"GFLFPEHTYFFRC",
        "βCATHphil":"YLDSGIHSGAKDKKC",
        "βCATHmod":"YLDSGIHSGALKGQC",
        "βCATHphob":"YLDSGIHSGAKKKAC",
        "βCATins1":"VVVVGYLDSGIHSGAC",
        "βCATins2":"YLDSGIHSGAGVVVVC",
    }


KD = {
 'A': 1.8,'R': -4.5,'N': -3.5,'D': -3.5,'C': 2.5,
 'Q': -3.5,'E': -3.5,'G': -0.4,'H': -3.2,'I': 4.5,
 'L': 3.8,'K': -3.9,'M': 1.9,'F': 2.8,'P': -1.6,
 'S': -0.8,'T': -0.7,'W': -0.9,'Y': -1.3,'V': 4.2
}

AA_MASS = {
 'A':89.09,'R':174.20,'N':132.12,'D':133.10,'C':121.16,
 'Q':146.15,'E':147.13,'G':75.07,'H':155.16,'I':131.17,
 'L':131.17,'K':146.19,'M':149.21,'F':165.19,'P':115.13,
 'S':105.09,'T':119.12,'W':204.23,'Y':181.19,'V':117.15
}

PKA_SIDE = {
    'D': 3.9, 'E': 4.1, 'C': 8.3, 'Y': 10.1, 'H': 6.0,
    'K': 10.5, 'R': 12.5
}
PKA_NTERM = 9.69
PKA_CTERM = 2.34

def mean_hydrophobicity(seq):
    vals = [KD.get(res, 0.0) for res in seq]
    return float(np.mean(vals)) if vals else np.nan

def count_fraction(seq, residues):
    c = sum(seq.count(r) for r in residues)
    return c, c/len(seq) if len(seq)>0 else (0, np.nan)

def molecular_weight(seq):
    return sum(AA_MASS.get(res, 0.0) for res in seq)

def net_charge_at_pH(seq, pH=7.0):
    pos = 1.0 / (1.0 + 10**(pH - PKA_NTERM))
    neg = -1.0 / (1.0 + 10**(PKA_CTERM - pH))
    charge = pos + neg
    for aa in seq:
        if aa in ('K','R','H'):
            pka = PKA_SIDE.get(aa)
            if pka:
                frac = 1.0 / (1.0 + 10**(pH - pka))
                charge += frac
        if aa in ('D','E','C','Y'):
            pka = PKA_SIDE.get(aa)
            if pka:
                frac = -1.0 / (1.0 + 10**(pka - pH))
                charge += frac
    return float(charge)

def estimate_pI(seq, ph_low=0.0, ph_high=14.0, tol=1e-3):
    a, b = ph_low, ph_high
    fa = net_charge_at_pH(seq, a)
    fb = net_charge_at_pH(seq, b)
    if abs(fa) < tol: return a
    if abs(fb) < tol: return b
    if fa * fb > 0:
        return np.nan
    for _ in range(60):
        m = (a + b) / 2.0
        fm = net_charge_at_pH(seq, m)
        if abs(fm) < tol:
            return float(round(m, 3))
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return float(round((a + b) / 2.0, 3))

def compute_peptide_properties(seqs_dict, ph=7.0):
    rows = []
    for name, seq in seqs_dict.items():
        seq_clean = re.sub(r'[^A-Za-z]', '', seq).upper()
        length = len(seq_clean)
        mean_kd = mean_hydrophobicity(seq_clean)
        mw = molecular_weight(seq_clean)
        arom_count, arom_fraction = count_fraction(seq_clean, ['F','W','Y'])
        pos_count, pos_fraction = count_fraction(seq_clean, ['K','R','H'])
        neg_count, neg_fraction = count_fraction(seq_clean, ['D','E'])
        frac_hydrophobic = sum(1 for aa in seq_clean if KD.get(aa,0) > 0) / length if length>0 else np.nan
        net_charge = net_charge_at_pH(seq_clean, pH=ph)
        pI = estimate_pI(seq_clean)
        rows.append({
            'peptide': name,
            'sequence': seq,
            'length': length,
            'mean_hydrophobicity': mean_kd,
            'mol_weight': mw,
            'aromatic_count': arom_count,
            'aromatic_fraction': arom_fraction,
            'pos_count': pos_count,
            'pos_fraction': pos_fraction,
            'neg_count': neg_count,
            'neg_fraction': neg_fraction,
            'frac_hydrophobic': frac_hydrophobic,
            f'net_charge_pH{ph}': net_charge,
            'pI': pI
        })
    df = pd.DataFrame(rows).set_index('peptide')
    return df

def build_metrics_df_from_scenarios(results_all_scenarios, dataset_name='Validation'):
    def _append_rows_from_report(rows_out, report_dict, scenario, model_name):
        if not isinstance(report_dict, dict):
            return 0
        n_added = 0
        for cls_label, vals in report_dict.items():
            # skip summary rows typically not per-class
            if cls_label in ('accuracy', 'macro avg', 'weighted avg', 'micro avg'):
                continue
            if not isinstance(vals, dict):
                continue
            precision = vals.get('precision', np.nan)
            recall = vals.get('recall', np.nan)
            f1 = vals.get('f1-score', vals.get('f1', np.nan))
            support = vals.get('support', 0)
            try:
                precision = float(precision)
            except Exception:
                precision = np.nan
            try:
                recall = float(recall)
            except Exception:
                recall = np.nan
            try:
                f1 = float(f1)
            except Exception:
                f1 = np.nan
            try:
                support = int(support)
            except Exception:
                support = np.nan
            rows_out.append({
                'scenario': scenario,
                'model': model_name,
                'peptide': cls_label,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            })
            n_added += 1
        return n_added

    def _append_rows_from_confusion(rows_out, metrics_dict, scenario, model_name):
        if not isinstance(metrics_dict, dict):
            return 0

        cm = metrics_dict.get('Confusion Matrix (w/o unknown)')
        labels = metrics_dict.get('Labels (confusion_matrix_known)')
        if cm is None or labels is None:
            cm = metrics_dict.get('Confusion Matrix (all)')
            labels = metrics_dict.get('Labels (confusion_matrix_all)')
        if cm is None or labels is None:
            return 0

        cm = np.asarray(cm)
        labels = list(labels)
        if cm.ndim != 2 or cm.shape[0] == 0 or cm.shape[0] != cm.shape[1] or cm.shape[0] != len(labels):
            return 0

        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0).astype(float) - tp
        fn = cm.sum(axis=1).astype(float) - tp
        support = cm.sum(axis=1).astype(float)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2.0 * precision * recall, precision + recall,
                       out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)

        n_added = 0
        for i, cls_label in enumerate(labels):
            rows_out.append({
                'scenario': scenario,
                'model': model_name,
                'peptide': cls_label,
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            })
            n_added += 1
        return n_added

    rows = []
    for scenario, scenario_results in results_all_scenarios.items():
        for mname, info in scenario_results.items():
            metrics = info.get('val') if dataset_name.lower().startswith('val') else info.get('test')
            if not metrics:
                continue
            report = {}
            for key in ("Classification Report (w/o unknown)", "Classification Report", "classification_report", "classification report"):
                if key in metrics and isinstance(metrics[key], dict):
                    report = metrics[key]
                    break
            if not report and isinstance(metrics, dict) and any(isinstance(v, dict) for v in metrics.values()):
                for k, v in metrics.items():
                    if isinstance(v, dict) and ('precision' in next(iter(v.keys())) if isinstance(v, dict) and len(v) else False):
                        report = v
                        break
            if not isinstance(report, dict):
                report = metrics if isinstance(metrics, dict) else {}
            n_added = _append_rows_from_report(rows, report, scenario, mname)
            if n_added == 0:
                _append_rows_from_confusion(rows, metrics, scenario, mname)
    if not rows:
        return pd.DataFrame(columns=['scenario','model','peptide','precision','recall','f1','support'])
    df = pd.DataFrame(rows)
    return df

def aggregate_and_merge(metrics_df, properties_df, agg='median'):
    if metrics_df.empty:
        return pd.DataFrame()
    agg_funcs = {'precision': 'median', 'recall': 'median', 'f1': 'median', 'support': 'median'}
    if 'peptide' not in metrics_df.columns:
        raise ValueError("metrics_df must have a 'peptide' column")
    agg_df = metrics_df.groupby('peptide').agg(agg_funcs).reset_index()
    merged = agg_df.set_index('peptide').join(properties_df, how='left')
    merged = merged.reset_index()
    return merged



def normalize_metric_columns(metrics_df):
    """
    Input: metrics_df with a 'peptide' column (or index).
    Output: normalized df with columns ['peptide','precision','recall','f1','support'] (some may be NaN)
    and a mapping dict of found columns.
    """
    df = metrics_df.copy()
    if df.index.name == 'peptide' and 'peptide' not in df.columns:
        df = df.reset_index()
    colmap = {}
    lower_to_col = {c.lower(): c for c in df.columns}
    wants = ['precision','recall','f1','support']
    for want in wants:
        if want in lower_to_col:
            colmap[want] = lower_to_col[want]
            continue
        if want == 'f1' and 'f1-score' in lower_to_col:
            colmap[want] = lower_to_col['f1-score']
            continue
        matches = [orig for low, orig in lower_to_col.items() if want in low]
        if matches:
            colmap[want] = matches[0]
    out = pd.DataFrame()
    out['peptide'] = df['peptide'] if 'peptide' in df.columns else df.index
    for want in wants:
        actual = colmap.get(want)
        if actual is not None and actual in df.columns:
            out[want] = pd.to_numeric(df[actual], errors='coerce')
        else:
            out[want] = np.nan
    return out, colmap

def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n+1)
    thresholds = (ranks / n) * alpha
    sorted_p = p[order]
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros_like(p, dtype=bool)
    max_i = np.where(below)[0].max()
    cutoff = sorted_p[max_i]
    return p <= cutoff

def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    x2 = x[mask]
    y2 = y[mask]
    if np.nanstd(x2) == 0 or np.nanstd(y2) == 0:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = stats.spearmanr(x2, y2)
    try:
        return float(rho), float(p)
    except Exception:
        return np.nan, np.nan

def spearman_correlations_matrix(merged_df, metric_names=None, prop_names=None):
    if metric_names is None:
        metric_names = ['precision','recall','f1','support']
    if prop_names is None:
        prop_names = [c for c in merged_df.columns if c not in ('peptide','sequence') and merged_df[c].dtype.kind in 'fiu']
        prop_names = [p for p in prop_names if p not in metric_names]
    rhos = pd.DataFrame(index=metric_names, columns=prop_names, dtype=float)
    pvals = pd.DataFrame(index=metric_names, columns=prop_names, dtype=float)
    for m in metric_names:
        for p in prop_names:
            if m not in merged_df.columns or p not in merged_df.columns:
                rhos.loc[m, p] = np.nan
                pvals.loc[m, p] = np.nan
                continue
            rho, pval = safe_spearman(merged_df[m].values, merged_df[p].values)
            rhos.loc[m, p] = rho
            pvals.loc[m, p] = pval
    return rhos, pvals


def get_colors_for_classes(class_labels):
    unique_labels = []
    for c in class_labels:
        if c not in unique_labels:
            unique_labels.append(c)

    # base palette (colorblind 10)
    colorblind10 = sns.color_palette("colorblind", 10)
    colorblind_hex = [mcolors.to_hex(c) for c in colorblind10]

    fixed_assignments = {
        'ßCAT':     0,
        'ßCATD':    1,
        'ßCATL':    2,
        'ßCATW':    3,
        'ßCATWW':   4,
        'ßCATGG':   5,
        'ßCATWWW':  6,
        'BCAR3':    7,
        'ßCAT20':   8,
        'ßCAT30':   9,
    }

    fixed_colors = {}
    for cls, idx in fixed_assignments.items():
        if 0 <= idx < len(colorblind_hex):
            fixed_colors[cls] = colorblind_hex[idx]

    remaining = [c for c in unique_labels if c not in fixed_colors]
    remaining_sorted = sorted(remaining, key=lambda x: hashlib.md5(x.encode()).hexdigest())

    n_extra = len(remaining_sorted)
    extra_colors = []
    used_indices = set([v for k, v in fixed_assignments.items() if k in fixed_colors])
    available_indices = [i for i in range(len(colorblind_hex)) if i not in used_indices]
    if n_extra <= len(available_indices):
        extra_colors = [colorblind_hex[i] for i in available_indices[:n_extra]]
    else:

        extra_colors = [colorblind_hex[i] for i in available_indices]
        need_more = n_extra - len(extra_colors)
        extra_colors += [mcolors.to_hex(c) for c in sns.husl_palette(need_more, s=0.9, l=0.65)]

    extra_map = dict(zip(remaining_sorted, extra_colors))
    color_map = {**fixed_colors, **extra_map}

    # ensure every unique label has a color (fallback to matplotlib default cycle)
    default_cycle = [mcolors.to_hex(c) for c in mpl.rcParams['axes.prop_cycle'].by_key()['color']]
    idx_def = 0
    for lbl in unique_labels:
        if lbl not in color_map:
            color_map[lbl] = default_cycle[idx_def % len(default_cycle)]
            idx_def += 1

    return color_map, unique_labels

def plot_physicochemical_relationships(merged_df,
                                      output_dir,
                                      metric_list=['precision','recall','f1','support'],
                                      property_list=None,
                                      ph=7.0,
                                      dpi=600,
                                      regress=False,
                                      show_point_labels=False,
                                      log_metrics=None):
    """
    Scatter-grid + heatmap.
    - log_metrics: list of metric names that should be shown on log10(x+1) scale (default ['total_count'])
    """
    os.makedirs(output_dir, exist_ok=True)

    if log_metrics is None:
        log_metrics = ['total_count']

    if property_list is None:
        property_list = ['length', 'mean_hydrophobicity', f'net_charge_pH{ph}', 'pI', 'mol_weight']

    metric_list_avail = [m for m in metric_list if m in merged_df.columns]
    prop_list_avail = [p for p in property_list if p in merged_df.columns]

    if len(metric_list_avail) == 0:
        raise RuntimeError(f"No metric columns found in merged_df. Available: {list(merged_df.columns)}")
    if len(prop_list_avail) == 0:
        raise RuntimeError(f"No property columns found in merged_df. Available: {list(merged_df.columns)}")

    all_peptides = list(merged_df['peptide'].astype(str).unique())
    color_map, ordered_labels = get_colors_for_classes(all_peptides)
    legend_labels = ordered_labels

    n_metrics = len(metric_list_avail)
    n_props = len(prop_list_avail)

    width_in = 6.7
    fig_h = max(2.0, 1.2 * n_props)

    small_font = 5
    plt.rcParams.update({
        'font.size': small_font,
        'axes.titlesize': small_font,
        'axes.labelsize': small_font,
        'xtick.labelsize': small_font,
        'ytick.labelsize': small_font,
        'legend.fontsize': small_font,
        'figure.titlesize': small_font,
    })

    fig, axes = plt.subplots(n_props, n_metrics, figsize=(width_in, fig_h), squeeze=False,
                             sharex=False, sharey=False)

    scatter_s = 12
    logger.info(f"Plotting grid with properties (rows): {prop_list_avail} and metrics (cols): {metric_list_avail}")

    def _ols_line_and_ci(x, y, xs=None, confidence=0.95):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]; y = y[mask]
        n = x.size
        if n < 3 or np.nanstd(x) == 0:
            return None
        slope, intercept = np.polyfit(x, y, 1)
        if xs is None:
            xs = np.linspace(np.min(x), np.max(x), 100)
        else:
            xs = np.asarray(xs, dtype=float)
        y_pred = slope * xs + intercept
        y_fit = slope * x + intercept
        residuals = y - y_fit
        dof = n - 2
        if dof <= 0:
            return xs, y_pred, np.full_like(y_pred, np.nan), np.full_like(y_pred, np.nan)
        s_err = np.sqrt(np.sum(residuals**2) / dof)
        x_mean = np.mean(x)
        ssx = np.sum((x - x_mean)**2)
        if ssx == 0:
            return xs, y_pred, np.full_like(y_pred, np.nan), np.full_like(y_pred, np.nan)
        tval = stats.t.ppf((1 + confidence) / 2., dof)
        se_mean = np.sqrt((1.0 / n) + ((xs - x_mean)**2) / ssx)
        delta = tval * s_err * se_mean
        lower = y_pred - delta
        upper = y_pred + delta
        return xs, y_pred, lower, upper

    for i, p in enumerate(prop_list_avail):
        for j, m in enumerate(metric_list_avail):
            ax = axes[i][j]

            if p not in merged_df.columns or m not in merged_df.columns:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', fontsize=small_font)
                ax.set_xlabel(p)
                ax.set_ylabel(m)
                continue

            sub = merged_df[[p, m, 'peptide']].dropna()
            if sub.empty:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', fontsize=small_font)
                ax.set_xlabel(p)
                ax.set_ylabel(m)
                continue

            x_vals = sub[p].to_numpy(dtype=float)
            y_vals_raw = sub[m].to_numpy(dtype=float)

            if m in (log_metrics or []):
                y_vals = y_vals_raw
            else:
                y_vals = y_vals_raw

            labels = sub['peptide'].astype(str).to_numpy()
            colors = [color_map.get(lbl, '#333333') for lbl in labels]

            ax.scatter(x_vals, y_vals, s=scatter_s, c=colors, alpha=0.9, edgecolors='none')

            if regress and x_vals.size >= 3 and np.nanstd(x_vals) > 0 and np.nanstd(y_vals) > 0:
                try:
                    res = _ols_line_and_ci(x_vals, y_vals)
                    if res is not None:
                        xs, y_pred, lower, upper = res
                        ax.plot(xs, y_pred, linewidth=0.6, alpha=0.9, color='black')
                        if lower is not None and not np.all(np.isnan(lower)):
                            ax.fill_between(xs, lower, upper, alpha=0.22, color='black')
                except Exception:
                    pass

            rho, pval = safe_spearman(x_vals, y_vals)
            rho_str = f"{rho:.2f}" if not np.isnan(rho) else "nan"
            p_str = f"{pval:.2g}" if not np.isnan(pval) else "nan"
            n_pts = (~np.isnan(x_vals) & ~np.isnan(y_vals)).sum()

            annot_text = f""
            if regress:
                annot_text = f"Spearman ρ = {rho_str}\n p = {p_str}\n n = {n_pts}"
            ax.annotate(annot_text,
                        xy=(0.02, 0.95), xycoords='axes fraction', va='top', fontsize=small_font)
            

            ax.set_title(f"{m} vs {p}", fontsize=small_font+0.5, pad=3, fontweight="bold")

            if show_point_labels and x_vals.size <= 200:
                for xi, yi, lab in zip(x_vals, y_vals, labels):
                    ax.text(xi, yi, lab, fontsize=max(4, small_font-1), alpha=0.9)

            ax.set_xlabel(p, fontsize=small_font)
            ylabel = m
            ax.set_ylabel(ylabel, fontsize=small_font)
            ax.tick_params(labelsize=small_font)

            ax.grid(False)

    if not show_point_labels:
        handles = []
        legend_labels_present = []
        for lbl in legend_labels:
            if lbl not in all_peptides:
                continue
            color = color_map.get(lbl, '#333333')
            h = mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5, linestyle='None')
            handles.append(h)
            legend_labels_present.append(lbl)
        n_items = len(legend_labels_present)
        ncol = min(8, max(1, int(n_items**0.5 * 2)))
        legend_space = 0.18
        plt.subplots_adjust(bottom=legend_space)
        fig.legend(handles, legend_labels_present, loc='lower center',
                   ncol=ncol, frameon=False, fontsize=max(4, small_font-0.5),
                   bbox_to_anchor=(0.5, -0.05))
        
    hspace = max(0.2, 0.08 * n_props)
    plt.subplots_adjust(bottom=legend_space, hspace=hspace)
    

    plt.tight_layout()
    out_png = os.path.join(output_dir, "supp_scatter_physchem_vs_metrics.png")
    out_svg = os.path.join(output_dir, "supp_scatter_physchem_vs_metrics.svg")
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.savefig(out_svg, dpi=300, bbox_inches='tight')
    plt.close(fig)

    
    small_font = 7
    plt.rcParams.update({
        'font.size': small_font,
        'axes.titlesize': small_font-2,
        'axes.labelsize': small_font-2,
        'xtick.labelsize': small_font-2,
        'ytick.labelsize': small_font-2,
        'legend.fontsize': small_font-2,
        'figure.titlesize': small_font-2,
    })
    merged_for_corr = merged_df.copy()
    for lm in (log_metrics or []):
        if lm in merged_for_corr.columns:
            merged_for_corr[lm] = np.log10(merged_for_corr[lm].fillna(0) + 1.0)

    rhos, pvals = spearman_correlations_matrix(merged_for_corr, metric_names=metric_list_avail, prop_names=prop_list_avail)
    pflat = pvals.values.flatten()
    idx_valid = ~np.isnan(pflat)
    sig_flat = np.zeros_like(pflat, dtype=bool)
    if idx_valid.sum() > 0:
        sig_flat[idx_valid] = benjamini_hochberg(pflat[idx_valid], alpha=0.05)
    sig_matrix = sig_flat.reshape(pvals.shape)

    heat_h = max(1.6, 0.5 + 0.6 * n_metrics)
    plt.figure(figsize=(width_in, heat_h))
    ax = sns.heatmap(rhos.astype(float), annot=True, fmt=".2f", cmap='vlag', center=0,
                     square=False, cbar_kws={'label': 'Spearman ρ'}, annot_kws={'fontsize': small_font})
    for ii in range(rhos.shape[0]):
        for jj in range(rhos.shape[1]):
            if sig_matrix.flatten()[ii * rhos.shape[1] + jj]:
                ax.text(jj + 0.5, ii + 0.5, '*', color='black', ha='center', va='center', fontsize=small_font + 1)

    ax.set_xlabel("Metrics", fontsize=small_font)
    ax.set_ylabel("Properties", fontsize=small_font)
    plt.title("Spearman correlations (stars = BH q<0.05)", fontsize=small_font)
    out_png2 = os.path.join(output_dir, "supp_corr_heatmap.png")
    out_svg2 = os.path.join(output_dir, "supp_corr_heatmap.svg")
    plt.tight_layout()
    plt.savefig(out_png2, dpi=dpi, bbox_inches='tight')
    plt.savefig(out_svg2, dpi=600, bbox_inches='tight')
    plt.close()

    merged_df.to_csv(os.path.join(output_dir, "supp_peptide_properties_and_metrics.csv"), index=False)
    rhos.to_csv(os.path.join(output_dir, "supp_spearman_rhos.csv"))
    pvals.to_csv(os.path.join(output_dir, "supp_spearman_pvals.csv"))


def create_physchem_vs_perf_supplement(
        results_all_scenarios,
        output_dir,
        dataset_name='Validation',
        ph=7.0,
        counts=None,
        counts_key='trainval'):
    os.makedirs(output_dir, exist_ok=True)
    seqs = get_peptide_sequences()
    props = compute_peptide_properties(seqs, ph=ph)

    metrics_df = build_metrics_df_from_scenarios(results_all_scenarios, dataset_name=dataset_name)

    def _norm_model_name(x):
        return re.sub(r'[^a-z0-9]+', '', str(x).lower())

    if not metrics_df.empty and "model" in metrics_df.columns:
        model_names = metrics_df["model"].astype(str)
        model_norm = model_names.map(_norm_model_name)
        feature_mask = (
            model_norm.str.contains("featurelgbm", regex=False)
            | ((model_norm.str.contains("feature", regex=False)) & (model_norm.str.contains("lgbm", regex=False)))
            | (model_norm == "lgbm")
        )
        if feature_mask.any():
            metrics_df = metrics_df[feature_mask].copy()
        elif model_norm.nunique() == 1:
            logger.info(
                "Single model present in metrics (%s); using it for physchem plots.",
                model_names.iloc[0],
            )
        else:
            logger.warning(
                "No featureLGBM rows found in metrics for dataset '%s'; using all available models for physchem plots.",
                dataset_name,
            )

    metrics_df.to_csv(os.path.join(output_dir, "diagnostic_raw_metrics_df.csv"), index=False)

    if metrics_df.empty:
        raise RuntimeError(
            "No per-class metrics found after model selection. "
            "Check results_all_scenarios, dataset_name, and model outputs."
        )

    normalized_metrics_df, colmap = normalize_metric_columns(metrics_df)
    normalized_metrics_df.to_csv(os.path.join(output_dir, "diagnostic_normalized_metrics_df.csv"), index=False)
    logger.info(f"Column mapping found for metrics: {colmap}")

    merged = aggregate_and_merge(normalized_metrics_df, props)
    merged.to_csv(os.path.join(output_dir, "diagnostic_merged_initial.csv"), index=False)

    if counts is not None and counts_key in counts:
        counts_map = counts[counts_key]
        merged['total_count'] = merged['peptide'].map(lambda x: int(counts_map.get(x, 0)))
    else:
        if 'support' in merged.columns:
            merged['total_count'] = merged['support'].fillna(0).astype(int)
        else:
            merged['total_count'] = 0

    merged.to_csv(os.path.join(output_dir, "diagnostic_merged_with_counts.csv"), index=False)

    usable_metrics = [m for m in ['precision', 'recall', 'total_count'] if m in merged.columns and merged[m].notna().sum() >= 3]
    if not usable_metrics:
        raise RuntimeError("Not enough non-NaN metric values across peptides (need >=3) to run correlations/plots.")

    plot_physicochemical_relationships(merged,
                                      output_dir,
                                      metric_list=usable_metrics,
                                      property_list=None,
                                      ph=ph,
                                      dpi=600,
                                      regress=False,
                                      show_point_labels=False,
                                      log_metrics=['total_count'])

    merged_for_corr = merged.copy()
    if 'total_count' in merged_for_corr.columns:
        merged_for_corr['total_count'] = np.log10(merged_for_corr['total_count'].fillna(0) + 1.0)

    rhos, pvals = spearman_correlations_matrix(merged_for_corr, metric_names=usable_metrics, prop_names=None)
    rhos.to_csv(os.path.join(output_dir, "supp_spearman_rhos_fixed.csv"))
    pvals.to_csv(os.path.join(output_dir, "supp_spearman_pvals_fixed.csv"))

    logger.info("create_physchem_vs_perf_supplement finished successfully.")
    return merged, rhos, pvals


def create_feature_vs_physchem_supplement(
        X=None,
        y=None,
        features_df=None,
        labels=None,
        n_features=5,
        outdir="feature_physchem_figures",
        ph=7.0,
        regress=False,
        show_point_labels=False,
        log_features=None,
        figsize_width_in=6.7,
        per_class=10000,
        seed=385):
    
    os.makedirs(outdir, exist_ok=True)
    log_features = log_features or []

    unique_classes = np.unique(y)
    classes = unique_classes
    sampled_idxs = sample_per_class(y, classes, per_class=per_class, seed=seed)

    X = [X[i] for i in sampled_idxs]
    y = [y[i] for i in sampled_idxs]

    if features_df is None:
        if X is None or y is None:
            raise ValueError("Either features_df or (X and y) must be provided")
        features_df, feature_names = extract_interpretable_features(X, use_catch22=True)
        labels = list(y)
    else:
        if labels is None:
            raise ValueError("If you pass features_df you must also pass labels (sample-level class labels).")
    
    features_df.columns = [shorten_feature_name(f) for f in features_df.columns]

    if len(labels) != len(features_df):
        raise ValueError("labels length does not match features_df rows")

    top_features, features_topdf = select_top_nonredundant_features(
        features_df, labels, n_features=n_features, corr_threshold=0.95
    )

    if isinstance(top_features, (list, tuple)):
        top_features = list(top_features)
    else:
        top_features = list(top_features)

    top_features = [f for f in top_features if f in features_df.columns]
    if len(top_features) == 0:
        raise RuntimeError("No top features found in features_df")
    
    

    # aggregate per-peptide (median, std, count)
    df_features = features_df.copy()
    df_features['peptide'] = labels
    agg_median = df_features.groupby('peptide')[top_features].median().rename(columns=lambda x: x + "_median")
    agg_std = df_features.groupby('peptide')[top_features].std().rename(columns=lambda x: x + "_std")
    agg_count = df_features.groupby('peptide').size().rename("n_samples")
    agg_df = pd.concat([agg_median, agg_std, agg_count], axis=1).reset_index().rename(columns={'index':'peptide'})

    # compute peptide properties and merge
    seqs = get_peptide_sequences()
    props = compute_peptide_properties(seqs, ph=ph).reset_index().rename(columns={'index':'peptide'})
    merged = agg_df.merge(props, on='peptide', how='left')

    merged.to_csv(os.path.join(outdir, "supp_feature_values_per_peptide.csv"), index=False)

    # plotting helpers & layout params
    small_font = 4 
    plt.rcParams.update({
        'font.size': small_font,
        'axes.titlesize': small_font,
        'axes.labelsize': small_font,
        'xtick.labelsize': small_font,
        'ytick.labelsize': small_font,
        'legend.fontsize': small_font,
        'figure.titlesize': small_font,
    })

    color_map, ordered_labels = get_colors_for_classes(list(merged['peptide'].astype(str).values))
    # keep deterministic order for legend
    legend_labels = ordered_labels

    # Scatter grid: properties (rows) x top_features (cols)
    property_list = ['length', 'mean_hydrophobicity', f'net_charge_pH{ph}', 'pI', 'mol_weight']
    prop_list_avail = [p for p in property_list if p in merged.columns]
    feat_list_avail = top_features

    n_props = len(prop_list_avail)
    n_feats = len(feat_list_avail)
    if n_props == 0:
        raise RuntimeError("No properties available in merged data to plot")

    width_in = float(figsize_width_in)
    height_in = max(1.0, 0.9 * n_props)
    fig, axes = plt.subplots(n_props, n_feats, figsize=(width_in, height_in), squeeze=False)
    scatter_s = 12

    for i, p in enumerate(prop_list_avail):
        for j, f in enumerate(feat_list_avail):
            ax = axes[i][j]
            # use median feature value per peptide
            x = merged[p].to_numpy(dtype=float)
            y = merged[f + "_median"].to_numpy(dtype=float)
            labels_pep = merged['peptide'].astype(str).to_numpy()
            colors = [color_map.get(lbl, '#333333') for lbl in labels_pep]

            y_plot = np.log10(y + 1.0) if f in log_features else y

            ax.scatter(x, y_plot, s=scatter_s, c=colors, alpha=0.9, edgecolors='none')
            if regress and (~np.isnan(x) & ~np.isnan(y_plot)).sum() >= 3 and np.nanstd(x) > 0 and np.nanstd(y_plot) > 0:
                try:
                    slope, intercept = np.polyfit(x[~np.isnan(x) & ~np.isnan(y_plot)], y_plot[~np.isnan(x) & ~np.isnan(y_plot)], 1)
                    xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                    ys = slope * xs + intercept
                    ax.plot(xs, ys, linewidth=0.6)
                    # simple CI by bootstrap-ish approx: use previous _ols_line_and_ci if you prefer
                except Exception:
                    pass

            rho, pval = safe_spearman(x, y_plot)
            rho_s = f"{rho:.2f}" if not np.isnan(rho) else "nan"
            p_s = f"{pval:.2g}" if not np.isnan(pval) else "nan"
            n_pts = int((~np.isnan(x) & ~np.isnan(y_plot)).sum())
            
            annot_text = f"ρ={rho_s}\n p={p_s}\n n={n_pts}"
            annot_text = ""
            ax.annotate(annot_text, xy=(0.02, 0.95), xycoords='axes fraction', va='top', fontsize=small_font)

            if len(f) > 30:
                f = f[:30]
            ax.set_title(f"{f} vs {p}", fontsize=small_font, pad=3, fontweight="bold")
            ax.set_xlabel(p, fontsize=small_font)
            xlabel = f + (" (log10+1)" if f in log_features else "")
            ax.tick_params(labelsize=small_font)

            if show_point_labels:
                for xi, yi, lab in zip(x, y_plot, labels_pep):
                    ax.text(xi, yi, lab, fontsize=max(4, small_font-1), alpha=0.9)

    if not show_point_labels:
        handles = []
        labels_for_legend = []
        for lbl in legend_labels:
            if lbl not in merged['peptide'].values:
                continue
            c = color_map.get(lbl, '#333333')
            h = mpl.lines.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=5, linestyle='None')
            handles.append(h); labels_for_legend.append(lbl)
        n_items = len(labels_for_legend)
        ncol = min(8, max(1, int(math.sqrt(n_items)*2)))
        legend_space = 0.18
        plt.subplots_adjust(bottom=legend_space, hspace=max(0.12, 0.06 * n_props))
        fig.legend(handles, labels_for_legend, loc='lower center', ncol=ncol, frameon=False, fontsize=max(4, small_font-0.5), bbox_to_anchor=(0.5, -0.05))
    else:
        plt.subplots_adjust(hspace=max(0.12, 0.06 * n_props))

    plt.tight_layout()
    scatter_png = os.path.join(outdir, "supp_features_vs_physchem_scattergrid.png")
    scatter_svg = os.path.join(outdir, "supp_features_vs_physchem_scattergrid.svg")
    plt.savefig(scatter_png, dpi=600, bbox_inches='tight')
    plt.savefig(scatter_svg, dpi=300, bbox_inches='tight')
    plt.close(fig)

    rows = len(feat_list_avail)
    h_err = max(6, 0.6 * rows)
    fig2, axes2 = plt.subplots(rows, 1, figsize=(width_in, h_err), squeeze=False)
    fig2.subplots_adjust(hspace=0.6)  
    for idx, f in enumerate(feat_list_avail):
        ax = axes2[idx, 0]
        med = merged[f + "_median"].to_numpy(dtype=float)
        std = merged[f + "_std"].to_numpy(dtype=float)
        labels_pep = merged['peptide'].astype(str).to_numpy()
        colors = [color_map.get(lbl, '#333333') for lbl in labels_pep]
        x_pos = np.arange(len(labels_pep))
        ax.errorbar(x_pos, med, yerr=std, fmt='o', ecolor='gray', elinewidth=0.6, capsize=2, markersize=4, markerfacecolor='none', markeredgecolor='black')
        ax.scatter(x_pos, med, c=colors, s=12, edgecolors='none')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_pep, rotation=90, fontsize=max(4, small_font-1))
        ax.set_title(f"Per-peptide median ± std | {f}", fontsize=small_font, pad=2)
        ax.tick_params(labelsize=small_font)
        ax.grid(False)
    plt.tight_layout()
    err_png = os.path.join(outdir, "supp_features_error_panel.png")
    err_svg = os.path.join(outdir, "supp_features_error_panel.svg")
    plt.savefig(err_png, dpi=600, bbox_inches='tight')
    plt.savefig(err_svg, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    merged_for_corr = merged.copy()
    for f in feat_list_avail:
        col = f + "_median"
        if f in log_features and col in merged_for_corr.columns:
            merged_for_corr[col] = np.log10(merged_for_corr[col].fillna(0) + 1.0)

    feat_cols_for_corr = [f + "_median" for f in feat_list_avail if f + "_median" in merged_for_corr.columns]
    prop_cols_for_corr = [p for p in prop_list_avail if p in merged_for_corr.columns]

    rhos, pvals = spearman_correlations_matrix(
        merged_for_corr.rename(columns={c: c for c in merged_for_corr.columns}),
        metric_names=feat_cols_for_corr, prop_names=prop_cols_for_corr
    )

    pflat = pvals.values.flatten()
    idx_valid = ~np.isnan(pflat)
    sig_flat = np.zeros_like(pflat, dtype=bool)
    if idx_valid.sum() > 0:
        sig_flat[idx_valid] = benjamini_hochberg(pflat[idx_valid], alpha=0.05)
    sig_matrix = sig_flat.reshape(pvals.shape)

    plt.figure(figsize=(width_in, max(1.2, 0.5 + 0.4 * len(feat_cols_for_corr))))
    ax = sns.heatmap(rhos.astype(float), annot=True, fmt=".2f", cmap='vlag', center=0, square=False,
                     cbar_kws={'label': 'Spearman ρ'}, annot_kws={'fontsize': small_font})
    for ii in range(rhos.shape[0]):
        for jj in range(rhos.shape[1]):
            if sig_matrix.flatten()[ii * rhos.shape[1] + jj]:
                ax.text(jj + 0.5, ii + 0.5, '*', color='black', ha='center', va='center', fontsize=small_font + 1)
    ax.set_xlabel("Features (median)", fontsize=small_font)
    ax.set_ylabel("Properties", fontsize=small_font)
    plt.title("Spearman correlations (features vs properties)", fontsize=small_font)
    plt.tight_layout()
    heat_png = os.path.join(outdir, "supp_features_physchem_heatmap.png")
    heat_svg = os.path.join(outdir, "supp_features_physchem_heatmap.svg")
    plt.savefig(heat_png, dpi=600, bbox_inches='tight')
    plt.savefig(heat_svg, dpi=300, bbox_inches='tight')
    plt.close()

    rhos.to_csv(os.path.join(outdir, "supp_spearman_rhos_features_vs_props.csv"))
    pvals.to_csv(os.path.join(outdir, "supp_spearman_pvals_features_vs_props.csv"))
    pd.Series(top_features, name="top_features").to_csv(os.path.join(outdir, "top_features_chosen.csv"), index=False)

    logger.info(f"Saved feature vs physchem figures and CSVs to {outdir}")

    return merged, rhos, pvals, top_features