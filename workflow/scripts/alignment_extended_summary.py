#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

def ensure_outdir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)



def plot_rejected_reads(df, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].scatter(df["rejected_reads_count"], df["ratio_rejected_reads_aligned_reads"])
    axes[0].set_xlabel("Rejected Reads Count")
    axes[0].set_ylabel("Fraction of rejected Reads that are aligned")
    axes[0].set_title("Rejected Count vs Ratio in Aligned")
    
    axes[1].scatter(df["rejected_reads_count"], df["ratio_rejected_reads_plr_unique_reads"])
    axes[1].set_xlabel("Rejected Reads Count")
    axes[1].set_ylabel("Fraction of rejected Reads that are detected as PLR")
    axes[1].set_title("Rejected Count vs Ratio in PLR")
    
    # Compute fraction of aligned reads that are rejected
    df["frac_aligned_rejected"] = (df["ratio_rejected_reads_aligned_reads"] * df["rejected_reads_count"]) / df["aligned_reads_to_ref"].replace(0, np.nan)
    df["frac_aligned_rejected"] = df["frac_aligned_rejected"].fillna(0)
    
    axes[2].scatter(df["rejected_reads_count"], df["frac_aligned_rejected"])
    axes[2].set_xlabel("Rejected Reads Count")
    axes[2].set_ylabel("Fraction of Aligned Reads that are Rejected")
    axes[2].set_title("Rejected Count vs Fraction Aligned Rejected")
    
    fig.tight_layout()
    fig.savefig(Path(outdir) / "rejected_reads_scatter.png", dpi=300)
    plt.close(fig)
    
    # Bar plots per reference
    references = df["reference"].unique()
    for ref in references:
        sub = df[df["reference"] == ref].sort_values("run")
        runs = sub["run"].astype(str).tolist()
        ratio1 = safe_numeric(sub["ratio_rejected_reads_aligned_reads"]).values
        ratio2 = safe_numeric(sub["ratio_rejected_reads_plr_unique_reads"]).values
        frac3 = safe_numeric(sub["frac_aligned_rejected"]).values
        
        x = np.arange(len(runs))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.25), 5))
        ax.bar(x - width, ratio1, width, label="Ratio rejected in aligned")
        ax.bar(x, ratio2, width, label="Ratio rejected in PLR")
        ax.bar(x + width, frac3, width, label="Fraction aligned that are rejected")
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=90)
        ax.set_title(f"Rejected Ratios and Fractions — {ref}")
        ax.legend()
        fig.tight_layout()
        safe = str(ref).replace("/", "_").replace("\\", "_")
        fig.savefig(Path(outdir) / f"rejected_ratios_{safe}.png", dpi=300)
        plt.close(fig)
    
    # median fraction of aligned reads that are rejected per reference
    summary = df.groupby("reference")["frac_aligned_rejected"].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary.index.astype(str), summary.values)
    ax.set_ylabel("Median fraction of aligned reads that are rejected")
    ax.tick_params(axis="x", rotation=90)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(Path(outdir) / "frac_aligned_rejected_by_reference.png", dpi=300)
    plt.close(fig)

def plot_reads_vs_alignments(df, outdir, control=True, y_percent=False):
    grouped = (
        df.groupby("run")
          .agg({
              "pod5_reads": "first",
              "basecalled_reads": "first",
              "active_channels": "first"
          })
          .reset_index()
    )

    pivot_aligned = df.pivot_table(index="run", columns="reference", values="aligned_reads_to_ref", aggfunc="sum", fill_value=0)
    pivot_aligned.columns = ["aligned_reads_to_" + str(col) for col in pivot_aligned.columns]
    grouped = grouped.merge(pivot_aligned, on="run", how="left")

    pivot_aligned_st30_en110 = df.pivot_table(index="run", columns="reference", values="aligned_reads_to_ref_st30_en110", aggfunc="sum", fill_value=0)
    pivot_aligned_st30_en110.columns = ["aligned_reads_to_ref_st30_en110_" + str(col) for col in pivot_aligned_st30_en110.columns]
    grouped = grouped.merge(pivot_aligned_st30_en110, on="run", how="left")

    grouped["aligned_reads_total"] = grouped.filter(like="aligned_reads_to_").sum(axis=1)

    runs = grouped["run"].astype(str).tolist()
    x = np.arange(len(runs))

    basecalled = grouped["basecalled_reads"].values
    aligned_CS = grouped["aligned_reads_to_DNA_CS"].values
    aligned_template = grouped["aligned_reads_to_template"].values
    aligned_template_threading = grouped["aligned_reads_to_ref_st30_en110_template_N0_threading"].values
    aligned_template_RCtemplate = grouped["aligned_reads_to_ref_st30_en110_template_N0_revComTemplate"].values

    if y_percent:
        for i in range(len(runs)):
            if control: 
                values = [basecalled[i], aligned_CS[i], aligned_template[i], aligned_template_threading[i], aligned_template_RCtemplate[i]]
            else: 
                values = [aligned_template[i], aligned_template_threading[i], aligned_template_RCtemplate[i]]
            max_val = max(values) if values else 1
            if max_val > 0:
                if control: 
                    basecalled[i] = (basecalled[i] / max_val) * 100
                    aligned_CS[i] = (aligned_CS[i] / max_val) * 100
                    aligned_template[i] = (aligned_template[i] / max_val) * 100
                    aligned_template_threading[i] = (aligned_template_threading[i] / max_val) * 100
                    aligned_template_RCtemplate[i] = (aligned_template_RCtemplate[i] / max_val) * 100
                else:
                    aligned_template[i] = (aligned_template[i] / max_val) * 100
                    aligned_template_threading[i] = (aligned_template_threading[i] / max_val) * 100
                    aligned_template_RCtemplate[i] = (aligned_template_RCtemplate[i] / max_val) * 100

    fig, ax = plt.subplots(figsize=(max(12, len(runs) * 0.5), 8))
    width = 0.15


    if control:
        ax.bar(x - 2*width, basecalled, width, label="Basecalled Reads", color='C0')
        ax.bar(x - width, aligned_CS, width, label="Aligned to DNA_CS", color='C1')
        ax.bar(x, aligned_template, width, label="Aligned to Template", color='C2')
        ax.bar(x + width, aligned_template_threading, width, label="Template Threading (Full)", color='C3')
        ax.bar(x + 2*width, aligned_template_RCtemplate, width, label="Template RevComp (Full)", color='C4')
    else:
        ax.bar(x - width, aligned_template, width, label="Aligned to Template", color='C2')
        ax.bar(x, aligned_template_threading, width, label="Template Threading (Full)", color='C3')
        ax.bar(x + width, aligned_template_RCtemplate, width, label="Template RevComp (Full)", color='C4')

    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha='right')
    ax.set_xlim(x[0] - 2.5*width, x[-1] + 2.5*width)  # Reduce white space on x-axis sides
    if y_percent:
        ax.set_ylabel("Percentage (%)")
        ax.yaxis.set_major_formatter(PercentFormatter())
    else:
        ax.set_ylabel("Number of Reads")
        def thousands_formatter(x, pos):
            if x >= 1e6:
                return f'{x / 1e6:.2f}M'
            elif x >= 1000:
                return f'{int(x / 1000)}K'
            else:
                return str(int(x))
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_title("Read Counts per Run")
    ax.grid(True, alpha=0.3)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=3)

    fig.tight_layout()
    if control:
        fname = "reads_basecalls_alignments_control_percent.png" if y_percent else "reads_basecalls_alignments_control.png"
    else:
        fname = "reads_basecalls_alignments_percent.png" if y_percent else "reads_basecalls_alignments.png"
    fig.savefig(Path(outdir) / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)



def plot_reads_vs_alignments_vs_PLRs(df, outdir):
    grouped = (
        df.groupby("run")
          .agg({
              "pod5_reads": "first",
              "basecalled_reads": "first",
              "active_channels": "first"
          })
          .reset_index()
    )

    pivot_aligned = df.pivot_table(index="run", columns="reference", values="plr_unique_reads", aggfunc="sum", fill_value=0)
    pivot_aligned.columns = ["plr_unique_reads_" + str(col) for col in pivot_aligned.columns]
    grouped = grouped.merge(pivot_aligned, on="run", how="left")

    pivot_aligned_st30_en110 = df.pivot_table(index="run", columns="reference", values="plr_unique_reads_full_alignment", aggfunc="sum", fill_value=0)
    pivot_aligned_st30_en110.columns = ["plr_unique_reads_full_alignment_" + str(col) for col in pivot_aligned_st30_en110.columns]
    grouped = grouped.merge(pivot_aligned_st30_en110, on="run", how="left")


    grouped["aligned_reads_total"] = grouped.filter(like="aligned_reads_to_").sum(axis=1)

    runs = grouped["run"].astype(str).tolist()
    x = np.arange(len(runs))

    aligned_template = grouped["plr_unique_reads_template"].values
    aligned_template_threading = grouped["plr_unique_reads_full_alignment_template_N0_threading"].values
    aligned_template_RCtemplate = grouped["plr_unique_reads_full_alignment_template_N0_revComTemplate"].values

    fig, ax = plt.subplots(figsize=(max(12, len(runs) * 0.5), 8))
    width = 0.15  

    ax.bar(x - width, aligned_template, width, label="PLR detected Template", color='C2')
    ax.bar(x, aligned_template_threading, width, label="PLR detected Template+Threading", color='C3')
    ax.bar(x + width, aligned_template_RCtemplate, width, label="PLR detected Template+RC template", color='C4')

    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha='right')
    ax.set_xlim(x[0] - 2.5*width, x[-1] + 2.5*width)  # Reduce white space on x-axis sides
    ax.set_ylabel("Number of Reads")
    ax.set_title("Read Counts per Run")
    ax.grid(True, alpha=0.3)

    def thousands_formatter(x, pos):
        if x >= 1e6:
            return f'{x / 1e6:.2f}M'
        elif x >= 1000:
            return f'{int(x / 1000)}K'
        else:
            return str(int(x))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=3)

    fig.tight_layout()
    fig.savefig(Path(outdir) / "reads_basecalls_PLRs.png", dpi=300, bbox_inches='tight')
    plt.close(fig)



def plot_active_bar(df, outdir):
    per_run = (
        df.groupby("run")
          .agg({"active_channels": "first", "basecalled_reads": "first", "aligned_reads_to_ref": "sum", "pod5_reads":"first"})
          .reset_index()
    )
    for c in ["active_channels","basecalled_reads","aligned_reads_to_ref","pod5_reads"]:
        if c in per_run.columns:
            per_run[c] = safe_numeric(per_run[c])

    runs = per_run["run"].astype(str).tolist()
    x = np.arange(len(runs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(runs) * 0.28), 6))
    ax.bar(x - 1.5*width, per_run["active_channels"].values, width, label="active_channels")
    ax.bar(x - 0.5*width, per_run["pod5_reads"].values, width, label="pod5_reads")
    ax.bar(x + 0.5*width, per_run["basecalled_reads"].values, width, label="basecalled_reads")
    ax.bar(x + 1.5*width, per_run["aligned_reads_to_ref"].values, width, label="aligned_reads_total")

    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=90)
    ax.set_ylabel("Counts / channels")
    ax.set_title("Per-run pore activity and read counts (pod5 / basecalled / aligned)")
    ax.legend(bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(Path(outdir) / "active_and_counts_bar.png", dpi=300)
    plt.close(fig)

def plot_plr_vs_all(df, outdir):
    references = df["reference"].unique()
    for ref in references:
        sub = df[df["reference"] == ref].sort_values("run")
        runs = sub["run"].astype(str).tolist()
        aligned = safe_numeric(sub["aligned_reads_to_ref"]).astype(int).values
        plr = safe_numeric(sub["plr_unique_reads"]).astype(int).values

        fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.25), 5))
        ax.bar(runs, aligned, label="aligned_reads")
        ax.bar(runs, plr, label="plr_unique_reads", alpha=0.8)
        ax.set_title(f"Aligned vs PLR reads — {ref}")
        ax.tick_params(axis="x", rotation=90)
        ax.legend()
        fig.tight_layout()
        safe = str(ref).replace("/", "_").replace("\\", "_")
        fig.savefig(Path(outdir) / f"aligned_vs_plr_{safe}.png", dpi=300)
        plt.close(fig)

    df["plr_frac"] = df.apply(lambda r: (safe_numeric(pd.Series([r.get("plr_unique_reads")])).iloc[0] / safe_numeric(pd.Series([r.get("aligned_reads_to_ref")])).iloc[0]) if (r.get("aligned_reads_to_ref") not in (None, 0)) else 0, axis=1)
    summary = df.groupby("reference")["plr_frac"].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary.index.astype(str), summary.values)
    ax.set_ylabel("median PLR fraction among aligned")
    ax.tick_params(axis="x", rotation=90)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(Path(outdir) / "plr_fraction_by_reference.png", dpi=300)
    plt.close(fig)

def plot_reads_vs_alignments_threading(df, outdir, template=True):
    grouped = (
        df.groupby("run")
          .agg({
              "pod5_reads": "first",
              "basecalled_reads": "first",
              "active_channels": "first"
          })
          .reset_index()
    )

    pivot_aligned = df.pivot_table(index="run", columns="reference", values="aligned_reads_to_ref", aggfunc="sum", fill_value=0)
    pivot_aligned.columns = ["aligned_reads_to_" + str(col) for col in pivot_aligned.columns]
    grouped = grouped.merge(pivot_aligned, on="run", how="left")

    pivot_aligned_st30_en110 = df.pivot_table(index="run", columns="reference", values="aligned_reads_to_ref_st30_en110", aggfunc="sum", fill_value=0)
    pivot_aligned_st30_en110.columns = ["aligned_reads_to_ref_st30_en110_" + str(col) for col in pivot_aligned_st30_en110.columns]
    grouped = grouped.merge(pivot_aligned_st30_en110, on="run", how="left")

    grouped["aligned_reads_total"] = grouped.filter(like="aligned_reads_to_").sum(axis=1)

    runs = grouped["run"].astype(str).tolist()
    x = np.arange(len(runs))

    aligned_template = grouped["aligned_reads_to_template"].values
    aligned_template_threading = grouped["aligned_reads_to_ref_st30_en110_template_N0_threading"].values
    aligned_template_RCtemplate = grouped["aligned_reads_to_ref_st30_en110_template_N0_revComTemplate"].values
    aligned_threading = grouped["aligned_reads_to_threading"].values
    aligned_threading_RCthreading = grouped["aligned_reads_to_ref_st30_en110_threading_revComp"].values

    fig, ax = plt.subplots(figsize=(max(12, len(runs) * 0.5), 8))
    width = 0.1

    if template:
        ax.bar(x - width, aligned_template, width, label="Aligned to Template", color='C2')
        ax.bar(x + width, aligned_template_RCtemplate, width, label="Template RevComp (Full)", color='C4')
    ax.bar(x, aligned_template_threading, width, label="Template Threading (Full)", color='C3')
    ax.bar(x - 2*width, aligned_threading, width, label="Aligned to Threading", color='C5')
    ax.bar(x + 2*width, aligned_threading_RCthreading, width, label="Threading RevComp (Full)", color='C6')

    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha='right')
    ax.set_xlim(x[0] - 2.5*width, x[-1] + 2.5*width)  # Reduce white space on x-axis sides
    ax.set_ylabel("Number of Reads")
    def thousands_formatter(x, pos):
        if x >= 1e6:
            return f'{x / 1e6:.2f}M'
        elif x >= 1000:
            return f'{int(x / 1000)}K'
        else:
            return str(int(x))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_title("Read Counts per Run")
    ax.grid(True, alpha=0.3)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=3)

    fig.tight_layout()
    if template:
        fname = "reads_basecalls_alignments_template_threading.png"
    else:
        fname = "reads_basecalls_alignments_threading.png"
    fig.savefig(Path(outdir) / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)



def plot_threading_vs_revcomp_ratios(df, outdir,
                                    threading_ref="template_N0_threading",
                                    revcomp_ref="template_N0_revComTemplate"):
    pivot_align = df.pivot_table(index="run", columns="reference", values="aligned_reads_to_ref", aggfunc="sum", fill_value=0)
    pivot_full = df.pivot_table(index="run", columns="reference", values="aligned_reads_to_ref_st30_en110", aggfunc="sum", fill_value=0)
    pivot_plr = df.pivot_table(index="run", columns="reference", values="plr_unique_reads", aggfunc="sum", fill_value=0)

    runs = sorted(set(df["run"].astype(str).unique()))

    aligned_ratio = []
    full_ratio = []
    plr_ratio = []

    for r in runs:
        thr = pivot_align.loc[r][threading_ref] if (threading_ref in pivot_align.columns and r in pivot_align.index) else 0
        rev = pivot_align.loc[r][revcomp_ref] if (revcomp_ref in pivot_align.columns and r in pivot_align.index) else 0
        aligned_ratio.append(thr / (thr + rev) if (thr + rev) > 0 else np.nan)

        thr_full = pivot_full.loc[r][threading_ref] if (threading_ref in pivot_full.columns and r in pivot_full.index) else 0
        rev_full = pivot_full.loc[r][revcomp_ref] if (revcomp_ref in pivot_full.columns and r in pivot_full.index) else 0
        full_ratio.append(thr_full / (thr_full + rev_full) if (thr_full + rev_full) > 0 else np.nan)

        thr_plr = pivot_plr.loc[r][threading_ref] if (threading_ref in pivot_plr.columns and r in pivot_plr.index) else 0
        rev_plr = pivot_plr.loc[r][revcomp_ref] if (revcomp_ref in pivot_plr.columns and r in pivot_plr.index) else 0
        plr_ratio.append(thr_plr / (thr_plr + rev_plr) if (thr_plr + rev_plr) > 0 else np.nan)

    aligned_ratio = np.array(aligned_ratio, dtype=float)
    full_ratio = np.array(full_ratio, dtype=float)
    plr_ratio = np.array(plr_ratio, dtype=float)


    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    axs = axes.ravel()

    axs[0].bar(runs, aligned_ratio)
    axs[0].set_xticklabels(runs, rotation=90)
    axs[0].set_title("Aligned reads ratio (threading / (threading+revcomp))")
    axs[0].set_ylabel("Fraction")
    axs[0].yaxis.set_major_formatter(PercentFormatter(1.0))

    axs[1].bar(runs, full_ratio)
    axs[1].set_xticklabels(runs, rotation=90)
    axs[1].set_title("Full alignments ratio (start<30 & end>110)")

    axs[2].bar(runs, plr_ratio)
    axs[2].set_xticklabels(runs, rotation=90)
    axs[2].set_title("PLR reads ratio (threading / (threading+revcomp))")

    fig.suptitle(f"Threading vs RevComp ratios ({threading_ref} vs {revcomp_ref})", y=1.02)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "threading_vs_revcomp_ratios.png", dpi=300)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="results/extended_summary.csv")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    df = pd.read_csv(args.input)

    #run_prefixes = ( "04-", "09_", "10_", "11_", "12_", "13_", "14_", "15_", "48_",) # ssdna, dsdna, only template
    #run_prefixes = ( "04-", "08", "09", "13_", "14_", "15_") # dsdna experiments
    #run_prefixes = ( "13_", "14_", "15_", "35_", "36_", "39") # cysteine free template
    #df = df[df["run"].astype(str).str.startswith(run_prefixes)]

    numeric_cols = [
        "pod5_reads", "basecalled_reads", "aligned_reads_to_ref",
        "aligned_reads_to_ref_st30_en110", "plr_unique_reads", "plr_unique_reads_full_alignment", "active_channels",
        "rejected_reads_count", "ratio_rejected_reads_aligned_reads", "ratio_rejected_reads_plr_unique_reads"
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    plot_reads_vs_alignments(df, args.outdir, control=True)
    plot_reads_vs_alignments(df, args.outdir, control=True, y_percent=True)
    plot_reads_vs_alignments(df, args.outdir, control=False)
    plot_reads_vs_alignments(df, args.outdir, control=False, y_percent=True)

    plot_reads_vs_alignments_threading(df, args.outdir, template=True)
    plot_reads_vs_alignments_threading(df, args.outdir, template=False)

    plot_reads_vs_alignments_vs_PLRs(df, args.outdir)

    print("Wrote plots to", args.outdir)

if __name__ == "__main__":
    main()
