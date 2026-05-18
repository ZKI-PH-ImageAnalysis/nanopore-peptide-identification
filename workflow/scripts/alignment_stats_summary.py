import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def ensure_outdir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    df['threshold'] = df['threshold'].astype(int)
    df['count'] = df['count'].astype(int)
    return df


def make_wide(df, thresholds):
    pivot = (
        df.pivot_table(index=['run','reference'], columns='threshold', values='count', aggfunc='sum', fill_value=0)
    )
    for t in thresholds:
        if t not in pivot.columns:
            pivot[t] = 0
    pivot = pivot[thresholds]
    pivot = pivot.reset_index()
    return pivot


def exclusive_bins_from_cumulative(row, thresholds):
    vals = [row[t] for t in thresholds]
    total = row['total']
    excl = {}
    excl['<59'] = total - vals[0]
    excl['59-69'] = vals[0] - vals[1]
    excl['70-89'] = vals[1] - vals[2]
    excl['90-99'] = vals[2] - vals[3]
    excl['100-109'] = vals[3] - vals[4]
    excl['>=110'] = vals[4]
    return excl


def plot_stacked_counts(wide_df, ref_name, thresholds, outdir):
    sub = wide_df[wide_df['reference']==ref_name].copy()
    sub = sub.sort_values('run')
    runs = sub['run'].tolist()
    counts = sub[thresholds].values

    fig, ax = plt.subplots(figsize=(max(8, len(runs)*0.25),6))

    bottom = np.zeros(len(runs), dtype=int)
    labels = [f"≥{t}" for t in thresholds]
    for i, t in enumerate(thresholds):
        vals = sub[t].values
        ax.bar(runs, vals, bottom=bottom, label=labels[i])
        bottom = bottom + vals

    ax.set_ylabel('Number of alignments')
    ax.set_xlabel('Run')
    ax.set_title(f'Stacked cumulative counts — {ref_name}')
    ax.tick_params(axis='x', rotation=90)
    ax.legend(title='Thresholds', bbox_to_anchor=(1.02,1), loc='upper left')
    fig.tight_layout()
    out_png = Path(outdir)/f"stacked_cumulative_counts_{ref_name}.png"
    out_svg = Path(outdir)/f"stacked_cumulative_counts_{ref_name}.svg"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)


def plot_stacked_percent(wide_df, ref_name, thresholds, outdir):
    sub = wide_df[wide_df['reference']==ref_name].copy()
    sub = sub.sort_values('run')
    runs = sub['run'].tolist()
    totals = sub['total'].values.astype(float)

    fig, ax = plt.subplots(figsize=(max(8, len(runs)*0.25),6))
    bottom = np.zeros(len(runs), dtype=float)
    labels = [f"≥{t}" for t in thresholds]

    for i, t in enumerate(thresholds):
        vals = (sub[t].values / totals)
        ax.bar(runs, vals, bottom=bottom, label=labels[i])
        bottom = bottom + vals

    ax.set_ylabel('Fraction of alignments')
    ax.set_xlabel('Run')
    ax.set_title(f'Stacked cumulative percent — {ref_name}')
    ax.tick_params(axis='x', rotation=90)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title='Thresholds', bbox_to_anchor=(1.02,1), loc='upper left')
    fig.tight_layout()
    out_png = Path(outdir)/f"stacked_cumulative_percent_{ref_name}.png"
    out_svg = Path(outdir)/f"stacked_cumulative_percent_{ref_name}.svg"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)


def plot_exclusive_stacked(wide_df, ref_name, thresholds, outdir):
    sub = wide_df[wide_df['reference']==ref_name].copy()
    sub = sub.sort_values('run')
    runs = sub['run'].tolist()

    sub['total'] = sub[thresholds[0]] 
    

    excl_cols = ['<59','59-69','70-89','90-99','100-109','>=110']
    excl_matrix = np.zeros((len(sub), len(excl_cols)), dtype=int)
    for i, (_, row) in enumerate(sub.iterrows()):
        excl = exclusive_bins_from_cumulative(row, thresholds)
        for j, c in enumerate(excl_cols):
            excl_matrix[i,j] = excl[c]

    fig, ax = plt.subplots(figsize=(max(8, len(runs)*0.25),6))
    bottom = np.zeros(len(runs), dtype=int)
    for j, c in enumerate(excl_cols):
        vals = excl_matrix[:,j]
        ax.bar(runs, vals, bottom=bottom, label=c)
        bottom += vals

    ax.set_ylabel('Number of alignments (exclusive bins)')
    ax.set_xlabel('Run')
    ax.set_title(f'Exclusive-bin stacked counts — {ref_name}')
    ax.tick_params(axis='x', rotation=90)
    ax.legend(title='Bins', bbox_to_anchor=(1.02,1), loc='upper left')
    fig.tight_layout()

    out_png = Path(outdir)/f"stacked_exclusive_counts_{ref_name}.png"
    out_svg = Path(outdir)/f"stacked_exclusive_counts_{ref_name}.svg"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)


def plot_threading_vs_revcomp(wide_df, thresholds, outdir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from pathlib import Path

    df = wide_df.copy()
    df['total_mapped_proxy'] = df[thresholds[0]]  
    thr = df[df['reference'] == 'template_N0_threading'].set_index('run')
    rev = df[df['reference'] == 'template_N0_revComTemplate'].set_index('run')
    runs = sorted(set(thr.index).union(set(rev.index)))

    threading_counts = [thr.loc[r, thresholds[-1]] if r in thr.index else 0 for r in runs]
    revcomp_counts   = [rev.loc[r, thresholds[-1]] if r in rev.index else 0 for r in runs]

    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.25), 6))
    ax.bar(runs, threading_counts, label='Threading')
    ax.bar(runs, revcomp_counts, bottom=threading_counts, label='RevComp')
    ax.set_ylabel('Number of alignments (>=110)')
    ax.set_xlabel('Run')
    ax.set_title('Full-length (>=110) threading vs revComp — counts')
    ax.tick_params(axis='x', rotation=90)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(Path(outdir) / "threading_revcomp_counts.png", dpi=300)
    fig.savefig(Path(outdir) / "threading_revcomp_counts.svg")
    plt.close(fig)

    combined_full_counts = [t + r for t, r in zip(threading_counts, revcomp_counts)]
    thr_pct = [(t / c) if c > 0 else 0 for t, c in zip(threading_counts, combined_full_counts)]
    rev_pct = [(r / c) if c > 0 else 0 for r, c in zip(revcomp_counts, combined_full_counts)]

    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.25), 6))
    ax.bar(runs, thr_pct, label='Threading')
    ax.bar(runs, rev_pct, bottom=thr_pct, label='RevComp')
    ax.set_ylabel('Fraction of full-length alignments (>=110)')
    ax.set_xlabel('Run')
    ax.set_title('Full-length threading vs revComp — percent (combined total = 100%)')
    ax.tick_params(axis='x', rotation=90)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(Path(outdir) / "threading_revcomp_percent.png", dpi=300)
    fig.savefig(Path(outdir) / "threading_revcomp_percent.svg")
    plt.close(fig)



def plot_ecdf_per_ref(df_long, thresholds, outdir):
    
    totals = df_long[df_long['threshold'] == thresholds[0]].set_index(['run', 'reference'])['count']
    
    df2 = df_long.set_index(['run', 'reference'])
    df2['total'] = totals
    df2 = df2.reset_index()
    
    df2 = df2[df2['total'] > 0].copy()
    df2['fraction'] = df2['count'] / df2['total']

    references = df2['reference'].unique()
    for ref in references:
        ref_data = df2[df2['reference'] == ref]
        summary = ref_data.groupby('threshold')['fraction'].median().reindex(thresholds)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(summary.index, summary.values, marker='o')
        ax.set_xlabel('Threshold (reference position)')
        ax.set_ylabel('Median fraction of reads reaching threshold')
        ax.set_title(f'Median drop-off curve — {ref}')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, 1)
        fig.tight_layout()
        ref_safe = ref.replace('/', '_').replace('\\', '_')
        fig.savefig(Path(outdir) / f"median_dropoff_curve_{ref_safe}.png", dpi=300)
        fig.savefig(Path(outdir) / f"median_dropoff_curve_{ref_safe}.svg")
        plt.close(fig)


def plot_ecdf(df_long, thresholds, outdir):
    totals = df_long[df_long['threshold'] == thresholds[0]].set_index(['run', 'reference'])['count']
    
    df2 = df_long.set_index(['run', 'reference'])
    df2['total'] = totals
    df2 = df2.reset_index()
    
    df2 = df2[df2['total'] > 0].copy()
    df2['fraction'] = df2['count'] / df2['total']

    summary = df2.groupby('threshold')['fraction'].median().reindex(thresholds)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary.index, summary.values, marker='o')
    ax.set_xlabel('Threshold (reference position)')
    ax.set_ylabel('Median fraction of reads reaching threshold')
    ax.set_title('Median drop-off curve across runs')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(Path(outdir) / "median_dropoff_curve.png", dpi=300)
    fig.savefig(Path(outdir) / "median_dropoff_curve.svg")
    plt.close(fig)


def write_summary_csv(wide_df, thresholds, outdir):
    rows = []
    for _, row in wide_df.iterrows():
        total = row[thresholds[0]] if row[thresholds[0]]>0 else 1
        r = {'run': row['run'], 'reference': row['reference'], 'total_proxy': total}
        for t in thresholds:
            r[f'p_ge_{t}'] = row[t] / total
        rows.append(r)

    summ = pd.DataFrame(rows)
    summ.to_csv(Path(outdir)/"alignment_reach_summary.csv", index=False)


def main(args):
    ensure_outdir(args.outdir)
    df = load_data(args.input)
    thresholds = [59,70,90,100,110]

    wide = make_wide(df, thresholds)
    wide['total'] = wide[59]

    wide.to_csv(Path(args.outdir)/"alignment_reach_wide.csv", index=False)

    references = wide['reference'].unique()
    for ref in references:
        plot_stacked_counts(wide, ref, thresholds, args.outdir)
        plot_stacked_percent(wide, ref, thresholds, args.outdir)
        plot_exclusive_stacked(wide, ref, thresholds, args.outdir)

    plot_threading_vs_revcomp(wide, thresholds, args.outdir)

    plot_ecdf(df, thresholds, args.outdir)
    plot_ecdf_per_ref(df, thresholds, args.outdir)
    write_summary_csv(wide, thresholds, args.outdir)

    print('Wrote plots and CSVs to', args.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='alignment_reach_all.csv')
    parser.add_argument('--outdir', required=True, help='output directory for plots')
    args = parser.parse_args()
    main(args)
