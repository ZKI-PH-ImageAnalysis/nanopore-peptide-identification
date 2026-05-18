#!/usr/bin/env python3
import argparse, json, subprocess
from pathlib import Path
import pandas as pd

def count_fastq_reads(fastq_path):
    fastq_path = str(fastq_path)
    if fastq_path.endswith(".gz"):
        cmd = ["gzip", "-cd", fastq_path]
    else:
        cmd = ["cat", fastq_path]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.run(["wc", "-l"], stdin=p1.stdout, capture_output=True, text=True)
    p1.stdout.close()
    if p2.returncode != 0:
        return None
    lines = int(p2.stdout.strip().split()[0])
    return lines // 4

def count_aligned_reads(bam_path):
    p = subprocess.run(["samtools", "view", "-F", "4", "-c", bam_path], capture_output=True, text=True)
    if p.returncode != 0:
        return None
    return int(p.stdout.strip())

def per_ref_idxstats(bam_path):
    p = subprocess.run(["samtools", "idxstats", bam_path], capture_output=True, text=True)
    if p.returncode != 0:
        return {}
    out = {}
    for line in p.stdout.strip().splitlines():
        chrom, length, mapped, unmapped = line.split("\t")
        out[chrom] = int(mapped)
    return out

def count_plr_reads(peptide_tsv):
    if not Path(peptide_tsv).exists():
        return {"plr_unique_reads": 0, "plr_count_rows": 0}
    df = pd.read_csv(peptide_tsv, sep="\t", low_memory=False)
    unique_reads = df.shape[0]

    unique_reads_full_alignment = len(df[(df['ref_start'] <= 30) & (df['ref_end'] >= 110)])

    return {"plr_unique_reads": int(unique_reads), "plr_unique_reads_full_alignment": int(unique_reads_full_alignment)}

def active_channels_from_seqsummary(seqsummary_path):
    df = pd.read_csv(seqsummary_path, sep="\t", usecols=["channel"])
    return len(set(df["channel"].dropna().astype(int).unique()))


def rejected_reads_from_seqsummary(seqsummary_path, peptide_tsv, bam_path):
    df = pd.read_csv(seqsummary_path, sep="\t")
    df = df[df['end_reason'].isin(['mux_change', 'signal_negative'])]
    rejected_read_ids = set(df['read_id'])
    
    peptide_df = pd.read_csv(peptide_tsv, sep="\t")
    peptide_read_ids = set(peptide_df['Read_ID'])
    shared_plr = len(rejected_read_ids & peptide_read_ids)
    
    cmd = ["samtools", "view", bam_path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    aligned_read_ids = set()
    for line in p.stdout:
        qname = line.split("\t")[0]
        aligned_read_ids.add(qname)
    p.wait()
    shared_aligned = len(rejected_read_ids & aligned_read_ids)
    
    total_rejected = len(rejected_read_ids)
    ratio_rejected_reads_aligned_reads = shared_aligned / total_rejected if total_rejected > 0 else 0
    ratio_rejected_reads_plr_unique_reads = shared_plr / total_rejected if total_rejected > 0 else 0

    ratio_aligned_reads_rejected_reads = shared_aligned / len(aligned_read_ids) if len(aligned_read_ids) > 0 else 0


    return ratio_rejected_reads_aligned_reads, ratio_rejected_reads_plr_unique_reads, len(rejected_read_ids), ratio_aligned_reads_rejected_reads


def count_alignment_ranges(bam_path, ref_name, start_lt=30, end_gt=110):
    cmd = ["samtools", "view", "-F", "4", bam_path, ref_name]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    total = 0
    spanning = 0

    for line in p.stdout:
        total += 1
        fields = line.split("\t")
        pos = int(fields[3])  # 1-based
        cigar = fields[5]

        ref_len = 0
        num = ""
        for c in cigar:
            if c.isdigit():
                num += c
            else:
                if c in ("M", "D", "N", "=", "X"):
                    ref_len += int(num)
                num = ""

        end = pos + ref_len - 1

        if pos < start_lt and end > end_gt:
            spanning += 1

    p.stdout.close()
    return {
        "aligned_reads_to_ref": total,
        "aligned_reads_to_ref_st30_en110": spanning,
    }


def count_pod5_reads(pod5_path):
    p1 = subprocess.Popen(
        ["pod5", "view", pod5_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    p2 = subprocess.run(
        ["wc", "-l"],
        stdin=p1.stdout,
        capture_output=True,
        text=True,
    )
    p1.stdout.close()

    if p2.returncode != 0:
        return None

    return int(p2.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastq", required=True)
    parser.add_argument("--bam", required=True)
    parser.add_argument("--peptides", required=True)
    parser.add_argument("--seq-summary", required=False, default="")
    parser.add_argument("--pod5", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    res = {}
    res['fastq'] = str(Path(args.fastq))
    res['bam'] = str(Path(args.bam))
    res['peptides'] = str(Path(args.peptides))
    res["pod5_reads"] = count_pod5_reads(args.pod5)
    res['basecalled_reads'] = count_fastq_reads(args.fastq)
    res['aligned_reads_to_ref'] = count_aligned_reads(args.bam)
    res['per_ref_idxstats'] = per_ref_idxstats(args.bam)

    

    plr = count_plr_reads(args.peptides)
    res.update(plr)

    if args.seq_summary:
        res['active_channels'] = active_channels_from_seqsummary(args.seq_summary)
        ratio_rejected_reads_aligned_reads, ratio_rejected_reads_plr_unique_reads, rejected_read_ids, ratio_aligned_reads_rejected_reads = rejected_reads_from_seqsummary(args.seq_summary, args.peptides, args.bam)
        res['ratio_rejected_reads_aligned_reads'] = ratio_rejected_reads_aligned_reads
        res['ratio_rejected_reads_plr_unique_reads'] = ratio_rejected_reads_plr_unique_reads
        res['rejected_reads_count'] = rejected_read_ids
        res['ratio_aligned_reads_rejected_reads'] = ratio_aligned_reads_rejected_reads
    else:
        res['active_channels'] = None
        res['ratio_rejected_reads_aligned_reads'] = None
        res['ratio_rejected_reads_plr_unique_reads'] = None
        res['rejected_reads_count'] = None
        res['ratio_aligned_reads_rejected_reads'] = None
    parts = Path(args.bam).parts
    try:
        run = parts[1]
        ref = parts[2]
    except Exception:
        run = None
        ref = None
    res['run'] = run
    res['reference'] = ref

    align_stats = count_alignment_ranges(
        args.bam,
        ref_name=ref,
        start_lt=30,
        end_gt=110,
    )
    res.update(align_stats)

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

if __name__ == "__main__":
    main()
