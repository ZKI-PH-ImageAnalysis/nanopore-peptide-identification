#!/usr/bin/env python3
import csv
import glob
import os
import concurrent.futures
import warnings
import gzip

fastq_paths = sorted(glob.glob("results/*/calls.fastq"))
run_names   = [os.path.basename(os.path.dirname(p)) for p in fastq_paths]

example_sams = sorted(glob.glob(f"results/{run_names[0]}/*/output.bam"))
ref_names    = [os.path.basename(os.path.dirname(p)) for p in example_sams]

run_meta = {}
with open("config/metadata.csv") as fh:
    rdr = csv.DictReader(fh)
    for row in rdr:
        run = row["run_name"]
        pep = row["class"]
        ds  = row.get("dataset", "").strip().lower()  # "" or "train" or "test"
        run_meta[run] = (pep, ds)

val_f1 = {}
try:
    with open("results/classification/per_peptide_val_metrics.csv") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            val_f1[row["peptide"]] = float(row["f1_score"])
except FileNotFoundError:
    print("Warning: per_peptide_val_metrics.csv not found, skipping...")

test_f1 = {}
try:
    with open("results/classification/per_peptide_test_metrics.csv") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            test_f1[row["peptide"]] = float(row["f1_score"])
except FileNotFoundError:
    print("Warning: per_peptide_test_metrics.csv not found, skipping...")
    def count_lines_fast(path, chunk_size=1 << 20):
        count = 0
        open_func = gzip.open if path.endswith(".gz") else open
        with open_func(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                count += chunk.count(b"\n")
        return count

    def count_basecalled_reads(run):
        fq_path = f"results/{run}/calls.fastq.gz"
        return run, count_lines_fast(fq_path) // 4

with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    basecounts = dict(executor.map(count_basecalled_reads, run_names))

example_sams = glob.glob(f"results/{run_names[0]}/*/output.bam")
ref_names = [os.path.basename(os.path.dirname(p)) for p in example_sams]

run_ref_pairs = [(run, ref) for run in run_names for ref in ref_names]

def process_run_ref(run_ref):
    run, ref = run_ref
    # look up peptide class and dataset for this run
    peptide, dataset = run_meta.get(run, ("<unknown>", ""))
    sam_ids_path = f"results/{run}/{ref}/read-ids-alignmethod.txt"
    plr_tsv      = f"results/{run}/{ref}/align-plots/peptide_signals.tsv"

    def safe_count(path):
        try:
            return count_lines_fast(path)
        except FileNotFoundError:
            warnings.warn(f"Missing file: {path} → count set to 0")
            return 0

    n_aligned = safe_count(sam_ids_path)
    n_plr     = safe_count(plr_tsv)
    n_base    = basecounts.get(run, 0)

    if dataset == "train":
        f1 = val_f1.get(peptide, "NaN")
    elif dataset == "test":
        f1 = test_f1.get(peptide, "NaN")
    else:
        f1 = "NaN"

    return [run, peptide, dataset, ref, n_base, n_aligned, n_plr, f1]

with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(executor.map(process_run_ref, run_ref_pairs))

with open("results/summary_alignment.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run_name","ref_name","n_basecalled","n_aligned","n_PLR_identified"])
    for row in results:
        run, pep, ds, ref, n_base, n_align, n_plr, f1 = row
        w.writerow([run, ref, n_base, n_align, n_plr])

with open("results/summary_accuracy.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run_name","peptide","dataset","f1_score"])
    for run in run_names:
        peptide, dataset = run_meta[run]
        f1 = val_f1[peptide] if dataset=="train" else test_f1[peptide]
        w.writerow([run, peptide, dataset, f1])