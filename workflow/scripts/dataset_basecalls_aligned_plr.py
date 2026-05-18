#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import gzip
import os
from pathlib import Path


def count_lines_fast(path, chunk_size=1 << 20):
    opener = gzip.open if str(path).endswith(".gz") else open
    count = 0
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            count += chunk.count(b"\n")
    return count


def extract_run_name(path):
    parts = Path(path).parts
    try:
        idx = parts.index("results")
    except ValueError as exc:
        raise ValueError(f"Path does not contain 'results': {path}") from exc

    if idx + 1 >= len(parts):
        raise ValueError(f"Could not infer run name from path: {path}")

    return parts[idx + 1]


def parse_filtering_summary(path):
    values = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = row.get("Filter", "").strip()
            count = row.get("Count", "0").strip()
            if not key:
                continue
            values[key] = int(count)
    return values


def count_basecalls_for_run(run_and_path):
    run_name, fastq_path = run_and_path
    return run_name, count_lines_fast(fastq_path) // 4


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a TSV with run name, number of basecalls, "
            "number of template alignments, and number of valid PLRs."
        )
    )
    parser.add_argument("--fastqs", nargs="+", required=True)
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of worker processes for FASTQ counting.",
    )
    args = parser.parse_args()

    fastq_map = {extract_run_name(path): path for path in args.fastqs}
    summary_map = {extract_run_name(path): path for path in args.summaries}

    runs_missing_fastq = sorted(set(summary_map) - set(fastq_map))
    runs_missing_summary = sorted(set(fastq_map) - set(summary_map))
    if runs_missing_fastq or runs_missing_summary:
        missing = []
        if runs_missing_fastq:
            missing.append(
                "missing fastq for runs: " + ", ".join(runs_missing_fastq)
            )
        if runs_missing_summary:
            missing.append(
                "missing filtering_summary for runs: "
                + ", ".join(runs_missing_summary)
            )
        raise SystemExit("; ".join(missing))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, args.workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        basecall_counts = dict(
            executor.map(count_basecalls_for_run, sorted(fastq_map.items()))
        )

    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "name",
                "number of basecalls",
                "number of alignment to template",
                "number of valid PLRs",
                "valid PLRs / alignments (%)",
            ]
        )

        for run_name in sorted(fastq_map):
            n_basecalls = basecall_counts[run_name]
            summary = parse_filtering_summary(summary_map[run_name])
            n_template_aligned = summary.get(
                "Total Input Reads", summary.get("Total Processed", 0)
            )
            n_valid_plr = summary.get("Valid Reads", 0)
            valid_plr_pct = (
                (n_valid_plr / n_template_aligned) * 100
                if n_template_aligned
                else 0.0
            )

            writer.writerow(
                [
                    run_name,
                    n_basecalls,
                    n_template_aligned,
                    n_valid_plr,
                    f"{valid_plr_pct:.2f}",
                ]
            )


if __name__ == "__main__":
    main()
