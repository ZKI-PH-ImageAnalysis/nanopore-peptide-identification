#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

def load_json(path):
    with open(path) as fh:
        return json.load(fh)

def safe_frac(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for p in args.input:
        j = load_json(p)

        row = {
            "run": j.get("run"),
            "reference": j.get("reference"),

            "pod5_reads": j.get("pod5_reads"),

            "basecalled_reads": j.get("basecalled_reads"),
            "aligned_reads_to_ref": j.get("aligned_reads_to_ref"),
            "plr_unique_reads": j.get("plr_unique_reads"),
            "plr_unique_reads_full_alignment": j.get("plr_unique_reads_full_alignment"),
            "plr_count_rows": j.get("plr_count_rows"),
            "active_channels": j.get("active_channels"),


            "ratio_rejected_reads_aligned_reads": j.get("ratio_rejected_reads_aligned_reads"),
            "ratio_rejected_reads_plr_unique_reads": j.get("ratio_rejected_reads_plr_unique_reads"),
            "ratio_aligned_reads_rejected_reads": j.get("ratio_aligned_reads_rejected_reads"),

            "rejected_reads_count": j.get("rejected_reads_count"),

            "aligned_reads_to_ref_st30_en110": j.get(
                "aligned_reads_to_ref_st30_en110"
            ),
        }

        per_ref = j.get("per_ref_idxstats") or {}
        row["mapped_to_ref_idxstats"] = per_ref.get(row["reference"])

        rows.append(row)

    df = pd.DataFrame(rows)

    df["basecall_yield"] = df.apply(
        lambda r: safe_frac(r["basecalled_reads"], r["pod5_reads"]),
        axis=1,
    )

    df["alignment_yield"] = df.apply(
        lambda r: safe_frac(r["aligned_reads_to_ref"], r["pod5_reads"]),
        axis=1,
    )

    df["full_alignment_yield"] = df.apply(
        lambda r: safe_frac(
            r["aligned_reads_to_ref_st30_en110"],
            r["aligned_reads_to_ref"],
        ),
        axis=1,
    )

    df["plr_fraction_of_aligned"] = df.apply(
        lambda r: safe_frac(
            r["plr_unique_reads"],
            r["aligned_reads_to_ref"],
        ),
        axis=1,
    )

    df["plr_fraction_of_aligned_full"] = df.apply(
        lambda r: safe_frac(
            r["plr_unique_reads_full_alignment"],
            r["aligned_reads_to_ref_st30_en110"],
        ),
        axis=1,
    )

    df.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
