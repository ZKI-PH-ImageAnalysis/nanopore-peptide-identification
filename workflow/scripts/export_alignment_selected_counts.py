#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a per-run CSV with selected alignment counts from "
            "alignment_reach_all.csv."
        )
    )
    parser.add_argument("--input", required=True, help="Input alignment_reach_all.csv")
    parser.add_argument("--out", required=True, help="Output CSV path")

    parser.add_argument("--template-ref", default="template")
    parser.add_argument("--template-threshold", type=int, default=59)

    parser.add_argument("--threading-ref", default="template_N0_threading")
    parser.add_argument("--threading-threshold", type=int, default=110)

    parser.add_argument("--revcomp-ref", default="template_N0_revComTemplate")
    parser.add_argument("--revcomp-threshold", type=int, default=110)

    return parser.parse_args()


def main():
    args = parse_args()

    template_col = (
        f"number_of_alignments_to_{args.template_ref}_plus_ge_{args.template_threshold}"
    )

    per_run = defaultdict(
        lambda: {
            template_col: 0,
            "number_of_alignments_to_template_N0_threading_gt_110": 0,
            "number_of_alignments_to_template_N0_revComTemplate": 0,
        }
    )

    with open(args.input, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run = row["run"]
            reference = row["reference"]
            threshold = int(row["threshold"])
            count = int(row["count"])

            if reference == args.template_ref and threshold == args.template_threshold:
                per_run[run][template_col] = count

            if reference == args.threading_ref and threshold == args.threading_threshold:
                per_run[run]["number_of_alignments_to_template_N0_threading_gt_110"] = count

            if reference == args.revcomp_ref and threshold == args.revcomp_threshold:
                per_run[run]["number_of_alignments_to_template_N0_revComTemplate"] = count

    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "run_name",
                template_col,
                "number_of_alignments_to_template_N0_threading_gt_110",
                "number_of_alignments_to_template_N0_revComTemplate",
            ]
        )

        for run_name in sorted(per_run):
            data = per_run[run_name]
            writer.writerow(
                [
                    run_name,
                    data[template_col],
                    data["number_of_alignments_to_template_N0_threading_gt_110"],
                    data["number_of_alignments_to_template_N0_revComTemplate"],
                ]
            )


if __name__ == "__main__":
    main()
