import pandas as pd
import os
import sys

def infer_class_from_run_name(run_name):
    parts = run_name.split('_')
    if len(parts) >= 4:
        return parts[3].split('-')[0]
    return "UNKNOWN"

def update_metadata(input_metadata_path, output_metadata_path, data_dirs):
    if os.path.exists(input_metadata_path):
        metadata = pd.read_csv(input_metadata_path)
    else:
        metadata = pd.DataFrame(columns=["run_name", "file_path", "class", "dataset"])
    existing_runs = set(metadata["run_name"].values)
    seen = set(existing_runs)

    new_entries = []
    for dir_path in data_dirs:
        p = dir_path.rstrip(os.sep).split(os.sep)
        if len(p) < 3 or p[-1] != "align-plots":
            continue
        run_name = p[-3]

        if run_name in seen:
            continue

        seen.add(run_name)

        dataset = "test" if run_name.startswith("unknown") else "train"

        inferred_class = infer_class_from_run_name(run_name)
        new_entries.append({
            "run_name": run_name,
            "file_path": f"results/{run_name}/ref_template_threading_N/align-plots/peptide_signals.tsv",
            "class": inferred_class,
            "dataset": dataset
        })

    if new_entries:
        metadata = pd.concat([metadata, pd.DataFrame(new_entries)], ignore_index=True)

    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    metadata.to_csv(output_metadata_path, index=False)

if __name__ == "__main__":
    input_metadata, output_metadata, *data_dirs = sys.argv[1:]
    update_metadata(input_metadata, output_metadata, data_dirs)
