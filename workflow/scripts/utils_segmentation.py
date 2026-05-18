#!/usr/bin/env python

import pod5 as p5
import numpy as np
import os
import time
import pandas as pd
import logging
import multiprocessing as mp
import threading
from remora import io, refine_signal_map, util
import random
import scipy.ndimage as ndi
from collections import OrderedDict

from utils_plot_segmentation import (
    plot_one_record,
    plot_single_read,
    plot_single_read_alignment,
    plot_single_read_movetable,
    plot_single_read_movetable_panels,
)


logging.getLogger("Remora").setLevel(logging.INFO)
logger = logging.getLogger("nanopore-peptide-classifier")


def _process_chunk_star(args):
    return process_chunk(*args)


def build_category_config(max_plots):
    return {
        "valid": ("peptide-region", max_plots),
        "plr-length-mismatch": ("plr-length-mismatch", 15),
        "signal-too-long": ("signal-too-long", 5),
        "no-signal-data": ("no-signal-data", 5),
        "no-boundary-ref-match": ("no-boundary-ref-match", 10),
        "no-aligned-pairs": ("no-aligned-pairs", 5),
        "empty-query-to-signal": ("empty-query-to-signal", 5),
        "min-alignment-position-filter": ("min-alignment-position-filter", 10),
        "no-candidate-regions": ("no-candidate-regions", 10),
        "no-candidates-after-filtering": ("no-candidates-after-filtering", 10),
        "no-candidate-meets-thresholds": ("no-candidate-meets-thresholds", 20),
        "no-candidate-length-too-small": ("no-candidate-length-too-small", 15),
        "no-candidate-pa-drop-too-low": ("no-candidate-pa-drop-too-low", 20),
        "no-candidate-move-diff-too-low": ("no-candidate-move-diff-too-low", 20),
        "no-candidate-distance-from-dna-end": ("no-candidate-distance-from-dna-end", 15),
        "signal-tail-region": ("signal-tail-region", 15),
        "dna-distribution-check": ("dna-distribution-check", 15),
        "no-peptide": ("no-peptide-region", 15),
        "signal-all-nan": ("signal-all-nan", 5),
    }


def flatten_worker_results(per_worker_results):
    if isinstance(per_worker_results, list) and per_worker_results and isinstance(per_worker_results[0], list):
        return [rec for worker_list in per_worker_results for rec in worker_list]
    return per_worker_results


def group_records_by_category(all_records):
    grouped = {}
    for rec in all_records:
        key = rec.get("category", "unknown")
        grouped.setdefault(key, []).append(rec)
    return grouped


def resolve_category_outputs(groups, outdir, max_plots):
    category_config = build_category_config(max_plots)
    subdirs = {}
    plot_limits = {}

    for category in groups.keys():
        dirname, limit = category_config.get(
            category,
            (category.replace("_", "-"), 10),
        )
        subdirs[category] = os.path.join(outdir, dirname)
        plot_limits[category] = limit
        os.makedirs(subdirs[category], exist_ok=True)

    return subdirs, plot_limits


def collect_records_single_process(read_ids, pod5_input, bam_fh, extend_by, signal_type, min_alignment_position):
    logger.info("DEBUG_SINGLE_PROCESS: running single-process analysis (no mp.Pool)")
    results = []
    with p5.Reader(pod5_input) as reader:
        for read_record in reader.reads(read_ids, missing_ok=True):
            meta = process_read(
                read_record,
                bam_fh,
                extend_by=extend_by,
                signal_type=signal_type,
                min_alignment_position=min_alignment_position,
            )
            if meta:
                results.append(meta)
    return results


def collect_records_multi_process(
    read_ids,
    pod5_input,
    bam_fh,
    seed,
    extend_by,
    signal_type,
    max_valid,
    min_alignment_position,
    chunk_size,
    num_processes,
    time_limit_seconds,
):
    logger.info(f"Using {num_processes} processes for parallel processing.")
    chunks = [read_ids[i:i + chunk_size] for i in range(0, len(read_ids), chunk_size)]

    manager = mp.Manager()
    valid_counter = manager.Value('i', 0)
    stop_event = manager.Event()
    counter_lock = manager.Lock()

    arg_list = [
        (
            chunk,
            pod5_input,
            bam_fh,
            seed,
            extend_by,
            signal_type,
            valid_counter,
            stop_event,
            counter_lock,
            max_valid,
            min_alignment_position,
            None,
            None,
        )
        for chunk in chunks
    ]

    with mp.Pool(num_processes) as pool:
        timer = None
        if time_limit_seconds is not None:
            def _kill_pool():
                logger.info(f"Time limit reached ({time_limit_seconds}s). Terminating worker pool.")
                try:
                    pool.terminate()
                except Exception:
                    pass

            timer = threading.Timer(time_limit_seconds, _kill_pool)
            timer.daemon = True
            timer.start()

        per_worker_results = []
        try:
            for res in pool.imap_unordered(_process_chunk_star, arg_list):
                per_worker_results.append(res)
                if stop_event.is_set():
                    break
        except Exception as exc:
            logger.info(f"Worker pool ended early during imap_unordered: {exc}")
        finally:
            if timer is not None:
                timer.cancel()

    return flatten_worker_results(per_worker_results)


def plot_grouped_records(groups, plot_limits, subdirs, pod5_path, bam_fh, signal_type, num_processes, max_plots):
    for category, records in groups.items():
        limit = plot_limits.get(category, max_plots)
        logger.info(f"Category '{category}': {len(records)} reads")
        n_plot = min(limit, len(records))
        if n_plot == 0:
            continue

        sampled = random.sample(records, k=n_plot)
        out_subdir = subdirs[category]
        plot_args = [(rec, pod5_path, bam_fh, out_subdir, signal_type) for rec in sampled]

        with mp.Pool(processes=num_processes) as plot_pool:
            plot_pool.map(plot_one_record, plot_args)


_global_sig_map_refiner = refine_signal_map.SigMapRefiner(
    kmer_model_filename="9mer_levels_v1.txt",
    do_rough_rescale=False,
    scale_iters=0,
    do_fix_guage=True,
)


class RejectionReason:
    SIGNAL_TOO_LONG = "signal-too-long"
    NO_SIGNAL_DATA = "no-signal-data"
    NO_BOUNDARY_REF_MATCH = "no-boundary-ref-match"
    NO_ALIGNED_PAIRS = "no-aligned-pairs"
    EMPTY_QUERY_TO_SIGNAL = "empty-query-to-signal"
    MIN_ALIGNMENT_POSITION_FILTER = "min-alignment-position-filter"
    NO_CANDIDATE_REGIONS = "no-candidate-regions"
    NO_CANDIDATES_AFTER_FILTERING = "no-candidates-after-filtering"
    NO_CANDIDATE_MEETS_THRESHOLDS = "no-candidate-meets-thresholds"
    NO_CANDIDATE_LENGTH_TOO_SMALL = "no-candidate-length-too-small"
    NO_CANDIDATE_PA_DROP_TOO_LOW = "no-candidate-pa-drop-too-low"
    NO_CANDIDATE_MOVE_DIFF_TOO_LOW = "no-candidate-move-diff-too-low"
    NO_CANDIDATE_DISTANCE_FROM_DNA_END = "no-candidate-distance-from-dna-end"
    SIGNAL_TAIL_REGION = "signal-tail-region"
    DNA_DISTRIBUTION_CHECK = "dna-distribution-check"
    PLR_LENGTH_MISMATCH = "plr-length-mismatch"
    VALID = "valid"


def format_elapsed_time(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s" if minutes > 0 else f"{secs:.1f}s"


def get_read_ids(pod5, read_txt):
    if read_txt:
        return read_ids_from_file(read_txt)
    return read_ids_from_pod5(pod5)


def analyze_candidate_failures(candidates, peptide_min_length, pa_thresh, move_thresh, max_distance):
    if not candidates:
        return RejectionReason.NO_CANDIDATES_AFTER_FILTERING
    best_cand = candidates[0]
    if best_cand["pa_drop"] < pa_thresh:
        return RejectionReason.NO_CANDIDATE_PA_DROP_TOO_LOW
    if best_cand["move_diff"] < move_thresh:
        return RejectionReason.NO_CANDIDATE_MOVE_DIFF_TOO_LOW
    if best_cand["len"] < peptide_min_length:
        return RejectionReason.NO_CANDIDATE_LENGTH_TOO_SMALL
    if max_distance is not None and np.isfinite(best_cand["distance_to_dna_end"]) and best_cand["distance_to_dna_end"] > max_distance:
        return RejectionReason.NO_CANDIDATE_DISTANCE_FROM_DNA_END
    return RejectionReason.NO_CANDIDATE_MEETS_THRESHOLDS


def summarize_and_export(peptide_data, outdir):
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "peptide_signals.tsv")

    cols = [
        "Read_ID",
        "ref_start",
        "ref_end",
        "global_median",
        "global_mad",
        "global_median_excl_plr",
        "global_mad_excl_plr",
        "median_before_plr",
        "mad_before_plr",
        "Signal",
    ]

    if not peptide_data:
        pd.DataFrame(columns=cols).to_csv(outfile, sep="\t", index=False)
        logger.info(f"No PLR signals found. Wrote empty file {outfile}")
        return

    df = pd.DataFrame(peptide_data).drop_duplicates(subset="Read_ID")

    def _sig_to_str(x):
        if isinstance(x, (list, np.ndarray)):
            return " ".join(str(float(v)) for v in x)
        return x

    df["Signal"] = df["Signal"].apply(_sig_to_str)
    for c in ["category", "PLR_start", "PLR_end"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df[cols]
    df.to_csv(outfile, sep="\t", index=False)
    logger.info(f"Wrote {len(df)} PLR signals to {outfile}")


def export_filtering_summary(all_records, groups, outdir, total_input_reads):
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "filtering_summary.tsv")

    filter_order = [
        ("Total Input Reads", None),
        ("Signal Too Long", RejectionReason.SIGNAL_TOO_LONG),
        ("No Signal Data", RejectionReason.NO_SIGNAL_DATA),
        ("No Boundary Ref Match", RejectionReason.NO_BOUNDARY_REF_MATCH),
        ("No Aligned Pairs", RejectionReason.NO_ALIGNED_PAIRS),
        ("Empty Query to Signal", RejectionReason.EMPTY_QUERY_TO_SIGNAL),
        ("Min Alignment Position Filter", RejectionReason.MIN_ALIGNMENT_POSITION_FILTER),
        ("No Candidate Regions", RejectionReason.NO_CANDIDATE_REGIONS),
        ("No Candidates After Filtering", RejectionReason.NO_CANDIDATES_AFTER_FILTERING),
        ("No Candidate Meets Thresholds", RejectionReason.NO_CANDIDATE_MEETS_THRESHOLDS),
        ("No Candidate - Length Too Small", RejectionReason.NO_CANDIDATE_LENGTH_TOO_SMALL),
        ("No Candidate - PA Drop Too Low", RejectionReason.NO_CANDIDATE_PA_DROP_TOO_LOW),
        ("No Candidate - Move Diff Too Low", RejectionReason.NO_CANDIDATE_MOVE_DIFF_TOO_LOW),
        ("No Candidate - Distance from DNA End", RejectionReason.NO_CANDIDATE_DISTANCE_FROM_DNA_END),
        ("Signal Tail Region", RejectionReason.SIGNAL_TAIL_REGION),
        ("DNA Distribution Check", RejectionReason.DNA_DISTRIBUTION_CHECK),
        ("PLR Length Mismatch", RejectionReason.PLR_LENGTH_MISMATCH),
        ("Other/Generic No-Peptide", "no-peptide"),
        ("Signal All NaN", "signal-all-nan"),
        ("Valid Reads", RejectionReason.VALID),
    ]

    summary_data = OrderedDict()
    category_counts = {category: len(records) for category, records in groups.items()}

    total_processed = len(all_records)
    valid_count = category_counts.get(RejectionReason.VALID, 0)
    rejected_count = max(0, total_processed - valid_count)
    not_processed_count = max(0, total_input_reads - total_processed)
    valid_pct = (valid_count / total_input_reads * 100) if total_input_reads > 0 else 0.0

    summary_data["Total Input Reads"] = int(total_input_reads)
    summary_data["Total Processed"] = int(total_processed)
    summary_data["Not Processed"] = int(not_processed_count)
    summary_data["Valid Reads"] = int(valid_count)
    summary_data["Rejected Reads"] = int(rejected_count)

    for label, category_key in filter_order:
        if category_key is None:
            continue
        summary_data[label] = int(category_counts.get(category_key, 0))

    df = pd.DataFrame([{"Filter": k, "Count": v} for k, v in summary_data.items()])

    def _format_summary_pct(row):
        denom = total_input_reads
        if denom <= 0:
            return "0.00%"
        return f"{(row['Count'] / denom * 100):.2f}%"

    df["Percentage"] = df.apply(_format_summary_pct, axis=1)

    df.to_csv(outfile, sep="\t", index=False)
    logger.info(f"Wrote filtering summary to {outfile}")
    logger.info(f"Valid reads: {valid_count}/{total_input_reads} ({valid_pct:.2f}%)")


def get_signal_from_read(read):
    offset = read.calibration.offset
    scale = read.calibration.scale
    signal = read.signal
    pA_signal = scale * (signal + offset)
    pA_signal = np.array(pA_signal)
    return pA_signal


def find_peptide_section_fastdrop(
    sam_input,
    read_record,
    signal_type="pa",
    search_window_half_width_bases=70,
    peptide_min_length=60,
    min_pa_drop_score=75.0,
    min_move_diff_score=15.0,
    short_drop_window=15,
    long_drop_window=35,
    per_sample_score_fraction=0.45,
    merge_gap_samples=20,
    min_alignment_position=None,
    anchor_ref_positions_by_name=(
        ("template", 70),
        ("template_N0_threading", 70),
        ("template_N20_threading", 70),
        ("template_threading_0N", 70),
        ("template_N0_revComTemplate", 70),
        ("template_N0_revCompTemplate", 70),
        ("template_cysteinefree", 99),
        ("template_cysteinefree_revComp", 99),
        ("threading_revComp", 55),
    ),
    baseline_left_fraction=0.25,
    baseline_max_samples=1000,
    min_per_sample_threshold=5.0,
    dna_last_ref_position=60,
    max_distance_from_dna_end_samples=500,
    tail_skip_fraction=0.90,
):

    aln = sam_input.get_first_alignment(str(read_record.read_id))
    io_read = getattr(read_record, "_cached_io_read", None)
    if io_read is None:
        io_read = io.Read.from_pod5_and_alignment(read_record, aln)
        try:
            read_record._cached_io_read = io_read
        except Exception:
            pass

    try:
        io_read.set_refine_signal_mapping(_global_sig_map_refiner, ref_mapping=False)
        io_read.set_refine_signal_mapping(_global_sig_map_refiner, ref_mapping=True)
    except Exception:
        pass

    sig = io_read.get_sig_type("pa")
    if sig is None or np.all(np.isnan(sig)):
        return None, None, sig, None, None, RejectionReason.NO_SIGNAL_DATA

    if aln is None or aln.is_unmapped:
        ref_start = ref_end = None
    else:
        ref_start = aln.reference_start
        ref_end = aln.reference_end

    ref_name = aln.reference_name if aln else None
    boundary_by_ref = dict(anchor_ref_positions_by_name)
    anchor_ref_position = boundary_by_ref.get(ref_name)
    if anchor_ref_position is None:
        return None, None, sig, None, None, RejectionReason.NO_BOUNDARY_REF_MATCH

    if aln.is_reverse:
        logger.debug(f"Read {io_read.read_id} is reverse strand; proceeding cautiously.")

    pairs = aln.get_aligned_pairs(matches_only=True, with_seq=False)
    if not pairs:
        return None, None, sig, None, None, RejectionReason.NO_ALIGNED_PAIRS
    ref2q = {r: q for q, r in pairs}

    try:
        q2s = np.asarray(io_read.query_to_signal, dtype=int)
    except Exception:
        return None, None, sig, None, None, RejectionReason.EMPTY_QUERY_TO_SIGNAL
    if q2s.size == 0:
        return None, None, sig, None, None, RejectionReason.EMPTY_QUERY_TO_SIGNAL

    def _snap_to_q2s(start_idx, end_idx):
        left = q2s[q2s <= int(start_idx)]
        right = q2s[q2s >= int(end_idx)]
        snapped_start = int(left.max()) if left.size else int(start_idx)
        snapped_end = int(right.min()) if right.size else int(end_idx)
        return snapped_start, snapped_end

    def _resolve_anchor():
        if anchor_ref_position in ref2q:
            q_idx = ref2q[anchor_ref_position]
            return int(q2s[q_idx]) if 0 <= q_idx < len(q2s) else None, None

        lower_refs = [r for r in ref2q if r < anchor_ref_position]
        if not lower_refs:
            return None, None
        anchor_prior_ref = max(lower_refs)
        q_idx = ref2q[anchor_prior_ref]
        anchor_idx = int(q2s[q_idx]) if 0 <= q_idx < len(q2s) else None
        return anchor_idx, anchor_prior_ref

    def _resolve_dna_edge_sigs():
        eligible = [r for r in ref2q if r >= dna_last_ref_position]
        if eligible:
            dna_start_ref = min(eligible)
            dna_end_ref = max(eligible)
        else:
            dna_start_ref = min(ref2q)
            dna_end_ref = max(ref2q)

        def _ref_to_sig(ref_pos):
            q_idx = ref2q.get(ref_pos)
            if q_idx is None or not (0 <= q_idx < len(q2s)):
                return None
            return int(q2s[q_idx])

        dna_start_sig = _ref_to_sig(dna_start_ref)
        dna_end_sig = _ref_to_sig(dna_end_ref)
        return dna_start_sig, dna_end_sig

    anchor, anchor_prior_ref = _resolve_anchor()
    if (
        anchor_prior_ref is not None
        and min_alignment_position is not None
        and anchor_prior_ref <= min_alignment_position
    ):
        return None, None, sig, None, None, RejectionReason.MIN_ALIGNMENT_POSITION_FILTER
    dna_start_sig, dna_end_sig = _resolve_dna_edge_sigs()

    if anchor is not None:
        try:
            avg_stride = int(np.median(np.diff(q2s))) if len(q2s) > 3 else 9
        except Exception:
            avg_stride = 9
        search_window_half_width_samples = search_window_half_width_bases * max(1, avg_stride)
        sig_start = max(0, anchor - search_window_half_width_samples)
        sig_end = min(len(sig), anchor + search_window_half_width_samples)
    else:
        sig_start, sig_end = 0, len(sig)

    
    signal_window = sig[sig_start:sig_end]

    # local DNA baseline from the left side of the anchored window
    left_end = min(len(signal_window), max(1, int(len(signal_window) * baseline_left_fraction)))
    baseline_slice = signal_window[max(0, left_end - baseline_max_samples):left_end]
    if baseline_slice.size:
        dna_median = float(np.nanmedian(baseline_slice))
    else:
        dna_median = float(np.nanmedian(signal_window[:min(baseline_max_samples, len(signal_window))]))

    move_levels = None
    try:
        move_levels = _global_sig_map_refiner.extract_levels(util.seq_to_int(io_read.seq))
        move_levels = move_levels * io_read.scale_pa_to_norm + io_read.shift_pa_to_norm
    except Exception:
        move_levels = None

    # combine short and long drop features
    drop = np.maximum(0.0, dna_median - signal_window)
    short_drop = ndi.uniform_filter1d(drop, size=max(1, short_drop_window), mode="nearest")
    long_drop = ndi.uniform_filter1d(drop, size=max(1, long_drop_window), mode="nearest")
    combined_drop = np.maximum(short_drop, long_drop)

    per_sample_score = combined_drop
    per_sample_score = ndi.uniform_filter1d(per_sample_score, size=max(1, int(short_drop_window)), mode="nearest")

    per_sample_threshold = max(min_pa_drop_score * per_sample_score_fraction, min_per_sample_threshold)
    mask = per_sample_score >= per_sample_threshold

    if merge_gap_samples > 1:
        mask = ndi.binary_closing(mask, structure=np.ones(int(merge_gap_samples), dtype=bool))

    labeled, nlab = ndi.label(mask)
    if nlab == 0:
        return None, None, sig, None, None, RejectionReason.NO_CANDIDATE_REGIONS

    candidates = []
    anchors = [s for s in (dna_start_sig, dna_end_sig) if s is not None]
    for lab in range(1, nlab + 1):
        idx = np.flatnonzero(labeled == lab)
        s_abs = sig_start + int(idx[0])
        e_abs = sig_start + int(idx[-1]) + 1
        seg_len = e_abs - s_abs
        seg_q25 = float(np.nanpercentile(sig[s_abs:e_abs], 25))
        pa_drop = float(dna_median - seg_q25)

        move_val = np.nan
        if move_levels is not None:
            center = (s_abs + e_abs) // 2
            q_center = int(np.searchsorted(q2s, center) - 1)
            q_center = max(0, min(len(move_levels) - 1, q_center))
            move_val = float(move_levels[q_center])

        move_diff = max(0.0, move_val - seg_q25) if not np.isnan(move_val) else 0.0

        region_center = (s_abs + e_abs) // 2
        # Use the nearest DNA anchor 
        distance_to_dna_end = min((abs(region_center - s) for s in anchors), default=np.inf)

        candidates.append(
            {
                "s_abs": s_abs,
                "e_abs": e_abs,
                "len": seg_len,
                "pa_drop": pa_drop,
                "move_diff": move_diff,
                "distance_to_dna_end": distance_to_dna_end,
                "score": float(pa_drop + move_diff),
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)

    def _candidate_meets_thresholds(cand, pa_thresh, move_thresh):
        if cand["pa_drop"] < pa_thresh or cand["move_diff"] < move_thresh:
            return False
        if cand["len"] < peptide_min_length:
            return False
        if (
            max_distance_from_dna_end_samples is not None
            and np.isfinite(cand["distance_to_dna_end"])
            and cand["distance_to_dna_end"] > max_distance_from_dna_end_samples
        ):
            return False
        return True

    chosen = next(
        (cand for cand in candidates if _candidate_meets_thresholds(cand, min_pa_drop_score, min_move_diff_score)),
        None,
    )
    if chosen is None:
        rejection_reason = analyze_candidate_failures(
            candidates, peptide_min_length, min_pa_drop_score, min_move_diff_score,
            max_distance_from_dna_end_samples
        )
        return None, None, sig, None, None, rejection_reason

    plr_start, plr_end = _snap_to_q2s(chosen["s_abs"], chosen["e_abs"])

    total_len = len(sig)
    cutoff = int(total_len * tail_skip_fraction)
    if (plr_start >= cutoff or plr_end >= cutoff):
        logger.debug(
            "Detected PLR falls within last %.0f%% of signal (start=%d end=%d total=%d); skipping",
            tail_skip_fraction * 100,
            plr_start,
            plr_end,
            total_len,
        )
        return None, None, sig, None, None, RejectionReason.SIGNAL_TAIL_REGION
    
    dna_left = sig[:plr_start]
    dna_right = sig[plr_end:]

    left_low = 45.0
    right_low = 45.0
    max_out_of_dist_points = 20

    left_out_of_dist = int(np.sum((~np.isfinite(dna_left)) | (dna_left < left_low) ))
    right_out_of_dist = int(np.sum((~np.isfinite(dna_right)) | (dna_right < right_low)))

    if left_out_of_dist > max_out_of_dist_points or right_out_of_dist > max_out_of_dist_points:
        logger.debug(
            "PLR rejected by DNA distribution check: left_out=%d (min=%.1f), right_out=%d (min=%.1f), max_allowed=%d",
            left_out_of_dist,
            left_low,
            right_out_of_dist,
            right_low,
            max_out_of_dist_points,
        )
        return None, None, sig, None, None, RejectionReason.DNA_DISTRIBUTION_CHECK

    # Keep downstream TSV outputs in normalized units regardless of plotting signal mode.
    signal = io_read.get_sig_type("norm")
    return plr_start, plr_end, signal, ref_start, ref_end, None


def read_ids_from_file(reads_txt):
    with open(reads_txt, 'r') as file:
        read_ids = [line.strip() for line in file]
    return read_ids

def read_ids_from_pod5(pod5_input):
    with p5.Reader(pod5_input) as reader:
        read_ids = [str(read_record.read_id) for read_record in reader.reads()]
    return read_ids

def pad_peptide_signals(peptide_signals, max_len=None):
    if max_len is None:
        max_len = max(len(signal) for signal in peptide_signals)  # Find the max length in the dataset

    padded_signals = np.array([np.pad(signal, (0, max_len - len(signal)), 'constant') for signal in peptide_signals])
    return padded_signals
    

def should_skip_signal(signal, read_id, max_length=15000):
    if signal is None or len(signal) > max_length:
        logger.debug(f"Read {read_id} removed since signal over maximum length of {max_length}")
        return True, RejectionReason.SIGNAL_TOO_LONG
    return False, None


def valid_regions(peptide_region, linker_region):
    return (
        peptide_region is not None and len(peptide_region) > 0 and
        linker_region is not None and len(linker_region) > 0
    )

def valid_peptide_linker_length(PLR_len, min_PLR_len=45, max_PLR_len=2000, signal_type='norm'):
    return min_PLR_len <= PLR_len <= max_PLR_len

def handle_valid_read(
    signal, PLR_start, PLR_end,
    read_id, signal_type="norm",
    ref_start=None, ref_end=None,
):
    peptide_signal = signal[PLR_start:PLR_end]

    global_exc = np.concatenate([signal[:PLR_start], signal[PLR_end:]]) if PLR_start or PLR_end < len(signal) else np.array([])
    gm_excl = np.median(global_exc) if global_exc.size else np.nan
    mad_excl = np.median(np.abs(global_exc - gm_excl)) if global_exc.size else np.nan

    bef = signal[:PLR_start] if PLR_start else np.array([])
    mb = np.median(bef) if bef.size else np.nan
    mad_b = np.median(np.abs(bef - mb)) if bef.size else np.nan

    gmed = np.median(signal)
    gmad = np.median(np.abs(signal - gmed)) or 1.0

    return {
        'Read_ID': read_id,
        'category': RejectionReason.VALID,
        'rejection_reason': None,
        'Signal': peptide_signal,
        'global_median': float(gmed),
        'global_mad': float(gmad),
        'global_median_excl_plr': float(gm_excl),
        'global_mad_excl_plr': float(mad_excl),
        'median_before_plr': float(mb),
        'mad_before_plr': float(mad_b),
        'PLR_start': PLR_start,
        'PLR_end': PLR_end,
        'ref_start': ref_start,
        'ref_end': ref_end,
    }

def handle_invalid_read(
    read_id, category, rejection_reason=None, **extra_data
):
    result = {
        'Read_ID': read_id,
        'category': rejection_reason if rejection_reason else category,
        'rejection_reason': rejection_reason if rejection_reason else category,
    }
    result.update(extra_data)
    return result

def process_chunk(read_id_chunk, 
                  pod5_input, 
                  sam_input, 
                  MASTER_SEED,  
                  extend_by, 
                  signal_type, 
                  valid_counter, 
                  stop_event, 
                  counter_lock, 
                  max_valid, 
                  min_alignment_position,
                  time_limit_seconds=None,
                  max_reads_per_chunk=None,
        ):
    seed = MASTER_SEED
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    local_results = []
    start_time = time.time()
    processed_in_chunk = 0
    with p5.Reader(pod5_input) as reader:
        for read_record in reader.reads(read_id_chunk, missing_ok=True):
            if stop_event.is_set():
                break

            meta = process_read(read_record, sam_input, extend_by=extend_by, signal_type=signal_type, min_alignment_position=min_alignment_position)
            if not meta:
                continue

            if meta.get("category") == RejectionReason.VALID:
                with counter_lock:
                    if valid_counter.value >= max_valid:
                        stop_event.set()
                        break
                    valid_counter.value += 1
                    if valid_counter.value >= max_valid:
                        stop_event.set()

            local_results.append(meta)

            processed_in_chunk += 1
            if max_reads_per_chunk is not None and processed_in_chunk >= int(max_reads_per_chunk):
                stop_event.set()
                break

            if time_limit_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed > float(time_limit_seconds):
                    logger.info(f"Worker time limit reached ({elapsed:.1f}s > {time_limit_seconds}s). Stopping worker.")
                    stop_event.set()
                    break

            if stop_event.is_set():
                break

    return local_results



def process_read(read_record, sam_input,
                 min_PLR_len=35,
                 max_PLR_len=5000,
                 extend_by=0,
                 signal_type='norm',
                 min_alignment_position=None,):
    # cache pA signal on the read to avoid repeated conversions
    pA_signal = getattr(read_record, '_cached_pA_signal', None)
    if pA_signal is None:
        pA_signal = get_signal_from_read(read_record)
        try:
            read_record._cached_pA_signal = pA_signal
        except Exception:
            pass
    read_id = str(read_record.read_id)

    should_skip, skip_reason = should_skip_signal(pA_signal, read_id)
    if should_skip:
        return handle_invalid_read(read_id, 'invalid', rejection_reason=skip_reason)

    ref_start = ref_end = None

    if sam_input is not None and getattr(read_record, '_cached_io_read', None) is None:
        try:
            aln_tmp = sam_input.get_first_alignment(str(read_record.read_id))
            io_tmp = io.Read.from_pod5_and_alignment(read_record, aln_tmp)
            try:
                read_record._cached_io_read = io_tmp
            except Exception:
                pass
        except Exception:
            pass

   
    PLR_start, PLR_end, signal, ref_start, ref_end, rejection_reason = find_peptide_section_fastdrop(sam_input, read_record, signal_type=signal_type, min_alignment_position=min_alignment_position)

    if PLR_start is None:
        # Use specific rejection reason if provided by the detection method, otherwise generic 'no-peptide'
        return handle_invalid_read(read_id, 'no-peptide', rejection_reason=rejection_reason if rejection_reason else 'no-peptide')

    if np.all(np.isnan(signal)):
        return handle_invalid_read(read_id, 'no-peptide', rejection_reason='signal-all-nan')

    plr_len = PLR_end - PLR_start
    if not valid_peptide_linker_length(plr_len, min_PLR_len, max_PLR_len):
        return {
            'Read_ID': read_id,
            'category': RejectionReason.PLR_LENGTH_MISMATCH,
            'rejection_reason': RejectionReason.PLR_LENGTH_MISMATCH,
            'plr_len': plr_len,
            'ref_start': ref_start,
            'ref_end': ref_end,
        }

    # extend the PLR region if needed
    PLR_start, PLR_end = extend_PLR(PLR_start, PLR_end, extend_by=extend_by)

    return handle_valid_read(signal, PLR_start, PLR_end, read_id, signal_type=signal_type, ref_start=ref_start, ref_end=ref_end)

def extend_PLR(PLR_start, PLR_end, extend_by=0):
    PLR_start = max(0, PLR_start - extend_by)
    PLR_end = PLR_end + extend_by
    return PLR_start, PLR_end


