#!/usr/bin/env python

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import bessel, butter, filtfilt, resample_poly, savgol_filter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import QuantileTransformer

try:
    import polars as pl
except Exception:
    pl = None

from utils_feature_extraction import is_monotonic_down_up, segment_signal

logger = logging.getLogger("nanopore-peptide-classifier")
logger.addHandler(logging.NullHandler())

_LOAD_FILE_CACHE = {}
_PARSED_FILE_CACHE = {}


def trim_padding(sig, pad_value=0.0):
    if sig is None:
        return np.array([])
    s = np.asarray(sig)
    if s.size == 0:
        return s
    neq = np.where(np.abs(s - pad_value) > 1e-12)[0]
    if neq.size == 0:
        return np.array([])
    first, last = neq[0], neq[-1]
    return s[first : last + 1]


def calculate_log_likelihood(data, start, end):
    segment = data[start : end + 1]
    mean = np.mean(segment)
    variance = np.var(segment)
    n = len(segment)
    if variance == 0:
        variance = 1e-6  # Avoid division by zero
    log_likelihood = -n * np.log(2 * np.pi * variance) / 2 - n / (
        2 * variance
    ) * np.sum((segment - mean) ** 2)
    return log_likelihood


def compute_train_stats(
    df_train,
    signals_trimmed,
    method="MAD-global",
    quantile_samples=200,
    qt_resample_len=128,
):
    stats = {}

    flat = (
        np.concatenate([s for s in signals_trimmed if s.size > 0])
        if len(signals_trimmed)
        else np.array([])
    )

    # global median/mad: prefer df columns 'global_median' / 'global_mad' aggregated if present
    if "global_median" in df_train.columns and "global_mad" in df_train.columns:
        try:
            stats["global_median"] = float(df_train["global_median"].median())
            stats["global_mad"] = float(
                df_train["global_mad"].median()
                if df_train["global_mad"].median() > 0
                else 1.0
            )
        except Exception:
            # fallback to flatten
            if flat.size:
                med = float(np.median(flat))
                mad = float(np.median(np.abs(flat - med)) or 1.0)
                stats["global_median"] = med
                stats["global_mad"] = mad
            else:
                stats["global_median"] = 0.0
                stats["global_mad"] = 1.0
    else:
        if flat.size:
            med = float(np.median(flat))
            mad = float(np.median(np.abs(flat - med)) or 1.0)
            stats["global_median"] = med
            stats["global_mad"] = mad
        else:
            stats["global_median"] = 0.0
            stats["global_mad"] = 1.0

    # If df has pre-PLR aggregates, use medians of those
    if (
        "global_median_excl_plr" in df_train.columns
        and "global_mad_excl_plr" in df_train.columns
    ):
        try:
            stats["global_median_excl_plr"] = float(
                df_train["global_median_excl_plr"].median()
            )
            gm = float(df_train["global_mad_excl_plr"].median())
            stats["global_mad_excl_plr"] = float(gm if gm > 0 else stats["global_mad"])
        except Exception:
            stats["global_median_excl_plr"] = stats["global_median"]
            stats["global_mad_excl_plr"] = stats["global_mad"]
    else:
        # fallback to same as global
        stats["global_median_excl_plr"] = stats["global_median"]
        stats["global_mad_excl_plr"] = stats["global_mad"]

    # percentiles for minmax-global
    if flat.size:
        stats["p_lo"] = float(np.percentile(flat, 0.1))
        stats["p_hi"] = float(np.percentile(flat, 99.9))
        stats["all_values_flat"] = flat  # optional, can be big
    else:
        stats["p_lo"] = -1.0
        stats["p_hi"] = 1.0
        stats["all_values_flat"] = flat

    if method == "quantile-global":
        n_samples = min(len(signals_trimmed), quantile_samples)
        if n_samples > 0:
            chosen = np.linspace(0, len(signals_trimmed) - 1, num=n_samples, dtype=int)
            sample_mat = np.stack(
                [_resample_to_len(signals_trimmed[i], qt_resample_len) for i in chosen],
                axis=0,
            )
            qt = QuantileTransformer(output_distribution="normal", random_state=42)
            qt.fit(sample_mat)  # shape (n_samples, qt_resample_len)
            stats["quantile_transformer"] = qt
        else:
            stats["quantile_transformer"] = None

    stats["lengths"] = [len(s) for s in signals_trimmed]
    stats["median_length"] = (
        int(np.median(stats["lengths"])) if len(stats["lengths"]) else 0
    )
    return stats


def _ensure_1d_float(x):
    a = np.asarray(x).ravel().astype(float)
    return a


def _resample_to_len(sig, target_len):
    a = _ensure_1d_float(sig)
    if a.size == 0:
        return np.zeros(target_len, dtype=float)
    if a.size == target_len:
        return a
    try:
        # resample_poly: up=target_len, down=len(a) approximates resampling
        out = resample_poly(a, up=target_len, down=len(a))
        # trim/pad
        if out.size >= target_len:
            return out[:target_len].astype(float)
        out2 = np.zeros(target_len, dtype=float)
        out2[: out.size] = out
        return out2
    except Exception:
        # linear interp fallback
        xp = np.linspace(0.0, 1.0, len(a))
        x = np.linspace(0.0, 1.0, target_len)
        return np.interp(x, xp, a).astype(float)


def plot_step(smoothed_df, steps_df, output):
    plt.figure(figsize=(10, 6))

    # Plot original and smoothed signal
    plt.plot(smoothed_df["Signal"], label="Smoothed Signal", color="blue")
    plt.plot(steps_df["_steps"], label="Detected Steps", color="red", linestyle="--")

    plt.title(f"Signal and Detected Steps for")
    plt.xlabel("Index")
    plt.ylabel("Signal Value")
    plt.legend()

    # Save plot
    plt.savefig(os.path.join(output, f"steps_plot.png"))
    plt.close()


def moving_average_filter(signal, window_size=5):
    signal = np.asarray(signal).flatten()
    return np.convolve(signal, np.ones(window_size) / window_size, mode="valid")


def low_pass_butter_filter(signal, cutoff=0.1, fs=1.0, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def low_pass_bessel_filter(signal, cutoff=0.05, fs=1.0, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = bessel(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def apply_smoothing(
    X,
    outdir,
    window_length=21,
    smoothing_method="",
    window_size=31,
    cutoff=0.1,
    fs=1.0,
    order=4,
):

    os.makedirs(outdir, exist_ok=True)
    outdir = os.path.join(outdir, "smoothed_signals")
    os.makedirs(outdir, exist_ok=True)

    smoothed_signals = []
    for idx, signal in enumerate(X):
        if smoothing_method == "savitzky-golay":
            smoothed_signal = savgol_filter(
                signal, window_length=window_length, polyorder=order
            )
        if smoothing_method == "moving-average":
            smoothed_signal = moving_average_filter(signal, window_size)
        if smoothing_method == "low-pass-butter":
            smoothed_signal = low_pass_butter_filter(signal, cutoff, fs, order)
        if smoothing_method == "low-pass-bessel":
            smoothed_signal = low_pass_bessel_filter(signal, cutoff, fs, order)
        else:
            smoothed_signal = signal

        smoothed_signals.append(smoothed_signal)

        if idx < 5:
            plt.figure(figsize=(10, 6))
            plt.plot(signal.flatten(), label="Original Signal", alpha=0.7)
            plt.plot(
                smoothed_signal.flatten(),
                label=f"Smoothed Signal ({smoothing_method})",
                alpha=0.7,
            )
            plt.legend()
            plt.title(
                f"Signal {idx+1} Before and After Smoothing (Method: {smoothing_method})"
            )
            plt.xlabel("Time")
            plt.ylabel("pA")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                f"{outdir}/smoothed_signal_{smoothing_method}_signal{idx+1}.png"
            )
            plt.close()
    return smoothed_signals


def parse_peptide_signal(signal_str):
    try:
        if not isinstance(signal_str, str) or signal_str == "":
            return None
        arr = np.fromstring(signal_str, sep=" ")
        return arr if arr.size > 0 else None
    except ValueError:
        return None


def _parse_signal_chunk(values):
    return [parse_peptide_signal(v) for v in values]


def parallel_apply(series, func, n_jobs=None, chunk_size=4096):
    vals = list(series)
    if not vals:
        return []

    if func is not parse_peptide_signal:
        return [func(x) for x in vals]

    if n_jobs is None:
        n_jobs = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    if n_jobs <= 1 or len(vals) < chunk_size:
        return [parse_peptide_signal(v) for v in vals]

    chunks = [vals[i : i + chunk_size] for i in range(0, len(vals), chunk_size)]
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        parsed_chunks = list(ex.map(_parse_signal_chunk, chunks))
    return [v for chunk in parsed_chunks for v in chunk]


def _read_signal_table(input_file, nrows, read_engine):
    engine = (read_engine or "pandas").strip().lower()

    if engine in {"polars", "auto"} and pl is not None:
        try:
            df_pl = pl.read_csv(
                input_file,
                separator="\t",
                n_rows=nrows,
                infer_schema_length=1000,
            )
            return df_pl.to_pandas()
        except Exception as exc:
            logger.debug("Polars read fallback for %s: %s", input_file, exc)

    if engine in {"pyarrow", "auto"}:
        try:
            return pd.read_csv(input_file, sep="\t", nrows=nrows, engine="pyarrow")
        except Exception as exc:
            logger.debug("PyArrow read fallback for %s: %s", input_file, exc)

    return pd.read_csv(input_file, sep="\t", nrows=nrows, memory_map=True)


def extract_run_id(input_file):
    # Take the directory right after 'results'
    part = Path(input_file).parts[1]
    # Extract leading digits before dash or underscore
    match = re.match(r"^(\d+)[-_]", part)
    return match.group(1) if match else "unknown"


def load_file(args):
    (
        input_file,
        class_label,
        testing,
        min_alignment_position,
        parse_jobs,
        read_engine,
    ) = args
    nrows = int(testing) if testing and testing > 0 else None
    read_engine = (read_engine or "pandas").strip().lower()
    cache_key = (input_file, class_label, nrows, min_alignment_position)
    cached = _PARSED_FILE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy(deep=False)

    try:
        raw_cache_key = (input_file, nrows, read_engine)
        raw_cached = _LOAD_FILE_CACHE.get(raw_cache_key)
        if raw_cached is None:
            df = _read_signal_table(input_file, nrows=nrows, read_engine=read_engine)
            _LOAD_FILE_CACHE[raw_cache_key] = df
        else:
            df = raw_cached.copy(deep=False)

        df["Class"] = class_label
        run_id = extract_run_id(input_file)
        df["Run_ID"] = run_id

        if "Signal" in df.columns:
            df = df[df["Signal"].notna() & (df["Signal"] != "")]

        if (
            min_alignment_position is not None
            and "ref_end" in df.columns
            and "ref_start" in df.columns
        ):
            df = df[(df["ref_end"] >= min_alignment_position) & (df["ref_start"] <= 30)]

        df["Signal"] = parallel_apply(
            df["Signal"], parse_peptide_signal, n_jobs=parse_jobs
        )
        df = df.dropna(subset=["Signal"])

        _PARSED_FILE_CACHE[cache_key] = df

        logger.debug(
            f"Loaded {len(df)} samples from {input_file} with class {class_label}"
        )
        return df.copy(deep=False)
    except Exception as e:
        logger.error(f"Could not load file {input_file}: {e}")
        return None


def process_parse_inputfiles(
    input_files,
    classes,
    output,
    dataset_stats,
    testing=0,
    seed=385,
    min_alignment_position=None,
    read_engine="pandas",
):
    logger.info("Loading training / testing data...")

    total_workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    n_workers = max(1, min(len(input_files), total_workers))
    per_file_parse_jobs = max(1, total_workers // n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        all_data = list(
            executor.map(
                load_file,
                [
                    (
                        f,
                        c,
                        testing,
                        min_alignment_position,
                        per_file_parse_jobs,
                        read_engine,
                    )
                    for f, c in zip(input_files, classes)
                ],
            )
        )

    all_data = [df for df in all_data if df is not None and len(df) > 0]

    if not all_data:
        return None

    # Concatenate all loaded data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Drop duplicate Read_IDs
    combined_df = combined_df.drop_duplicates(subset="Read_ID")

    # Subsample if testing is set, without groupby.apply (faster, avoids FutureWarning).
    if testing > 0:
        logger.info(f"Subsampling {testing} samples per class for testing.")
        sampled = []
        for _, grp in combined_df.groupby("Class", sort=False):
            sampled.append(grp.sample(n=min(len(grp), testing), random_state=seed))
        combined_df = (
            pd.concat(sampled, ignore_index=True) if sampled else combined_df.iloc[0:0]
        )

    # Signals are already parsed in load_file and cached.

    # Use categorical dtype for efficiency
    combined_df["Class"] = combined_df["Class"].astype("category")

    # Print class distribution
    logger.info("Samples per class:")
    logger.info(combined_df["Class"].value_counts())

    # Optional plotting
    if dataset_stats:
        plot_simple_stats(combined_df, output, filename="total")

    return combined_df


def plot_simple_stats(input_df, output_dir, filename="train"):
    input_df["Signal_Length"] = input_df["Signal"].apply(len)
    input_df["Signal_Max"] = input_df["Signal"].apply(max)
    input_df["Signal_Min"] = input_df["Signal"].apply(min)
    input_df["Signal_Median"] = input_df["Signal"].apply(
        lambda x: pd.Series(x).median()
    )

    stats_to_plot = ["Signal_Length", "Signal_Max", "Signal_Min", "Signal_Median"]

    for stat in stats_to_plot:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=input_df, x="Class", y=stat, palette="muted")
        plt.title(f"{stat} Distribution by Class")
        plt.xlabel("Class")
        plt.ylabel(stat)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{stat}_violin_plot_{filename}.png")
        plt.savefig(plot_path)
        plt.close()

    stats_summary = input_df.groupby("Class")[stats_to_plot].agg(
        ["min", "median", "max"]
    )
    stats_summary_path = os.path.join(output_dir, "stats_summary.tsv")
    stats_summary.to_csv(stats_summary_path, sep="\t")

    logger.info(f"Plots and stats summary saved in {output_dir}")


def percentile_pad_signals(
    signals,
    classes,
    method="percentile",
    fixed_length=1000,
    percentile=90,
    padding_value=0,
    remove_outlier=False,
):
    if method == "percentile":
        target_length = int(
            np.percentile([len(signal) for signal in signals], percentile)
        )
    elif method == "mean":
        target_length = int(np.mean([len(signal) for signal in signals]))
    elif method == "median":
        target_length = int(np.median([len(signal) for signal in signals]))
    elif method == "fixed":
        if fixed_length is None:
            raise ValueError(
                "You must provide a fixed length when using the 'fixed' method."
            )
        target_length = int(fixed_length)
    else:
        raise ValueError(
            "Invalid method. Choose from 'percentile', 'mean', 'median', or 'fixed'."
        )

    # Filter out signals longer than the target length if remove_outlier is True
    if remove_outlier:
        # Keep only signals and classes with length <= target_length
        filtered_data = [
            (signal, cls)
            for signal, cls in zip(signals, classes)
            if len(signal) <= target_length
        ]
        signals, classes = zip(*filtered_data) if filtered_data else ([], [])

    logger.debug("Padding signals...")
    logger.debug("Target Length: " + str(target_length))
    logger.debug("Used method: " + str(method))
    # Pad or trim each signal to the target length
    padded_signals = np.array(
        [
            np.pad(
                signal[:target_length],
                (0, max(0, target_length - len(signal))),
                constant_values=padding_value,
            )
            for signal in signals
        ]
    )

    return padded_signals, np.array(classes)


def normalize_signal(df, option="MAD-global"):
    if option == "MAD-global":
        # use df columns MAD-global MAD-median
        return [
            (np.array(signal) - row_median) / (row_mad if row_mad > 0 else 1)
            for signal, row_median, row_mad in zip(
                df["Signal"], df["global_median"], df["global_mad"]
            )
        ]
    elif option == "MAD-local":
        # Normalize each signal using its own median and MAD
        normalized_signals = []
        for signal in df["Signal"]:
            signal_array = np.array(signal)
            local_median = np.median(signal_array)
            mad = np.median(np.abs(signal_array - local_median))
            local_mad = mad if mad > 0 else 1
            normalized_signals.append((signal_array - local_median) / local_mad)
        return normalized_signals
    else:
        return [np.array(signal) for signal in df["Signal"]]


def prepare_data(
    df,
    norm_method=None,
    padding_method="percentile",
    fixed_length=1000,
    percentile=90,
    padding_value=0,
    remove_outlier=False,
):
    signals = normalize_signal(df, norm_method)
    classes = df["Class"].values

    padded_signals, classes = percentile_pad_signals(
        signals,
        classes,
        padding_method,
        fixed_length,
        percentile,
        padding_value,
        remove_outlier,
    )

    X = padded_signals[:, np.newaxis, :]
    y = classes

    logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")
    return np.array(X), np.array(y)


def normalize_signals(
    signals, df=None, method="MAD-local", train_stats=None, eps=1e-12
):
    df_present = df is not None
    normed = []
    offsets = []
    scales = []
    frac_clipped_list = []

    def _get_df_val(i, col, default=np.nan):
        if not df_present or col not in df.columns:
            return default
        v = df.iloc[i][col]
        return (
            v
            if (v is not None and not (isinstance(v, float) and np.isnan(v)))
            else default
        )

    default_k = 6.0
    k_global = None
    k_lo_global = None
    k_hi_global = None
    mad_floor_global = None
    if train_stats is not None:
        k_global = train_stats.get("minmax_k", None)
        k_lo_global = train_stats.get("minmax_k_lo", None)
        k_hi_global = train_stats.get("minmax_k_hi", None)
        mad_floor_global = train_stats.get("mad_floor", None)
        # fallback mad floor: very small fraction of global mad
        if mad_floor_global is None and "global_mad" in train_stats:
            mad_floor_global = float(train_stats["global_mad"]) * 1e-4

    for i, s in enumerate(signals):
        a = np.asarray(s, dtype=float)
        if a.size == 0:
            normed.append(a)
            offsets.append(np.nan)
            scales.append(np.nan)
            continue

        if method in (None, "None"):
            normed.append(a)
            offsets.append(0.0)
            scales.append(1.0)
            continue

        if method == "minmax-decile-median":
            n_seg = min(10, a.size)
            # split keeping order; np.array_split returns lists of subarrays
            segs = np.array_split(a, n_seg)
            seg_mins = []
            seg_maxs = []
            for seg in segs:
                # skip segments that are all-NaN
                if seg.size == 0 or np.all(np.isnan(seg)):
                    continue
                seg_mins.append(float(np.nanmin(seg)))
                seg_maxs.append(float(np.nanmax(seg)))
            # If no valid segments (shouldn't happen because we've checked all-NaN earlier),
            # fallback to local min/max
            if len(seg_mins) == 0 or len(seg_maxs) == 0:
                lo = float(np.nanmin(a))
                hi = float(np.nanmax(a))
            else:
                lo = float(np.nanmedian(np.array(seg_mins, dtype=float)))
                hi = float(np.nanmedian(np.array(seg_maxs, dtype=float)))

            # safety: ensure finite and non-zero span
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
                hi = lo + max(1.0, abs(lo) * 1e-3)

            clipped = np.clip(a, lo, hi)
            scaled = (clipped - lo) / (hi - lo + eps)
            num_clipped = float(((a < lo) | (a > hi)).sum())
            frac_clipped = num_clipped / float(a.size)
            normed.append(scaled)
            offsets.append(lo)
            scales.append(hi - lo)
            frac_clipped_list.append(frac_clipped)
            continue

        if method == "MAD-local":
            m = np.nanmedian(a)
            mad = np.nanmedian(np.abs(a - m))
            if mad <= 0:
                mad = 1.0
            normed.append((a - m) / mad)
            offsets.append(m)
            scales.append(mad)
            continue

        if method == "MAD-before-plr":
            med_before = _get_df_val(i, "median_before_plr", np.nan)
            mad_before = _get_df_val(i, "mad_before_plr", np.nan)
            # if missing use global_excl_plr or fallback to local
            if np.isnan(med_before) or np.isnan(mad_before):
                if train_stats is not None:
                    med_before = train_stats.get(
                        "global_median_excl_plr", train_stats.get("global_median", 0.0)
                    )
                    mad_before = train_stats.get(
                        "global_mad_excl_plr", train_stats.get("global_mad", 1.0)
                    )
                else:
                    med_before = np.nanmedian(a)
                    mad_before = np.nanmedian(np.abs(a - med_before))
            if mad_before <= 0:
                mad_before = 1.0
            normed.append((a - med_before) / mad_before)
            offsets.append(med_before)
            scales.append(mad_before)
            continue

        if method == "min-max-before-plr":
            # get pre-PLR median/mad (per-row preferred)
            med_before = _get_df_val(i, "median_before_plr", np.nan)
            mad_before = _get_df_val(i, "mad_before_plr", np.nan)

            # fallback: use train_stats global excl plr if available, else local median/mad
            if np.isnan(med_before) or np.isnan(mad_before):
                if train_stats is not None:
                    med_before = train_stats.get(
                        "global_median_excl_plr",
                        train_stats.get("global_median", np.nan),
                    )
                    mad_before = train_stats.get(
                        "global_mad_excl_plr", train_stats.get("global_mad", np.nan)
                    )
                # final fallback to local
                if np.isnan(med_before):
                    med_before = np.nanmedian(a)
                if np.isnan(mad_before):
                    mad_before = np.nanmedian(np.abs(a - med_before))

            # enforce mad floor
            mad_floor = mad_floor_global if mad_floor_global is not None else 1e-6
            if mad_before <= mad_floor:
                mad_before = mad_floor

            # choose k values (lo/hi). Priority: train_stats minmax_k_lo/minmax_k_hi -> train_stats minmax_k -> default_k
            k_lo = (
                k_lo_global
                if (k_lo_global is not None)
                else (k_global if k_global is not None else default_k)
            )
            k_hi = (
                k_hi_global
                if (k_hi_global is not None)
                else (k_global if k_global is not None else default_k)
            )

            # compute lo/hi bounds using median +/- k * mad
            lo = float(med_before - k_lo * mad_before)
            hi = float(med_before + k_hi * mad_before)

            # safety: if hi==lo or hi<lo, adjust slightly
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
                # fallback to symmetric small window around median
                hi = med_before + max(1.0, abs(med_before) * 1e-3)
                lo = med_before - max(1.0, abs(med_before) * 1e-3)

            # optional further fallback: if train_stats provides p_lo/p_hi for pre-PLR, prefer those
            if (
                train_stats is not None
                and "preplr_p_lo" in train_stats
                and "preplr_p_hi" in train_stats
            ):
                # Use train-level percentiles as absolute mapping (overrides per-read)
                tlo = float(train_stats["preplr_p_lo"])
                thi = float(train_stats["preplr_p_hi"])
                if np.isfinite(tlo) and np.isfinite(thi) and (thi - tlo) > eps:
                    lo, hi = tlo, thi

            # clip and scale to [0,1]
            clipped = np.clip(a, lo, hi)
            scaled = (clipped - lo) / (hi - lo + eps)
            # record fraction clipped for diagnostics
            num_clipped = float(((a < lo) | (a > hi)).sum())
            frac_clipped = num_clipped / float(a.size)
            normed.append(scaled)
            offsets.append(lo)
            scales.append(hi - lo)
            frac_clipped_list.append(frac_clipped)
            continue

        if method == "median-before-plr":
            med_before = _get_df_val(i, "median_before_plr", np.nan)
            if np.isnan(med_before):
                med_before = (
                    train_stats.get("global_median_excl_plr", np.nanmedian(a))
                    if train_stats is not None
                    else np.nanmedian(a)
                )
            normed.append(a - med_before)
            offsets.append(med_before)
            scales.append(np.nan)
            continue

        if method == "MAD-global":
            if train_stats is None:
                raise ValueError("train_stats required for MAD-global")
            med = train_stats.get("global_median", 0.0)
            mad = train_stats.get("global_mad", 1.0)
            normed.append((a - med) / (mad if mad > 0 else 1.0))
            offsets.append(med)
            scales.append(mad)
            continue

        if method == "minmax-global":
            if train_stats is None:
                raise ValueError("train_stats required for minmax-global")
            lo = train_stats.get("p_lo", None)
            hi = train_stats.get("p_hi", None)
            if lo is None or hi is None:
                raise ValueError("train_stats missing p_lo/p_hi for minmax-global")
            clipped = np.clip(a, lo, hi)
            normed.append((clipped - lo) / (hi - lo + eps))
            offsets.append(lo)
            scales.append(hi - lo)
            continue

        if method == "quantile-global":
            if train_stats is None or "quantile_transformer" not in train_stats:
                raise ValueError(
                    "quantile_global requires train_stats['quantile_transformer']"
                )
            qt = train_stats["quantile_transformer"]
            # we must resample the series to the qt.resample length used to fit; assume qt was fitted on fixed-length vectors
            target_len = qt.n_quantiles_ if hasattr(qt, "n_quantiles_") else 128
            a_res = _resample_to_len(a, target_len)
            transformed = qt.transform(a_res.reshape(1, -1)).ravel()
            normed.append(transformed)
            offsets.append(np.nan)
            scales.append(np.nan)
            continue

        # fallback to MAD-local
        m = np.nanmedian(a)
        mad = np.nanmedian(np.abs(a - m))
        if mad <= 0:
            mad = 1.0
        normed.append((a - m) / mad)
        offsets.append(m)
        scales.append(mad)

    info = {"offsets": offsets, "scales": scales, "method": method}
    return normed, info


def _safe_array(x):
    return np.asarray(x, dtype=float) if x is not None else np.array([], dtype=float)


def resample_center_weighted(a, target_length, center_portion=0.6):
    N = len(a)
    if N == target_length:
        return a.copy()
    # create a sampling density that is higher in the center
    x = np.linspace(-1, 1, N)  # -1..1, center at 0
    # shape a center-peaked distribution; alpha controls sharpness
    alpha = max(2.0, 5.0 * (1 - center_portion))  # less center_portion -> sharper alpha
    density = np.exp(-(x**2) * alpha)
    density /= density.sum()
    cdf = np.cumsum(density)
    target_positions = np.linspace(0.0, 1.0, target_length)
    src_pos = np.interp(target_positions, cdf, np.arange(N) / (N - 1))
    # linear interpolation from a
    resampled = np.interp(src_pos * (N - 1), np.arange(N), a)
    return resampled


def pad_trim_or_resample(
    signals,
    target_length,
    mode="truncate_or_pad",
    pad_values=None,
    downsample_if_longer=False,
    return_kept_idx=True,
    return_length_feature=True,
):
    if target_length <= 0:
        raise ValueError("target_length must be > 0")

    allowed_modes = {
        "truncate_or_pad",
        "pad_only",
        "trim_only",
        "strict",
        "resample",
        "resample_if_longer",
        "resample_center",
    }
    if downsample_if_longer:
        mode = "resample_if_longer"
    if mode not in allowed_modes:
        raise ValueError(f"Unknown mode '{mode}'. Allowed: {allowed_modes}")

    N = len(signals)
    pad_vals = None
    if pad_values is None:
        pad_vals = [0.0] * N
    elif np.isscalar(pad_values):
        pad_vals = [float(pad_values)] * N
    else:
        pad_vals = list(pad_values)
        if len(pad_vals) < N:
            # pad the pad_values list with last value
            pad_vals = pad_vals + [pad_vals[-1]] * (N - len(pad_vals))

    kept_idx = []
    kept_lengths = []
    rows = []

    for i, s in enumerate(signals):
        a = _safe_array(s)
        orig_len = a.size

        if mode == "resample_center":
            out = resample_center_weighted(a, target_length, center_portion=0.6)
            rows.append(out)
            kept_idx.append(i)
            kept_lengths.append(orig_len)
            continue

        if mode == "resample":
            # resample everything
            out = resample_signal_to_length(a, target_length)
            rows.append(out)
            kept_idx.append(i)
            kept_lengths.append(orig_len)
            continue

        if mode == "resample_if_longer":
            if orig_len > target_length:
                out = resample_signal_to_length(a, target_length)
                rows.append(out)
                kept_idx.append(i)
                kept_lengths.append(orig_len)
                continue

        if orig_len == target_length:
            rows.append(a.copy())
            kept_idx.append(i)
            kept_lengths.append(orig_len)
            continue

        if orig_len < target_length:
            if mode in ("truncate_or_pad", "pad_only", "resample_if_longer"):
                pad_val = pad_vals[i] if i < len(pad_vals) else 0.0
                out = np.empty(target_length, dtype=float)
                out[:orig_len] = a
                out[orig_len:] = pad_val
                rows.append(out)
                kept_idx.append(i)
                kept_lengths.append(orig_len)
            else:
                continue
        else:
            if mode in ("truncate_or_pad", "trim_only"):
                out = a[:target_length].copy()
                rows.append(out)
                kept_idx.append(i)
                kept_lengths.append(orig_len)
            else:
                # pad_only or strict -> drop
                continue

    if len(rows) == 0:
        X = np.zeros((0, target_length), dtype=float)
    else:
        X = np.vstack(rows).astype(float)

    if return_kept_idx and return_length_feature:
        return X, kept_idx, np.array(kept_lengths, dtype=int)
    elif return_kept_idx:
        return X, kept_idx, None
    elif return_length_feature:
        return X, None, np.array(kept_lengths, dtype=int)
    else:
        return X, None, None


def resample_signal_to_length(sig, target_len):
    a = _safe_array(sig).ravel()
    if target_len <= 0:
        raise ValueError("target_len must be > 0")
    if a.size == 0:
        return np.zeros(target_len, dtype=float)
    L = len(a)
    if L == target_len:
        return a.copy()
    up = target_len
    down = L
    g = gcd(up, down)
    up //= g
    down //= g
    try:
        res = resample_poly(a, up, down)
    except Exception:
        x_old = np.linspace(0, 1, L)
        x_new = np.linspace(0, 1, target_len)
        res = np.interp(x_new, x_old, a)
    if len(res) > target_len:
        res = res[:target_len]
    elif len(res) < target_len:
        res = np.pad(res, (0, target_len - len(res)), mode="edge")
    return res


def _length_stats_from_lengths(lengths):
    """Return a dict of descriptive stats for a 1D integer array of lengths."""
    if len(lengths) == 0:
        return {
            "count": 0,
            "min": np.nan,
            "p1": np.nan,
            "p5": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(len(lengths)),
        "min": int(np.min(lengths)),
        "p1": float(np.percentile(lengths, 1)),
        "p5": float(np.percentile(lengths, 5)),
        "p25": float(np.percentile(lengths, 25)),
        "median": float(np.median(lengths)),
        "mean": float(np.mean(lengths)),
        "p75": float(np.percentile(lengths, 75)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(np.max(lengths)),
    }


def log_length_distributions(
    df, df_name="dataset", class_col="Class", signal_col="Signal", out_csv=None
):
    lengths = []
    for i, row in df.iterrows():
        sig = row.get(signal_col, None)
        if sig is None:
            lengths.append(0)
        else:
            try:
                trimmed = trim_padding(np.asarray(sig))
                lengths.append(len(trimmed))
            except Exception:
                # fallback: try len on raw
                try:
                    lengths.append(len(sig))
                except Exception:
                    lengths.append(0)

    lengths = np.array(lengths, dtype=int)
    overall = _length_stats_from_lengths(lengths)

    per_class_rows = []
    if class_col in df.columns:
        grouped = df.groupby(class_col, observed=False)
        for cls, group in grouped:
            group_lengths = []
            for i, row in group.iterrows():
                sig = row.get(signal_col, None)
                if sig is None:
                    group_lengths.append(0)
                else:
                    try:
                        trimmed = trim_padding(np.asarray(sig))
                        group_lengths.append(len(trimmed))
                    except Exception:
                        try:
                            group_lengths.append(len(sig))
                        except Exception:
                            group_lengths.append(0)
            cl_arr = np.array(group_lengths, dtype=int)
            stats = _length_stats_from_lengths(cl_arr)
            stats["class"] = cls
            per_class_rows.append(stats)
    else:
        logger.info(
            "No class column '%s' found in DataFrame; skipping per-class breakdown.",
            class_col,
        )

    if out_csv is not None:
        try:
            per_class_df = pd.DataFrame(per_class_rows).set_index("class")
            per_class_df.to_csv(out_csv)
            logger.info("Wrote per-class length stats to %s", out_csv)
        except Exception as e:
            logger.warning("Could not write per-class CSV to %s: %s", out_csv, e)

    return {"overall": overall, "per_class": per_class_rows}


def onehot_encode_labels(y_int, n_classes):
    y = np.asarray(y_int, dtype=int)
    oh = np.zeros((len(y), n_classes), dtype=float)
    for i, v in enumerate(y):
        if 0 <= v < n_classes:
            oh[i, v] = 1.0
    return oh


def evaluate_model_from_arrays(y_true_labels, y_pred_labels, dataset_name="Validation"):

    y_true = np.asarray(y_true_labels)
    y_pred = np.asarray(y_pred_labels)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    labels = (
        np.unique(np.concatenate([y_true, y_pred])) if y_true.size else np.array([])
    )
    cm = (
        confusion_matrix(y_true, y_pred, labels=labels)
        if labels.size
        else np.array([[]])
    )
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "Classification Report": report,
        "Confusion Matrix": cm,
    }


def resample_signals_to_length(signals, target_len, n_jobs=1):
    out = np.zeros((len(signals), target_len), dtype=float)
    for i, s in enumerate(signals):
        out[i, :] = resample_signal_to_length(s, target_len)
    return out


def trim_edges(signal, trim_fraction=0.03, trim_absolute=None):
    L = signal.size
    if trim_absolute is not None:
        t = int(trim_absolute)
    else:
        t = max(1, int(np.round(trim_fraction * L)))
    if t * 2 >= L:
        return signal.copy()
    return signal[t : L - t].copy()


def _plot_segmentation(
    signal,
    segments,
    outpath,
    title=None,
    cls=None,
    reason=None,
    highlight_high_std=None,
):
    x = np.arange(len(signal))
    plt.figure(figsize=(10, 3))
    plt.plot(x, signal, alpha=0.6)
    for s, e, mean, std, l in segments:
        lw = 2
        style = "--"
        if highlight_high_std and std is not None and std > highlight_high_std:
            plt.hlines(mean, s, e - 1, linestyles="-", linewidth=lw + 1)
            plt.fill_betweenx([signal.min(), signal.max()], s, e - 1, alpha=0.08)
        else:
            plt.hlines(mean, s, e - 1, linestyles=style, linewidth=lw)
        plt.axvline(s, alpha=0.18)
    if segments:
        plt.axvline(segments[-1][1], alpha=0.18)
    ttl = title or ""
    if cls is not None or reason is not None:
        ttl = (ttl + " — ") if ttl else ""
        ttl += f'class:{cls or "?"} reason:{reason or "ok"}'
    if ttl:
        plt.title(ttl)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def apply_step_detection(
    signals,
    classes=None,
    run_ids=None,
    outdir=".",
    threshold=18,
    min_len=10,
    step=1,
    smoothing_before=False,
    sg_win=11,
    sg_poly=2,
    max_examples=5,
    noise_std_threshold=None,
):

    od = os.path.join(outdir, "step_detection")
    os.makedirs(od, exist_ok=True)

    kept_idx = []
    excluded = []
    reasons = {}  
    seg_cache = {} 
    kept_count = 0
    rej_count = 0

    for idx, sig in enumerate(signals):
        s = np.asarray(sig).flatten()
        # smoothing before detection optional
        if smoothing_before and len(s) >= sg_win:
            s_seg = savgol_filter(s, sg_win, sg_poly)
        else:
            s_seg = s

        segs = segment_signal(s_seg, threshold=threshold, min_len=min_len, step=step)
        seg_cache[idx] = segs

        if any(l < min_len for (_, _, _, _, l) in segs):
            excluded.append(idx)
            reasons[idx] = "short_segment"
            continue

        if noise_std_threshold is not None:
            high_std_steps = [
                (i, seg[3])
                for i, seg in enumerate(segs)
                if seg[3] > noise_std_threshold
            ]
            if high_std_steps:
                excluded.append(idx)
                reasons[idx] = (
                    f"high_std_in_segment: step_idxs={ [i for i,_ in high_std_steps] }"
                )
                continue

        ok = is_monotonic_down_up(segs)
        if ok:
            kept_idx.append(idx)
        else:
            excluded.append(idx)
            reasons[idx] = "non_monotonic"

        if ok and kept_count < max_examples:
            cls = classes[idx] if classes is not None else None
            _plot_segmentation(
                s,
                segs,
                os.path.join(od, f"KEPT_example_{kept_count}_idx{idx}_cls{cls}.png"),
                title=f"KEPT idx {idx} segs:{len(segs)} (Reason: monotonic)",
            )
            kept_count += 1

        if (not ok) and rej_count < max_examples:
            reason = reasons[idx]
            cls = classes[idx] if classes is not None else None
            _plot_segmentation(
                s,
                segs,
                os.path.join(od, f"REJ_example_{rej_count}_idx{idx}_cls{cls}.png"),
                title=f"REJ idx {idx} segs:{len(segs)} (Reason: {reason})",
            )
            rej_count += 1

    per_class = {}
    if classes is not None:
        classes = np.asarray(classes)
        unique = np.unique(classes)
        for u in unique:
            idxs = np.where(classes == u)[0]
            tot = len(idxs)
            kept_c = sum(1 for i in idxs if i in kept_idx)
            excl_reasons = {}
            for i in idxs:
                if i in reasons:
                    excl_reasons[reasons[i]] = excl_reasons.get(reasons[i], 0) + 1
            per_class[u] = {
                "total": tot,
                "kept": kept_c,
                "kept_pct": 100.0 * kept_c / tot if tot else 0.0,
                "excluded_reasons": {
                    k: (v, 100.0 * v / tot if tot else 0.0)
                    for k, v in excl_reasons.items()
                },
            }

    info = {
        "total": len(signals),
        "kept_count": len(kept_idx),
        "kept_indices": kept_idx,
        "excluded_indices": excluded,
        "reasons": reasons,
        "per_class": per_class,
        "params": {
            "threshold": threshold,
            "min_len": min_len,
            "step": step,
            "smoothing_before": smoothing_before,
            "sg_win": sg_win,
            "sg_poly": sg_poly,
            "noise_std_threshold": noise_std_threshold,
        },
    }
    kept_signals = [signals[i] for i in kept_idx]

    summary_lines = []
    total = info["total"]
    kept_c = info["kept_count"]
    kept_pct_overall = 100.0 * kept_c / total if total else 0.0
    overall_line = f"ALL: kept {kept_c}/{total} ({kept_pct_overall:.1f}%)"
    summary_lines.append(overall_line)
    logger.info(f"[STEP DET] {overall_line}")

    for cls, stats in per_class.items():
        kept = stats["kept"]
        tot = stats["total"]
        kept_pct = stats["kept_pct"]
        excl = stats["excluded_reasons"]
        excl_str = (
            "; ".join([f"{r}: {cnt} ({pct:.1f}%)" for r, (cnt, pct) in excl.items()])
            if excl
            else "none"
        )
        line = f"{cls}: kept {kept}/{tot} ({kept_pct:.1f}%), excluded {excl_str}"
        summary_lines.append(line)
        logger.info(f"[STEP DET] {line}")

    info["summary_lines"] = summary_lines

    return kept_signals, np.array(kept_idx, dtype=int), info


def prepare_dataset_from_df(
    df,
    *,
    train_stats=None,
    smoothing=None,
    norm_method="MAD-local",
    padding_method="percentile",
    percentile=90,
    fixed_length=None,
    padding_mode="truncate_or_pad",
    outdir=".",
    cutoff=0.05,
    order=10,
    trimming=True,
    trim_fraction=0.05,
    step_detection=False,
    step_threshold=18.0,
    step_min_len=5,
    step_step=1,
    step_smoothing_before=False,
    step_sg_win=11,
    step_sg_poly=2,
    step_max_examples=40,
    step_noise_std_threshold=None,
):
    # Ensure columns exist
    if "Signal" not in df.columns:
        raise ValueError("DataFrame must contain signal_raw column")

    signals_trimmed = [(np.asarray(s)) for s in df["Signal"].values]
    classes = df["Class"].values
    run_ids = df["Run_ID"].values

    # Optional trimming
    if trimming:
        signals_trimmed = [
            trim_edges(s, trim_fraction=trim_fraction, trim_absolute=None)
            for s in signals_trimmed
        ]

    step_info = None
    if step_detection and signals_trimmed:
        kept_signals, kept_idx, step_info = apply_step_detection(
            signals_trimmed,
            classes=classes,
            run_ids=run_ids,
            outdir=outdir,
            threshold=step_threshold,
            min_len=step_min_len,
            step=step_step,
            smoothing_before=step_smoothing_before,
            sg_win=step_sg_win,
            sg_poly=step_sg_poly,
            max_examples=step_max_examples,
            noise_std_threshold=step_noise_std_threshold,
        )
        if kept_idx.size == 0:
            return (
                np.empty((0,)),
                np.array([]),
                [],
                np.array([]),
                {
                    "target_length": 0,
                    "lengths": [],
                    "norm_info": {},
                    "mode": padding_mode,
                    "step_detection": step_info,
                },
            )
        signals_trimmed = [signals_trimmed[i] for i in kept_idx]
        classes = classes[kept_idx]
        run_ids = run_ids[kept_idx]

    if smoothing and smoothing != "None":
        signals_trimmed = apply_smoothing(
            signals_trimmed,
            outdir=outdir,
            smoothing_method=smoothing,
            cutoff=cutoff,
            order=order,
        )

    signals_norm, norm_info = normalize_signals(
        signals_trimmed, method=norm_method, train_stats=train_stats
    )

    lengths = [len(s) for s in signals_norm]
    if padding_method == "percentile":
        target = (
            int(np.percentile(lengths, percentile))
            if len(lengths) > 0
            else (fixed_length or 0)
        )
    elif padding_method == "median":
        target = int(np.median(lengths)) if len(lengths) > 0 else (fixed_length or 0)
    elif padding_method == "mean":
        target = int(np.mean(lengths)) if len(lengths) > 0 else (fixed_length or 0)
    elif padding_method == "fixed":
        if fixed_length is None:
            raise ValueError("fixed_length must be provided for fixed padding")
        target = int(fixed_length)
    else:
        raise ValueError("Unknown padding_method")

    if target <= 0:
        raise ValueError(
            "Computed target length <= 0; check your signals and parameters."
        )

    pad_values = []
    df_present = df is not None
    for i, s in enumerate(signals_norm):
        if df_present and "median_before_plr" in df.columns:
            v = 0.0  # df.iloc[i].get('median_before_plr', None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0.0 if s.size > 0 else 0.0
        else:
            v = 0.0 if s.size > 0 else 0.0
        pad_values.append(v)

    X, kept, _ = pad_trim_or_resample(
        signals_norm,
        target_length=target,
        mode=padding_mode,
        pad_values=pad_values,
        return_length_feature=True,
    )

    if kept:
        classes = classes[kept]
        run_ids = run_ids[kept]
        signals_norm = [signals_norm[i] for i in kept]

    info = {
        "target_length": target,
        "lengths": lengths,
        "norm_info": norm_info,
        "mode": padding_mode,
    }
    return X, np.array(classes), signals_norm, np.array(run_ids), info
