#!/usr/bin/env python

import math
import os
import re
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket
from joblib import Parallel, delayed
from pycatch22 import catch22_all
from scipy import stats
from scipy.signal import decimate, find_peaks
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA

from utils_dtw import dtw_features_to_templates

COMPACT_NON_C22_FEATURES_35 = [
    "idl_60_40",
    "trimmed_MAD",
    "MAD_over_IQR",
    "q90",
    "kurtosis",
    "cross_rate_q25",
    "cross_rate_q50",
    "hjorth_complexity",
    "slope_q10",
    "slope_q40",
    "slope_q75",
    "median_slope",
    "total_variation",
    "median_rolling_slope",
    "rolling_slope_q25",
    "drop_post_min",
    "drop_duration",
    "drop_auc_norm",
    "mean_peak_dist",
    "step_mean_mad",
    "step_mean_iqr",
    "max_step_height",
    "min_step_height",
    "median_abs_step_height",
    "ms32_tv_mean",
    "max",
    "slope_q25",
    "slope_q60",
    "slope_q90",
    "cross_rate_q10",
    "rolling_slope_q60",
    "mean_step_len",
    "median_step_len",
    "q25_step_std",
    "ms16_tv_mean",
]

COMPACT_NON_C22_FEATURES_25 = COMPACT_NON_C22_FEATURES_35

TOP50_FEATURES = [
    "min_step_height",
    "rolling_slope_q25",
    "min_slope",
    "skew",
    "c22_DN_HistogramMode_10",
    "c22_DN_OutlierInclude_p_001_mdrmd",
    "cross_rate_q25",
    "c22_DN_HistogramMode_5",
    "drop_depth",
    "drop_auc_norm",
    "slope_q60",
    "drop_auc",
    "max",
    "slope_q75",
    "mean_peak_prom",
    "drop_idx",
    "c22_CO_trev_1_num",
    "drop_post_min",
    "drop_duration",
    "median_rolling_slope",
    "kurtosis",
    "c22_SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "total_variation",
    "drop_pre_mean",
    "c22_DN_OutlierInclude_n_001_mdrmd",
    "slope_q90",
    "mean_step_height",
    "MAD_over_IQR",
    "idl_60_40",
    "c22_CO_HistogramAMI_even_2_5",
    "c22_SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "max_step_height",
    "c22_MD_hrv_classic_pnn40",
    "c22_SB_MotifThree_quantile_hh",
    "min_rolling_slope",
    "q60",
    "trimmed_MAD",
    "median",
    "cross_rate_q10",
    "cross_rate_q50",
    "median_slope",
    "max_rolling_slope",
    "rolling_slope_q60",
    "slope_q10",
    "range",
    "mean_slope",
    "q90",
    "q75",
    "c22_FC_LocalSimple_mean1_tauresrat",
    "mean_peak_dist",
]

FEATURE_SET_PRESETS = {
    "all": None,
    "top50": TOP50_FEATURES,
}

_BASIC_5_FEATURE_NAMES = {
    "std",
    "median",
    "skew",
    "kurtosis",
    "range",
    "length_th",
}

_BASIC_PARAM_FEATURE_NAMES = {
    "length",
    "min",
    "max",
    "mean",
    "median",
    "q10",
    "q25",
    "q40",
    "q60",
    "q75",
    "q90",
    "iqr",
    "range",
    "idl",
    "idl_60_40",
    "midhinge",
    "trimean",
    "interquartile_mean",
    "MAD",
    "coef_var",
    "normalized_MAD",
    "MAD_over_IQR",
    "trimmed_MAD",
}

_SLOPE_FEATURE_NAMES = {
    "mean_slope",
    "median_slope",
    "std_slope",
    "max_slope",
    "min_slope",
    "avg_abs_slope",
    "slope_q10",
    "slope_q25",
    "slope_q40",
    "slope_q60",
    "slope_q75",
    "slope_q90",
    "total_variation",
}

_SECOND_DERIVATIVE_FEATURE_NAMES = {
    "median_second_slope",
    "second_slope_q60",
    "std_second_slope",
    "mean_second_slope",
    "max_second_slope",
    "min_second_slope",
    "avg_abs_second_slope",
    "second_slope_q10",
    "second_slope_q25",
    "second_slope_q75",
    "second_slope_q90",
    "second_total_variation",
}

_PEAK_FEATURE_NAMES = {
    "n_peaks",
    "n_troughs",
    "mean_peak_val",
    "mean_trough_val",
    "mean_peak_prom",
    "mean_trough_prom",
    "mean_peak_dist",
    "median_peak_dist",
    "peak_dist_q10",
    "peak_dist_q90",
}

_TREND_FEATURE_NAMES = {
    "energy",
    "rms",
    "auc",
    "abs_auc",
    "trend_slope",
    "trend_intercept",
    "detrended_std",
}

_SPECTRAL_FEATURE_NAMES = {
    "total_power",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "dominant_freq",
    "zcr",
    "ac_lag1",
    "ac_zero_cross_lag",
}

_STEP_EXACT_FEATURE_NAMES = {
    "n_steps",
    "n_high_std_steps",
    "frac_high_std_steps",
    "mono_full",
    "mono_excl_last",
}


def _catch22_worker(signal_values, result_queue):
    try:
        result_queue.put(catch22_all(signal_values))
    except Exception as exc:
        result_queue.put({"error": str(exc)})


def catch22_safe(s, min_len=10, min_std=1e-8, use_subprocess=False, timeout=5.0):
    arr = np.asarray(s, dtype=float)

    if arr.size < min_len:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    std = float(np.nanstd(arr))
    if std < min_std:
        return None
    if np.nanmax(np.abs(arr)) > 1e12:
        return None

    if np.any(np.isnan(arr)):
        arr = np.where(np.isnan(arr), float(np.nanmedian(arr)), arr)

    if not use_subprocess:
        try:
            return catch22_all(arr.tolist())
        except Exception:
            return None

    q = Queue()
    p = Process(target=_catch22_worker, args=(arr.tolist(), q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    # retrieve result if present
    if not q.empty():
        res = q.get()
        if isinstance(res, dict) and res.get("error") is not None:
            return None
        return res
    return None


def spectral_on_decimated(s, decim_factor=2):
    if decim_factor <= 1 or s.size < 32:
        s2 = s
    else:
        # use zero-phase decimate for speed/quality
        s2 = decimate(s, decim_factor, zero_phase=True)
    s_centered = s2 - np.nanmean(s2)
    fft = np.fft.rfft(s_centered)
    mags = np.abs(fft)
    freqs = np.fft.rfftfreq(s2.size, d=1.0)
    return mags, freqs


def _multiscale_profile_features(s, scales=(4, 8, 16, 32)):
    out = {}
    arr = np.asarray(s, dtype=float)
    n = arr.size
    if n == 0:
        for sc in scales:
            out[f"ms{sc}_std"] = 0.0
            out[f"ms{sc}_iqr"] = 0.0
            out[f"ms{sc}_tv_mean"] = 0.0
            out[f"ms{sc}_ac1"] = 0.0
            out[f"ms{sc}_zcr"] = 0.0
        return out

    for sc in scales:
        if sc <= 1:
            pooled = arr
        else:
            m = n // sc
            pooled = arr if m <= 0 else arr[: m * sc].reshape(m, sc).mean(axis=1)

        if pooled.size == 0:
            out[f"ms{sc}_std"] = 0.0
            out[f"ms{sc}_iqr"] = 0.0
            out[f"ms{sc}_tv_mean"] = 0.0
            out[f"ms{sc}_ac1"] = 0.0
            out[f"ms{sc}_zcr"] = 0.0
            continue

        dif = np.diff(pooled)
        q25 = float(np.nanpercentile(pooled, 25))
        q75 = float(np.nanpercentile(pooled, 75))
        zc = int(np.sum((pooled[:-1] * pooled[1:]) < 0)) if pooled.size > 1 else 0

        out[f"ms{sc}_std"] = float(np.nanstd(pooled))
        out[f"ms{sc}_iqr"] = float(q75 - q25)
        out[f"ms{sc}_tv_mean"] = float(np.nanmean(np.abs(dif))) if dif.size else 0.0
        out[f"ms{sc}_ac1"] = float(ac_lag1_fast(pooled)) if pooled.size > 1 else 0.0
        out[f"ms{sc}_zcr"] = float(zc / max(1, pooled.size - 1))

    return out


def _extract_one_feature(
    sig,
    thresholds,
    use_catch22=True,
    basic_params=True,
    drop_window=15,
    basic_5=True,
    slope_features=True,
    drop_features=True,
    peak_features=True,
    trend_features=True,
    spectral_features=False,
    second_derivative_features=True,
    rolling_window_slope_features=True,
    step_detection_features=True,
    multiscale_features=True,
    requested_features_set=None,
):
    s = np.asarray(sig, dtype=float)
    if s.size == 0:
        return {"length": 0}

    feat = {}
    n = s.size

    # Basic stats (compute once)
    non_nan_mask = np.isfinite(s)
    if not np.any(non_nan_mask):
        # all NaN: return mostly zeros
        return {"length": int(n)}

    s_clean = (
        s 
    )
    requested = (
        set(requested_features_set) if requested_features_set is not None else None
    )

    def _wants(name):
        return (requested is None) or (name in requested)

    if basic_5:
        s_low = s_clean[s_clean <= 0]

        kurt_v = stats.kurtosis(s_low)
        skew_v = stats.skew(s_low)

        min_v = float(np.nanmin(s_clean))
        max_v = float(np.nanmax(s_clean))
        rng_v = max_v - min_v

        std_v = float(np.nanstd(s_low))

        t_below = len(s_low)

        median_v = float(np.nanmedian(s_low))

        feat.update(
            {
                "std": std_v,
                "median": median_v,
                "skew": skew_v,
                "kurtosis": kurt_v,
                "range": rng_v,
                "length_th": t_below,
            }
        )

    # Basic descriptive stats
    if basic_params:
        min_v = float(np.nanmin(s_clean))
        max_v = float(np.nanmax(s_clean))
        mean_v = float(np.nanmean(s_clean))
        median_v = float(np.nanmedian(s_clean))
        q25 = float(percentile_k(s_clean, 25))
        q75 = float(percentile_k(s_clean, 75))
        q10 = float(percentile_k(s_clean, 10))
        q90 = float(percentile_k(s_clean, 90))
        q40 = float(percentile_k(s_clean, 40))
        q60 = float(percentile_k(s_clean, 60))
        iqr = q75 - q25
        idl = q90 - q10
        idl_60_40 = q60 - q40
        midhinge = (q25 + q75) / 2
        trimean = (q25 + 2 * median_v + q75) / 4
        rng_v = max_v - min_v
        mean_abs_dev = (
            float(np.nanmean(np.abs(s_clean - np.nanmean(s_clean))))
            if n > 0
            else np.nan
        )
        mad = mean_abs_dev
        eps = 1e-12
        normalized_MAD = float(mad / (abs(median_v) + eps))  # MAD relative to median
        MAD_over_IQR = float(mad / (iqr + eps))  # MAD relative to IQR

        trim_pct = 0.10  # 10% trim each tail
        if n > 10:
            lo_p = int(np.floor(n * trim_pct))
            hi_p = int(np.ceil(n * (1 - trim_pct)))
            s_sorted = np.sort(s_clean)
            s_trim = s_sorted[lo_p:hi_p] if hi_p > lo_p else s_sorted
            trimmed_mad = (
                float(np.nanmedian(np.abs(s_trim - np.nanmedian(s_trim))))
                if s_trim.size
                else float(mad)
            )
        else:
            trimmed_mad = float(mad)

        coef_var = float(np.nanstd(s_clean) / (abs(mean_v) + 1e-12))

        iqm_vals = s_clean[(s_clean >= q25) & (s_clean <= q75)]
        interquartile_mean = (
            float(np.nanmean(iqm_vals)) if iqm_vals.size > 0 else np.nan
        )

        feat.update(
            {
                "length": int(n),
                "min": min_v,
                "max": max_v,
                "mean": mean_v,
                "median": median_v,
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
                "range": rng_v,
                "q10": q10,
                "q90": q90,
                "q40": q40,
                "q60": q60,
                "idl": idl,
                "idl_60_40": idl_60_40,
                "midhinge": midhinge,
                "trimean": trimean,
                "interquartile_mean": interquartile_mean,
                "MAD": mad,
                "coef_var": coef_var,
                "normalized_MAD": normalized_MAD,
                "MAD_over_IQR": MAD_over_IQR,
                "trimmed_MAD": trimmed_mad,
            }
        )
    else:
        mean_v = float(np.nanmean(s_clean))
        median_v = float(np.nanmedian(s_clean))

    diffs = np.diff(s_clean)

    if slope_features:
        if diffs.size > 0:
            mean_slope = float(np.nanmean(diffs))
            median_slope = float(np.nanmedian(diffs))
            std_slope = float(np.nanstd(diffs))
            max_slope = float(np.nanmax(diffs))
            min_slope = float(np.nanmin(diffs))
            avg_abs_slope = float(np.nanmean(np.abs(diffs)))
            slope_q10 = float(percentile_k(diffs, 10))
            slope_q90 = float(percentile_k(diffs, 90))
            slope_q25 = float(percentile_k(diffs, 25))
            slope_q75 = float(percentile_k(diffs, 75))
            slope_q40 = float(percentile_k(diffs, 40))
            slope_q60 = float(percentile_k(diffs, 60))
            total_variation = float(np.nansum(np.abs(diffs)))
        else:
            mean_slope = median_slope = std_slope = max_slope = min_slope = (
                avg_abs_slope
            ) = prop_large_slope = 0.0
            slope_q10 = slope_q90 = 0.0

        feat.update(
            {
                "mean_slope": mean_slope,
                "median_slope": median_slope,
                "std_slope": std_slope,
                "max_slope": max_slope,
                "min_slope": min_slope,
                "avg_abs_slope": avg_abs_slope,
                "slope_q10": slope_q10,
                "slope_q90": slope_q90,
                "slope_q25": slope_q25,
                "slope_q75": slope_q75,
                "slope_q40": slope_q40,
                "slope_q60": slope_q60,
                "total_variation": total_variation,
            }
        )

    if second_derivative_features:
        second_diffs = np.diff(np.diff(s_clean))
        if second_diffs.size > 0:
            median_second_slope = float(np.nanmedian(second_diffs))
            second_slope_q60 = float(percentile_k(second_diffs, 60))
            std_second_slope = float(np.nanstd(second_diffs))
            mean_second_slope = float(np.nanmean(second_diffs))
            max_second_slope = float(np.nanmax(second_diffs))
            min_second_slope = float(np.nanmin(second_diffs))
            avg_abs_second_slope = float(np.nanmean(np.abs(second_diffs)))
            second_slope_q25 = float(percentile_k(second_diffs, 25))
            second_slope_q75 = float(percentile_k(second_diffs, 75))
            second_slope_q90 = float(percentile_k(second_diffs, 90))
            second_slope_q10 = float(percentile_k(second_diffs, 10))
            second_total_variation = float(np.nansum(np.abs(second_diffs)))
        else:
            median_second_slope = second_slope_q60 = std_second_slope = (
                mean_second_slope
            ) = 0.0
            max_second_slope = min_second_slope = avg_abs_second_slope = 0.0
            second_slope_q25 = second_slope_q75 = 0.0
            second_total_variation = second_tv_per_len = 0.0
        feat.update(
            {
                "median_second_slope": median_second_slope,
                "second_slope_q60": second_slope_q60,
                "std_second_slope": std_second_slope,
                "mean_second_slope": mean_second_slope,
                "max_second_slope": max_second_slope,
                "min_second_slope": min_second_slope,
                "avg_abs_second_slope": avg_abs_second_slope,
                "second_slope_q25": second_slope_q25,
                "second_slope_q75": second_slope_q75,
                "second_slope_q90": second_slope_q90,
                "second_slope_q10": second_slope_q10,
                "second_total_variation": second_total_variation,
            }
        )

    if rolling_window_slope_features:
        rolling_window_size = 100
        w = rolling_window_size
        if n < w:
            zeros = {
                k: 0.0
                for k in (
                    "median_rolling_slope",
                    "rolling_slope_q60",
                    "std_rolling_slope",
                    "mean_rolling_slope",
                    "max_rolling_slope",
                    "min_rolling_slope",
                    "avg_abs_rolling_slope",
                    "rolling_slope_q25",
                    "rolling_slope_q75",
                )
            }
            feat.update(zeros)
        else:
            x = np.arange(w, dtype=float)
            sum_x = x.sum()
            sum_x2 = (x * x).sum()
            denom = w * sum_x2 - sum_x * sum_x  # > 0 for w>1

            y = s_clean.astype(float)

            finite_mask = np.isfinite(y).astype(float)
            y_zeroed = np.where(np.isfinite(y), y, 0.0)

            ones = np.ones(w, dtype=float)
            sum_y = np.convolve(y_zeroed, ones, mode="valid")
            sum_y2 = np.convolve(y_zeroed * y_zeroed, ones, mode="valid")
            sum_xy = np.convolve(y_zeroed, x[::-1], mode="valid")

            counts = np.convolve(finite_mask, ones, mode="valid")

            slopes = (w * sum_xy - sum_x * sum_y) / denom

            with np.errstate(invalid="ignore", divide="ignore"):
                mean_y = sum_y / counts
                var_y = (sum_y2 - (sum_y * sum_y) / counts) / counts

            invalid_mask = (
                (counts < w) | (counts == 0) | (var_y <= 1e-12) | ~np.isfinite(slopes)
            )
            slopes_masked = np.where(invalid_mask, np.nan, slopes)

            if np.all(np.isnan(slopes_masked)):
                zeros = {
                    k: 0.0
                    for k in (
                        "median_rolling_slope",
                        "rolling_slope_q60",
                        "std_rolling_slope",
                        "mean_rolling_slope",
                        "max_rolling_slope",
                        "min_rolling_slope",
                        "avg_abs_rolling_slope",
                        "rolling_slope_q25",
                        "rolling_slope_q75",
                    )
                }
                feat.update(zeros)
            else:
                rolling_slopes = slopes_masked  # length n-w+1 with NaNs

                median_rolling_slope = float(np.nanmedian(rolling_slopes))
                rolling_slope_q60 = float(np.nanpercentile(rolling_slopes, 60))
                std_rolling_slope = float(np.nanstd(rolling_slopes))
                mean_rolling_slope = float(np.nanmean(rolling_slopes))
                max_rolling_slope = float(np.nanmax(rolling_slopes))
                min_rolling_slope = float(np.nanmin(rolling_slopes))
                avg_abs_rolling_slope = float(np.nanmean(np.abs(rolling_slopes)))
                rolling_slope_q25 = float(np.nanpercentile(rolling_slopes, 25))
                rolling_slope_q75 = float(np.nanpercentile(rolling_slopes, 75))

                feat.update(
                    {
                        "median_rolling_slope": median_rolling_slope,
                        "rolling_slope_q60": rolling_slope_q60,
                        "std_rolling_slope": std_rolling_slope,
                        "mean_rolling_slope": mean_rolling_slope,
                        "max_rolling_slope": max_rolling_slope,
                        "min_rolling_slope": min_rolling_slope,
                        "avg_abs_rolling_slope": avg_abs_rolling_slope,
                        "rolling_slope_q25": rolling_slope_q25,
                        "rolling_slope_q75": rolling_slope_q75,
                    }
                )

    if peak_features:
        peaks, _ = find_peaks(s_clean) if n > 0 else (np.array([], dtype=int), {})
        peak_updates = {}

        if _wants("n_peaks"):
            peak_updates["n_peaks"] = int(peaks.size)
        if _wants("mean_peak_val"):
            peak_updates["mean_peak_val"] = (
                float(np.nanmean(s_clean[peaks])) if peaks.size else 0.0
            )
        if _wants("mean_peak_prom"):
            peak_updates["mean_peak_prom"] = (
                float(np.nanmean(s_clean[peaks] - median_v)) if peaks.size else 0.0
            )

        if (
            _wants("mean_peak_dist")
            or _wants("median_peak_dist")
            or _wants("peak_dist_q10")
            or _wants("peak_dist_q90")
        ):
            if peaks.size >= 2:
                peak_dists = np.diff(peaks)
                if _wants("mean_peak_dist"):
                    peak_updates["mean_peak_dist"] = float(np.nanmean(peak_dists))
                if _wants("median_peak_dist"):
                    peak_updates["median_peak_dist"] = float(np.nanmedian(peak_dists))
                if _wants("peak_dist_q10"):
                    peak_updates["peak_dist_q10"] = float(percentile_k(peak_dists, 10))
                if _wants("peak_dist_q90"):
                    peak_updates["peak_dist_q90"] = float(percentile_k(peak_dists, 90))
            else:
                if _wants("mean_peak_dist"):
                    peak_updates["mean_peak_dist"] = float("nan")
                if _wants("median_peak_dist"):
                    peak_updates["median_peak_dist"] = float("nan")
                if _wants("peak_dist_q10"):
                    peak_updates["peak_dist_q10"] = float("nan")
                if _wants("peak_dist_q90"):
                    peak_updates["peak_dist_q90"] = float("nan")

        if (
            _wants("n_troughs")
            or _wants("mean_trough_val")
            or _wants("mean_trough_prom")
        ):
            troughs, _ = (
                find_peaks(-s_clean) if n > 0 else (np.array([], dtype=int), {})
            )
            if _wants("n_troughs"):
                peak_updates["n_troughs"] = int(troughs.size)
            if _wants("mean_trough_val"):
                peak_updates["mean_trough_val"] = (
                    float(np.nanmean(s_clean[troughs])) if troughs.size else 0.0
                )
            if _wants("mean_trough_prom"):
                peak_updates["mean_trough_prom"] = (
                    float(np.nanmean(median_v - s_clean[troughs]))
                    if troughs.size
                    else 0.0
                )

        feat.update(peak_updates)

    if trend_features:
        energy = float(np.nansum(s_clean**2))
        rms = float(np.sqrt(np.nanmean(s_clean**2))) if n > 0 else 0.0
        auc = float(np.nansum(s_clean))
        abs_auc = float(np.nansum(np.abs(s_clean)))
        feat.update({"energy": energy, "rms": rms, "auc": auc, "abs_auc": abs_auc})

        try:
            x = np.arange(n)
            trend_slope, trend_intercept = fast_slope(s)
            detrended = s_clean - (trend_slope * x + trend_intercept)
            detrended_std = float(np.nanstd(detrended))
        except Exception:
            trend_slope = 0.0
            trend_intercept = 0.0
            detrended_std = float(np.nanstd(s_clean) if n > 0 else 0.0)
        feat.update(
            {
                "trend_slope": trend_slope,
                "trend_intercept": trend_intercept,
                "detrended_std": detrended_std,
            }
        )

    if spectral_features:
        try:
            s_centered = s_clean - mean_v
            mags, freqs = spectral_on_decimated(s_centered, decim_factor=2)
            total_power = float(np.sum(mags**2))
            if np.sum(mags) > 0:
                spectral_centroid = float(np.sum(freqs * mags) / (np.sum(mags) + 1e-12))
                spectral_bandwidth = float(
                    np.sqrt(
                        np.sum(((freqs - spectral_centroid) ** 2) * mags)
                        / (np.sum(mags) + 1e-12)
                    )
                )
                ps = mags / (np.sum(mags) + 1e-12)
                spectral_entropy = float(stats.entropy(ps + 1e-12))
                dominant_freq = float(freqs[int(np.nanargmax(mags))])
            else:
                spectral_centroid = spectral_bandwidth = spectral_entropy = (
                    dominant_freq
                ) = 0.0
        except Exception:
            spectral_centroid = spectral_bandwidth = spectral_entropy = (
                dominant_freq
            ) = 0.0
            total_power = 0.0

        feat.update(
            {
                "total_power": total_power,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "spectral_entropy": spectral_entropy,
                "dominant_freq": dominant_freq,
            }
        )

        try:
            zc = int(np.sum((s_clean[:-1] * s_clean[1:]) < 0))
            zcr = float(zc / max(1, n - 1))
        except Exception:
            zcr = 0.0

        try:
            s_mean = mean_v
            centered = s_clean - s_mean
            ac_lag1 = float(ac_lag1_fast(centered))
            if n > 1:
                ac_full = np.correlate(centered, centered, mode="full")
                ac_pos = ac_full[ac_full.size // 2 :]
                if ac_pos.size > 0 and ac_pos[0] != 0:
                    acf = ac_pos / ac_pos[0]
                    zero_cross_lag = (
                        int(np.where(acf <= 0)[0][0])
                        if np.any(acf <= 0)
                        else int(len(acf) - 1)
                    )
                else:
                    zero_cross_lag = 0
            else:
                zero_cross_lag = 0
        except Exception:
            ac_lag1 = 0.0
            zero_cross_lag = 0

        feat.update(
            {"zcr": zcr, "ac_lag1": ac_lag1, "ac_zero_cross_lag": zero_cross_lag}
        )

    if multiscale_features:
        ms_updates = {}

        if (
            _wants("hjorth_activity")
            or _wants("hjorth_mobility")
            or _wants("hjorth_complexity")
        ):
            var0 = float(np.nanvar(s_clean)) if n > 0 else 0.0
            d1 = diffs
            d2 = np.diff(d1) if d1.size > 1 else np.array([], dtype=float)
            var1 = float(np.nanvar(d1)) if d1.size > 0 else 0.0
            var2 = float(np.nanvar(d2)) if d2.size > 0 else 0.0
            mobility = float(np.sqrt(var1 / (var0 + 1e-12)))
            complexity = float(np.sqrt(var2 / (var1 + 1e-12)) / (mobility + 1e-12))
            if _wants("hjorth_activity"):
                ms_updates["hjorth_activity"] = var0
            if _wants("hjorth_mobility"):
                ms_updates["hjorth_mobility"] = mobility
            if _wants("hjorth_complexity"):
                ms_updates["hjorth_complexity"] = complexity

        cross_map = {
            "cross_rate_q10": 10,
            "cross_rate_q25": 25,
            "cross_rate_q50": 50,
            "cross_rate_q75": 75,
            "cross_rate_q90": 90,
        }
        needed_cross = [k for k in cross_map if _wants(k)]
        if needed_cross:

            def _cross_rate(x, thr):
                if x.size < 2:
                    return 0.0
                b = x > thr
                return float(np.sum(b[:-1] != b[1:]) / max(1, x.size - 1))

            q_cache = {}
            for k in needed_cross:
                q = cross_map[k]
                if q not in q_cache:
                    q_cache[q] = float(np.nanpercentile(s_clean, q))
                ms_updates[k] = _cross_rate(s_clean, q_cache[q])

        need_ms_profiles = (requested is None) or any(
            re.match(r"^ms\d+_", f) for f in requested
        )
        if need_ms_profiles:
            ms_all = _multiscale_profile_features(s_clean, scales=(4, 8, 16, 32))
            if requested is None:
                ms_updates.update(ms_all)
            else:
                for k, v in ms_all.items():
                    if k in requested:
                        ms_updates[k] = v

        feat.update(ms_updates)

    if drop_features:
        if diffs.size > 0:
            idx_drop = int(np.argmin(diffs))
            w = int(min(drop_window, idx_drop + 1))
            pre_start = max(0, idx_drop - w + 1)
            pre_mean = (
                float(np.nanmedian(s_clean[pre_start : idx_drop + 1]))
                if (idx_drop + 1 - pre_start) > 0
                else float(median_v)
            )
            post_end = min(n, idx_drop + 1 + drop_window)
            post_min = (
                float(np.nanmin(s_clean[idx_drop + 1 : post_end]))
                if (post_end - (idx_drop + 1)) > 0
                else float(min_v)
            )
            drop_depth = float(pre_mean - post_min)
            threshold_return = pre_mean - 0.25 * drop_depth
            ret_idx = idx_drop + 1
            while ret_idx < n and s_clean[ret_idx] <= threshold_return:
                ret_idx += 1
            drop_duration = int(ret_idx - (idx_drop + 1))
            drop_segment = (
                s_clean[idx_drop + 1 : ret_idx]
                if ret_idx > idx_drop + 1
                else np.array([])
            )
            drop_auc = (
                float(np.sum(pre_mean - drop_segment)) if drop_segment.size else 0.0
            )
            drop_auc_norm = drop_auc / (np.sum(np.abs(s_clean)) + 1e-12)
        else:
            idx_drop = -1
            pre_mean = float(median_v)
            post_min = float(min_v)
            drop_depth = 0.0
            drop_duration = 0
            drop_auc = 0.0
            drop_auc_norm = 0.0

        feat.update(
            {
                "drop_idx": int(idx_drop),
                "drop_pre_mean": pre_mean,
                "drop_post_min": post_min,
                "drop_depth": float(drop_depth),
                "drop_duration": int(drop_duration),
                "drop_auc": float(drop_auc),
                "drop_auc_norm": float(drop_auc_norm),
            }
        )

    if use_catch22:
        try:
            c22_res = catch22_safe(
                s_clean, min_len=10, min_std=1e-8, use_subprocess=False, timeout=5.0
            )
            if c22_res:
                for nme, val in zip(c22_res["names"], c22_res["values"]):
                    feat[f"c22_{nme}"] = float(val)
        except Exception:
            pass

    if step_detection_features:
        segs = segment_signal(s_clean)
        n_steps = len(segs)
        step_updates = {}

        if _wants("n_steps"):
            step_updates["n_steps"] = int(n_steps)

        need_lens = any(
            _wants(k)
            for k in (
                "min_step_len",
                "max_step_len",
                "mean_step_len",
                "median_step_len",
                "std_step_len",
                "prop_short_steps",
            )
        )
        step_lens = (
            np.array([l for *_, l in segs], dtype=int)
            if (n_steps and need_lens)
            else np.array([], dtype=int)
        )
        if need_lens:
            if _wants("min_step_len"):
                step_updates["min_step_len"] = (
                    int(np.min(step_lens)) if step_lens.size else 0
                )
            if _wants("max_step_len"):
                step_updates["max_step_len"] = (
                    int(np.max(step_lens)) if step_lens.size else 0
                )
            if _wants("mean_step_len"):
                step_updates["mean_step_len"] = (
                    float(np.mean(step_lens)) if step_lens.size else 0.0
                )
            if _wants("median_step_len"):
                step_updates["median_step_len"] = (
                    float(np.median(step_lens)) if step_lens.size else 0.0
                )
            if _wants("std_step_len"):
                step_updates["std_step_len"] = (
                    float(np.std(step_lens)) if step_lens.size else 0.0
                )
            if _wants("prop_short_steps"):
                step_updates["prop_short_steps"] = (
                    float(np.sum(step_lens < 10) / step_lens.size)
                    if step_lens.size
                    else 0.0
                )

        need_means = any(
            _wants(k)
            for k in (
                "mean_step_mean",
                "std_step_mean",
                "min_step_mean",
                "max_step_mean",
                "q25_step_mean",
                "q75_step_mean",
                "step_mean_iqr",
                "step_mean_mad",
                "mean_step_height",
                "mean_abs_step_height",
                "max_step_height",
                "min_step_height",
                "median_abs_step_height",
            )
        )
        step_means = (
            np.array([m for _, _, m, _, _ in segs], dtype=float)
            if (n_steps and need_means)
            else np.array([], dtype=float)
        )

        if need_means:
            if _wants("mean_step_mean"):
                step_updates["mean_step_mean"] = (
                    float(np.nanmean(step_means)) if step_means.size else 0.0
                )
            if _wants("std_step_mean"):
                step_updates["std_step_mean"] = (
                    float(np.nanstd(step_means)) if step_means.size else 0.0
                )
            if _wants("min_step_mean"):
                step_updates["min_step_mean"] = (
                    float(np.nanmin(step_means)) if step_means.size else 0.0
                )
            if _wants("max_step_mean"):
                step_updates["max_step_mean"] = (
                    float(np.nanmax(step_means)) if step_means.size else 0.0
                )

            if (
                _wants("q25_step_mean")
                or _wants("q75_step_mean")
                or _wants("step_mean_iqr")
            ):
                q25_mean = (
                    float(np.nanpercentile(step_means, 25)) if step_means.size else 0.0
                )
                q75_mean = (
                    float(np.nanpercentile(step_means, 75)) if step_means.size else 0.0
                )
                if _wants("q25_step_mean"):
                    step_updates["q25_step_mean"] = q25_mean
                if _wants("q75_step_mean"):
                    step_updates["q75_step_mean"] = q75_mean
                if _wants("step_mean_iqr"):
                    step_updates["step_mean_iqr"] = q75_mean - q25_mean

            if _wants("step_mean_mad"):
                step_updates["step_mean_mad"] = (
                    float(np.nanmedian(np.abs(step_means - np.nanmedian(step_means))))
                    if step_means.size
                    else 0.0
                )

            if (
                _wants("mean_step_height")
                or _wants("mean_abs_step_height")
                or _wants("max_step_height")
                or _wants("min_step_height")
                or _wants("median_abs_step_height")
            ):
                if step_means.size >= 2:
                    h = np.diff(step_means)
                    if _wants("mean_step_height"):
                        step_updates["mean_step_height"] = float(np.mean(h))
                    if _wants("mean_abs_step_height"):
                        step_updates["mean_abs_step_height"] = float(np.mean(np.abs(h)))
                    if _wants("max_step_height"):
                        step_updates["max_step_height"] = float(np.max(h))
                    if _wants("min_step_height"):
                        step_updates["min_step_height"] = float(np.min(h))
                    if _wants("median_abs_step_height"):
                        step_updates["median_abs_step_height"] = float(
                            np.median(np.abs(h))
                        )
                else:
                    if _wants("mean_step_height"):
                        step_updates["mean_step_height"] = 0.0
                    if _wants("mean_abs_step_height"):
                        step_updates["mean_abs_step_height"] = 0.0
                    if _wants("max_step_height"):
                        step_updates["max_step_height"] = 0.0
                    if _wants("min_step_height"):
                        step_updates["min_step_height"] = 0.0
                    if _wants("median_abs_step_height"):
                        step_updates["median_abs_step_height"] = 0.0

        need_stds = any(
            _wants(k)
            for k in (
                "mean_step_std",
                "q25_step_std",
                "q75_step_std",
                "n_high_std_steps",
                "frac_high_std_steps",
            )
        )
        step_stds = (
            np.array([sd for _, _, _, sd, _ in segs], dtype=float)
            if (n_steps and need_stds)
            else np.array([], dtype=float)
        )
        if need_stds:
            if _wants("mean_step_std"):
                step_updates["mean_step_std"] = (
                    float(np.nanmean(step_stds)) if step_stds.size else 0.0
                )
            if _wants("q25_step_std"):
                step_updates["q25_step_std"] = (
                    float(np.nanpercentile(step_stds, 25)) if step_stds.size else 0.0
                )
            if _wants("q75_step_std"):
                step_updates["q75_step_std"] = (
                    float(np.nanpercentile(step_stds, 75)) if step_stds.size else 0.0
                )

            if _wants("n_high_std_steps") or _wants("frac_high_std_steps"):
                overall_std = float(np.nanstd(s_clean)) if n > 0 else 0.0
                thresh = max(overall_std * 2.0, 1e-12)
                n_high = int(np.sum(step_stds > thresh)) if step_stds.size else 0
                if _wants("n_high_std_steps"):
                    step_updates["n_high_std_steps"] = n_high
                if _wants("frac_high_std_steps"):
                    step_updates["frac_high_std_steps"] = (
                        float(n_high) / n_steps if n_steps else 0.0
                    )

        if _wants("mean_step_dist_frac") or _wants("median_step_dist_frac"):
            starts = (
                np.array([s0 for s0, _, _, _, _ in segs], dtype=int)
                if n_steps
                else np.array([], dtype=int)
            )
            if starts.size >= 2:
                d = np.diff(starts)
                if _wants("mean_step_dist_frac"):
                    step_updates["mean_step_dist_frac"] = float(np.mean(d) / max(1, n))
                if _wants("median_step_dist_frac"):
                    step_updates["median_step_dist_frac"] = float(
                        np.median(d) / max(1, n)
                    )
            else:
                if _wants("mean_step_dist_frac"):
                    step_updates["mean_step_dist_frac"] = 0.0
                if _wants("median_step_dist_frac"):
                    step_updates["median_step_dist_frac"] = 0.0

        if _wants("mono_full") or _wants("mono_excl_last"):
            mono_full = bool(is_monotonic_down_up(segs))
            mono_excl_last = False
            if (not mono_full) and n_steps > 1:
                mono_excl_last = bool(is_monotonic_down_up(segs[:-1]))
            if _wants("mono_full"):
                step_updates["mono_full"] = bool(mono_full)
            if _wants("mono_excl_last"):
                step_updates["mono_excl_last"] = bool(mono_excl_last)

        feat.update(step_updates)

    def _to_scalar(v):
        if v is None:
            return np.nan
        if isinstance(v, (np.ndarray, list, tuple)):
            arr = np.asarray(v)
            if arr.size == 0:
                return 0.0
            if arr.size == 1:
                try:
                    return float(arr.ravel()[0])
                except Exception:
                    return np.nan
            try:
                return float(np.nanmean(arr.astype(float)))
            except Exception:
                return np.nan
        if isinstance(v, (np.generic,)):
            try:
                return v.item()
            except Exception:
                pass
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except Exception:
            return np.nan

    for k, v in list(feat.items()):
        feat[k] = _to_scalar(v)

    return feat


def fast_slope(y):
    n = y.size
    if n < 2:
        return 0.0, 0.0
    idx = np.arange(n, dtype=float)
    sum_x = n * (n - 1) / 2.0  # sum(idx)
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6.0  # sum(idx^2)
    sum_y = np.nansum(y)
    sum_xy = np.nansum(idx * y)
    denom = sum_x2 - (sum_x**2) / n
    if denom == 0:
        return 0.0, float(sum_y / n)
    slope = (sum_xy - (sum_x * sum_y) / n) / denom
    intercept = (sum_y - slope * sum_x) / n
    return float(slope), float(intercept)


def ac_lag1_fast(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2:
        return 0.0
    mean = np.nanmean(y)
    denom = np.nansum((y - mean) ** 2)
    if denom <= 0:
        return 0.0
    num = np.nansum((y[:-1] - mean) * (y[1:] - mean))
    return float(num / denom)


def percentile_k(y, k):
    if y.size == 0:
        return np.nan
    idx = int(np.floor(k / 100.0 * (y.size - 1)))
    return float(np.partition(y, idx)[idx])


def aeon_minirocket_features(
    sigs, num_kernels=10000, max_dilations_per_kernel=32, n_jobs=1, random_state=0
):
    mr = MiniRocket(
        n_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    X_mr = mr.fit_transform(sigs)
    X_mr = np.asarray(X_mr)
    n_feats = X_mr.shape[1]
    feat_names = [f"MiniRocket_{i}" for i in range(n_feats)]
    df_mr = pd.DataFrame(X_mr, columns=feat_names)
    return df_mr, feat_names


def dtw_derived_features_from_df(df_dtw, template_names=None, n_pca_components=None):
    # basic stats
    df = df_dtw.values  # N x M numpy
    N, M = df.shape

    df_log = np.log1p(df)  # shape N x M

    row_min = np.min(df, axis=1)
    row_2nd = np.partition(df, 1, axis=1)[:, 1] if M > 1 else np.full(N, np.nan)
    row_max = np.max(df, axis=1)
    row_mean = np.mean(df, axis=1)
    row_median = np.median(df, axis=1)
    row_std = np.std(df, axis=1, ddof=0)
    row_mad = np.median(np.abs(df - row_median[:, None]), axis=1)
    row_iqr = np.percentile(df, 75, axis=1) - np.percentile(df, 25, axis=1)
    row_p25 = np.percentile(df, 25, axis=1)
    row_p75 = np.percentile(df, 75, axis=1)
    row_p90 = np.percentile(df, 90, axis=1)
    row_p99 = np.percentile(df, 99, axis=1)
    row_skew = skew(df, axis=1, bias=False)
    row_kurt = kurtosis(df, axis=1, fisher=True, bias=False)

    # nearest/template-based features
    argmin_idx = np.argmin(df, axis=1)  # integer 0..M-1
    if template_names is None:
        template_names = list(df_dtw.columns)
    nearest_template = [template_names[i] for i in argmin_idx]

    second_min = row_2nd
    min_ratio = row_min / (second_min + 1e-12)  # small eps to avoid div0
    min_margin = second_min - row_min

    # counts within thresholds
    th1 = 1.5 * row_median
    th2 = 2.0 * row_median
    count_within_th1 = np.sum(df <= th1[:, None], axis=1)
    prop_within_th1 = count_within_th1 / float(M)

    # per-template z-score features: compute column medians / MAD
    col_med = np.median(df, axis=0)
    col_mad = np.median(np.abs(df - col_med[None, :]), axis=0) + 1e-12
    df_z = (df - col_med[None, :]) / col_mad[None, :]
    mean_col_z = np.mean(df_z, axis=1)
    max_col_z = np.max(df_z, axis=1)
    min_col_z = np.min(df_z, axis=1)

    # softmax-like similarity + entropy
    # choose alpha = 1 / median(row_median^2) or 1/median^2
    global_median = np.median(row_median) + 1e-12
    alpha = 1.0 / (global_median**2)
    # compute softmax over -alpha * d  => higher when close
    exp_sim = np.exp(-alpha * df)  # N x M
    sum_exp = np.sum(exp_sim, axis=1)
    p = exp_sim / (sum_exp[:, None] + 1e-12)
    # entropy
    ent = -np.sum(p * np.log(p + 1e-12), axis=1)
    max_p = np.max(p, axis=1)

    beta = 1.0 / (global_median**2)
    rbf_mean = np.mean(np.exp(-beta * (df**2)), axis=1)
    rbf_max = np.max(np.exp(-beta * (df**2)), axis=1)

    out = {
        "dtw_min": row_min,
        "dtw_2nd_min": second_min,
        "dtw_min_ratio": min_ratio,
        "dtw_min_margin": min_margin,
        "dtw_max": row_max,
        "dtw_mean": row_mean,
        "dtw_median": row_median,
        "dtw_std": row_std,
        "dtw_mad": row_mad,
        "dtw_iqr": row_iqr,
        "dtw_p25": row_p25,
        "dtw_p75": row_p75,
        "dtw_p90": row_p90,
        "dtw_p99": row_p99,
        "dtw_skew": row_skew,
        "dtw_kurt": row_kurt,
        "dtw_count_within_1.5xmed": count_within_th1,
        "dtw_prop_within_1.5xmed": prop_within_th1,
        "dtw_mean_col_z": mean_col_z,
        "dtw_max_col_z": max_col_z,
        "dtw_min_col_z": min_col_z,
        "dtw_softmax_entropy": ent,
        "dtw_softmax_max": max_p,
        "dtw_rbf_mean": rbf_mean,
        "dtw_rbf_max": rbf_max,
    }

    df_feats = pd.DataFrame(out)
    if n_pca_components is not None and n_pca_components > 0:
        pca = PCA(n_components=min(n_pca_components, M))
        pcs = pca.fit_transform(df_log)  # use log-distances for PCA
        for i in range(pcs.shape[1]):
            df_feats[f"dtw_pca_{i+1}"] = pcs[:, i]

    return df_feats


def _resolve_requested_feature_names(feature_set="all", selected_features=None):
    if selected_features is not None:
        requested = [str(f) for f in selected_features if str(f)]
        return list(dict.fromkeys(requested))

    feature_set_key = str(feature_set or "all").strip().lower()
    if feature_set_key not in FEATURE_SET_PRESETS:
        allowed = ", ".join(sorted(FEATURE_SET_PRESETS.keys()))
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. Allowed values: {allowed}"
        )

    preset = FEATURE_SET_PRESETS[feature_set_key]
    if preset is None:
        return None
    return list(preset)


def _required_feature_blocks_from_selection(selected_features):
    names = set(selected_features or [])

    if not names:
        return {
            "use_catch22": False,
            "basic_params": False,
            "basic_5": False,
            "slope_features": False,
            "drop_features": False,
            "peak_features": False,
            "trend_features": False,
            "spectral_features": False,
            "second_derivative_features": False,
            "rolling_window_slope_features": False,
            "step_detection_features": False,
            "multiscale_features": False,
        }

    def _any(pred):
        return any(pred(n) for n in names)

    required = {
        "use_catch22": _any(lambda n: n.startswith("c22_")),
        "basic_params": _any(lambda n: n in _BASIC_PARAM_FEATURE_NAMES),
        "basic_5": _any(lambda n: n in _BASIC_5_FEATURE_NAMES),
        "slope_features": _any(lambda n: n in _SLOPE_FEATURE_NAMES),
        "drop_features": _any(lambda n: n.startswith("drop_")),
        "peak_features": _any(lambda n: n in _PEAK_FEATURE_NAMES),
        "trend_features": _any(lambda n: n in _TREND_FEATURE_NAMES),
        "spectral_features": _any(
            lambda n: (n in _SPECTRAL_FEATURE_NAMES) or n.startswith("spectral_")
        ),
        "second_derivative_features": _any(
            lambda n: (n in _SECOND_DERIVATIVE_FEATURE_NAMES) or n.startswith("second_")
        ),
        "rolling_window_slope_features": _any(lambda n: "rolling_slope" in n),
        "step_detection_features": _any(
            lambda n: (n in _STEP_EXACT_FEATURE_NAMES) or ("step_" in n)
        ),
        "multiscale_features": _any(
            lambda n: n.startswith("hjorth_")
            or n.startswith("cross_rate_")
            or bool(re.match(r"^ms\d+_", n))
        ),
    }

    if required["drop_features"]:
        required["basic_params"] = True

    return required


def _reduce_feature_set_compact(
    features_df,
    compact_non_c22_features=None,
    keep_catch22_features=True,
    keep_extra_feature_prefixes=(),
):
    if features_df is None or features_df.shape[1] == 0:
        return features_df

    cols = list(features_df.columns)
    compact = compact_non_c22_features or COMPACT_NON_C22_FEATURES_35
    keep = [c for c in compact if c in cols]

    if keep_catch22_features:
        keep.extend([c for c in cols if c.startswith("c22_")])

    if keep_extra_feature_prefixes:
        for prefix in keep_extra_feature_prefixes:
            keep.extend([c for c in cols if c.startswith(prefix)])

    seen = set()
    keep_unique = []
    for c in keep:
        if c not in seen:
            seen.add(c)
            keep_unique.append(c)

    if not keep_unique:
        return features_df

    return features_df.loc[:, keep_unique]


def _extract_feature_chunk(
    sig_chunk,
    thresholds,
    use_catch22,
    basic_params,
    drop_window,
    basic_5,
    slope_features,
    drop_features,
    peak_features,
    trend_features,
    spectral_features,
    second_derivative_features,
    rolling_window_slope_features,
    step_detection_features,
    multiscale_features,
    requested_features_set=None,
):
    out = []
    for sig in sig_chunk:
        out.append(
            _extract_one_feature(
                sig,
                thresholds,
                use_catch22,
                basic_params,
                drop_window,
                basic_5,
                slope_features,
                drop_features,
                peak_features,
                trend_features,
                spectral_features,
                second_derivative_features,
                rolling_window_slope_features,
                step_detection_features,
                multiscale_features,
                requested_features_set,
            )
        )
    return out


def extract_interpretable_features(
    signals,
    signals_padded=None,
    thresholds=None,
    use_catch22=True,
    use_dtw_medoid=False,
    use_tsfel=False,
    basic_params=True,
    use_minirocket=False,
    drop_window=30,
    dtw_templates=None,
    dtw_template_names=None,
    dtw_window_frac=0.10,
    n_jobs=None,
    dtw_norm="path",
    disable_spectral=False,
    basic_5=True,
    slope_features=True,
    drop_features=True,
    peak_features=True,
    trend_features=True,
    spectral_features=False,
    second_derivative_features=True,
    rolling_window_slope_features=True,
    step_detection_features=True,
    multiscale_features=True,
    compact_feature_set=False,
    compact_non_c22_features=None,
    keep_catch22_features=True,
    keep_extra_feature_prefixes=(),
    feature_set="all",
    selected_features=None,
    chunk_size=512,
    parallel_backend="threads",
):

    if thresholds is None:
        thresholds = {"abs_drop": -1, "low": -2, "high": 0}

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    requested_features = _resolve_requested_feature_names(
        feature_set=feature_set,
        selected_features=selected_features,
    )

    if requested_features is not None:
        required = _required_feature_blocks_from_selection(requested_features)
        use_catch22 = required["use_catch22"]
        basic_params = required["basic_params"]
        basic_5 = required["basic_5"]
        slope_features = required["slope_features"]
        drop_features = required["drop_features"]
        peak_features = required["peak_features"]
        trend_features = required["trend_features"]
        spectral_features = required["spectral_features"]
        second_derivative_features = required["second_derivative_features"]
        rolling_window_slope_features = required["rolling_window_slope_features"]
        step_detection_features = required["step_detection_features"]
        multiscale_features = required["multiscale_features"]

    sigs = list(signals)
    if not sigs:
        return pd.DataFrame(), []

    prefer_mode = (
        "threads" if str(parallel_backend).lower().startswith("thread") else "processes"
    )

    if chunk_size is None or chunk_size < 1:
        chunk_size = 1
    if len(sigs) <= chunk_size:
        results = _extract_feature_chunk(
            sigs,
            thresholds,
            use_catch22,
            basic_params,
            drop_window,
            basic_5,
            slope_features,
            drop_features,
            peak_features,
            trend_features,
            spectral_features,
            second_derivative_features,
            rolling_window_slope_features,
            step_detection_features,
            multiscale_features,
            requested_features,
        )
    else:
        chunks = [sigs[i : i + chunk_size] for i in range(0, len(sigs), chunk_size)]
        chunked = Parallel(
            n_jobs=n_jobs, prefer=prefer_mode, batch_size=1, pre_dispatch="2*n_jobs"
        )(
            delayed(_extract_feature_chunk)(
                ch,
                thresholds,
                use_catch22,
                basic_params,
                drop_window,
                basic_5,
                slope_features,
                drop_features,
                peak_features,
                trend_features,
                spectral_features,
                second_derivative_features,
                rolling_window_slope_features,
                step_detection_features,
                multiscale_features,
                requested_features,
            )
            for ch in chunks
        )
        results = [item for sub in chunked for item in sub]

    features_df = pd.DataFrame(results).sort_index(axis=1)

    if dtw_templates is not None and len(dtw_templates) > 0:
        df_dtw = dtw_features_to_templates(
            sigs,
            dtw_templates,
            template_names=dtw_template_names,
            window_frac=dtw_window_frac,
            n_jobs=n_jobs,
            norm=dtw_norm,
        )
        df_dtw_feats = dtw_derived_features_from_df(
            df_dtw, template_names=list(df_dtw.columns), n_pca_components=10
        )
        features_df = pd.concat(
            [features_df.reset_index(drop=True), df_dtw_feats.reset_index(drop=True)],
            axis=1,
        )

    if use_minirocket and signals_padded is not None:
        df_mr, mr_names = aeon_minirocket_features(
            signals_padded, num_kernels=10000, n_jobs=n_jobs
        )
        features_df = pd.concat(
            [features_df.reset_index(drop=True), df_mr.reset_index(drop=True)], axis=1
        )

    if compact_feature_set:
        n_before = features_df.shape[1]
        features_df = _reduce_feature_set_compact(
            features_df=features_df,
            compact_non_c22_features=compact_non_c22_features,
            keep_catch22_features=keep_catch22_features,
            keep_extra_feature_prefixes=keep_extra_feature_prefixes,
        )

    if requested_features is not None:
        features_df = features_df.reindex(columns=requested_features, fill_value=np.nan)

    feature_names = list(features_df.columns)
    return features_df, feature_names


def _seg_stats_from_prefix(ps, ss, a, b, eps=1e-12):
    L = b - a
    if L <= 0:
        return 0.0, eps, L
    s = ps[b] - ps[a]
    m = s / L
    sq = ss[b] - ss[a]
    var = sq / L - m * m
    if var <= 0 or not np.isfinite(var):
        var = eps * eps
    return float(m), float(math.sqrt(var)), L


def segment_signal(signal, threshold=18, min_len=10, step=1, eps=1e-12):
    sig = np.asarray(signal, dtype=float).flatten()
    n = sig.size
    if n == 0:
        return []
    if n < 2 * min_len:
        m = float(sig.mean()) if n > 0 else 0.0
        s = float(sig.std()) if n > 0 else eps
        return [(0, n, m, s, n)]

    ps = np.zeros(n + 1, dtype=float)
    ss = np.zeros(n + 1, dtype=float)
    ps[1:] = np.cumsum(sig)
    ss[1:] = np.cumsum(sig * sig)

    out = []
    stack = [(0, n)]
    log_eps = math.log(eps)

    while stack:
        a, b = stack.pop()
        t = b - a
        if t < 2 * min_len:
            m, s_, L = _seg_stats_from_prefix(ps, ss, a, b, eps=eps)
            out.append((a, b, m, s_, int(L)))
            continue

        # compute total std (scalar)
        _, std_total, _ = _seg_stats_from_prefix(ps, ss, a, b, eps=eps)
        std_total = max(std_total, eps)
        # candidate split positions (absolute indices)
        cand = np.arange(a + min_len, b - min_len, step)
        if cand.size == 0:
            m, s_, L = _seg_stats_from_prefix(ps, ss, a, b, eps=eps)
            out.append((a, b, m, s_, int(L)))
            continue

        left_L = cand - a
        right_L = b - cand

        left_sum = ps[cand] - ps[a]
        left_sq = ss[cand] - ss[a]
        left_mean = left_sum / left_L
        left_var = left_sq / left_L - left_mean * left_mean
        left_var = np.where(left_var > 0, left_var, eps * eps)
        left_std = np.sqrt(left_var)

        right_sum = ps[b] - ps[cand]
        right_sq = ss[b] - ss[cand]
        right_mean = right_sum / right_L
        right_var = right_sq / right_L - right_mean * right_mean
        right_var = np.where(right_var > 0, right_var, eps * eps)
        right_std = np.sqrt(right_var)

        left_log = np.log(left_std + eps)
        right_log = np.log(right_std + eps)
        total_log = math.log(std_total + eps)
        scores = t * total_log - (left_L * left_log + right_L * right_log)

        imax = int(np.argmax(scores))
        max_score = float(scores[imax])
        split = int(cand[imax])

        if max_score > threshold:
            stack.append((split, b))
            stack.append((a, split))
        else:
            m, s_, L = _seg_stats_from_prefix(ps, ss, a, b, eps=eps)
            out.append((a, b, m, s_, int(L)))

    out.sort(key=lambda x: x[0])
    return out


def is_monotonic_down_up(segments, tol=1e-8):
    if not segments:
        return False
    means = np.asarray([s[2] for s in segments], dtype=float)

    if means.size >= 2:
        d = np.diff(means)
        if np.all(d >= -tol) or np.all(d <= tol):
            return True

    def _down_then_up(m):
        if m.size < 3:
            return False
        mi = int(np.argmin(m))
        if mi == 0 or mi == m.size - 1:
            return False
        if np.any(np.diff(m[: mi + 1]) > tol):
            return False
        if np.any(np.diff(m[mi:]) < -tol):
            return False
        return True

    if _down_then_up(means):
        return True
    if means.size > 1 and _down_then_up(means[:-1]):
        return True

    return False
