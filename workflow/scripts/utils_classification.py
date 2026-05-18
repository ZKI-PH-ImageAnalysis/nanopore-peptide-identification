#!/usr/bin/env python

import os

# Suppress low-level startup logs
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import csv
import inspect
from functools import lru_cache
import logging
import random
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aeon.classification.convolution_based import (
    MiniRocketClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier,
)
from aeon.classification.dictionary_based import MUSE, WEASEL
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.interval_based import QUANTClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    import polars as pl
except Exception:
    pl = None


def _import_tensorflow_silenced():
    silence = os.environ.get("NP_SILENCE_TF_STARTUP", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    if not silence:
        import tensorflow as _tf

        return _tf, _tf.keras.layers, _tf.keras.models, _tf.keras.callbacks.Callback

    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        import tensorflow as _tf

        _layers = _tf.keras.layers
        _models = _tf.keras.models
        _callback = _tf.keras.callbacks.Callback
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)
    return _tf, _layers, _models, _callback


@lru_cache(maxsize=1)
def _import_aeon_deep_learning_silenced():
    silence = os.environ.get("NP_SILENCE_TF_STARTUP", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    if not silence:
        from aeon.classification.deep_learning import (
            InceptionTimeClassifier as _InceptionTimeClassifier,
        )
        from aeon.classification.deep_learning import (
            ResNetClassifier as _ResNetClassifier,
        )

        return _InceptionTimeClassifier, _ResNetClassifier

    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        from aeon.classification.deep_learning import (
            InceptionTimeClassifier as _InceptionTimeClassifier,
        )
        from aeon.classification.deep_learning import (
            ResNetClassifier as _ResNetClassifier,
        )
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)
    return _InceptionTimeClassifier, _ResNetClassifier


tf, layers, keras_models, Callback = _import_tensorflow_silenced()
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from utils_feature_extraction import (
    extract_interpretable_features,
)
from utils_model_architectures import NativeInceptionTimeClassifier, TimesNetClassifier

from utils_data_preprocessing import (
    apply_step_detection,
    apply_smoothing,
    compute_train_stats,
    log_length_distributions,
    normalize_signals,
    pad_trim_or_resample,
    prepare_data,
    prepare_dataset_from_df,
    process_parse_inputfiles,
    resample_signals_to_length,
    trim_edges,
)

from utils_plot_classification import (
    plot_confusion_matrices,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nanopore-peptide-classifier")
tf.get_logger().setLevel("ERROR")

DEFAULT_FEATURE_KWARGS = {
    "use_catch22": True,
    "basic_params": True,
    "use_tsfel": False,
    "use_minirocket": False,
    "use_dtw_medoid": False,
    "dtw_templates": None,
    "dtw_template_names": None,
    "dtw_window_frac": 0.1,
    "dtw_norm": "path",
    "basic_5": True,
    "slope_features": True,
    "drop_features": True,
    "peak_features": True,
    "trend_features": True,
    "spectral_features": False,
    "second_derivative_features": False,
    "rolling_window_slope_features": True,
    "feature_set": "all",
}


def is_inception_model(model_name):
    return str(model_name) in ("InceptionTime", "InceptionTime-large", "TimesNet")


def resolve_inception_preprocessing(ds_kwargs, models_list):
    if not any(is_inception_model(m) for m in models_list):
        return dict(ds_kwargs), {}

    inception_kwargs = dict(ds_kwargs)
    changed = {}

    if inception_kwargs.get("smoothing") in (None, "None"):
        inception_kwargs["smoothing"] = "low-pass-bessel"
        changed["smoothing"] = "low-pass-bessel"

    return inception_kwargs, changed


def should_use_weighted_inception_loss(
    y_train_labels,
    imbalance_threshold=2.0,
    weighted_loss_mode="auto",
):
    mode = str(weighted_loss_mode or "auto").strip().lower()
    if mode in {"on", "true", "1", "yes"}:
        return True
    if mode in {"off", "false", "0", "no"}:
        return False

    imbalance_threshold = float(imbalance_threshold)
    labels = np.asarray(y_train_labels)
    if labels.size == 0:
        return False
    counts = pd.Series(labels).value_counts()
    if counts.empty:
        return False
    min_count = max(float(counts.min()), 1.0)
    imbalance_ratio = float(counts.max()) / min_count
    return imbalance_ratio >= float(imbalance_threshold)


def flatten_numeric_metrics(metrics, prefix):
    flat = {}
    if not isinstance(metrics, dict):
        return flat
    for key, value in metrics.items():
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            val = float(value)
            if np.isfinite(val):
                flat[f"{prefix} :: {key}"] = val
    return flat


def summarize_model_seed_runs(model_name, model_runs, outdir):
    if not model_runs:
        return 1

    rows = []
    for run in model_runs:
        row = {
            "run_index": int(run["run_index"]),
            "run_seed": int(run["run_seed"]),
            "run_dir": str(run["run_dir"]),
        }
        row.update(flatten_numeric_metrics(run.get("val"), "Validation"))
        row.update(flatten_numeric_metrics(run.get("test"), "Test"))
        rows.append(row)

    run_df = pd.DataFrame(rows)
    run_tsv = os.path.join(outdir, f"{model_name}-seed-runs.tsv")
    run_df.to_csv(run_tsv, sep="\t", index=False)

    metric_preference = [
        "Validation :: Accuracy (w/o unknown)",
        "Validation :: Accuracy",
        "Validation :: Balanced Accuracy (w/o unknown)",
        "Validation :: Macro F1-Score",
        "Test :: Accuracy (w/o unknown)",
        "Test :: Accuracy",
    ]

    selected_metric = None
    selected_values = None
    for metric_name in metric_preference:
        if metric_name in run_df.columns:
            vals = pd.to_numeric(run_df[metric_name], errors="coerce").dropna()
            if len(vals) > 0:
                selected_metric = metric_name
                selected_values = vals
                break

    if selected_metric is None:
        candidate_cols = [
            c for c in run_df.columns if c not in {"run_index", "run_seed", "run_dir"}
        ]
        for metric_name in candidate_cols:
            vals = pd.to_numeric(run_df[metric_name], errors="coerce").dropna()
            if len(vals) > 0:
                selected_metric = metric_name
                selected_values = vals
                break

    median_run_index = int(run_df["run_index"].iloc[0]) if len(run_df) else 1
    if (
        selected_metric is not None
        and selected_values is not None
        and len(selected_values) > 0
    ):
        median_value = float(np.median(selected_values.values))
        candidate = run_df.loc[
            selected_values.index, ["run_index", selected_metric]
        ].copy()
        candidate["_abs_diff"] = (candidate[selected_metric] - median_value).abs()
        candidate = candidate.sort_values(["_abs_diff", "run_index"], kind="mergesort")
        median_run_index = int(candidate.iloc[0]["run_index"])

    metric_cols = [
        c for c in run_df.columns if c not in {"run_index", "run_seed", "run_dir"}
    ]
    summary_rows = []
    for metric_name in metric_cols:
        vals = pd.to_numeric(run_df[metric_name], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        summary_rows.append(
            {
                "model": model_name,
                "metric": metric_name,
                "n_runs": int(len(vals)),
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(vals.median()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "median_selection_metric": selected_metric if selected_metric else "",
                "median_run_index": int(median_run_index),
                "median_run_seed": int(
                    run_df.loc[
                        run_df["run_index"] == median_run_index, "run_seed"
                    ].iloc[0]
                ),
                "median_run_dir": str(
                    run_df.loc[run_df["run_index"] == median_run_index, "run_dir"].iloc[
                        0
                    ]
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_tsv = os.path.join(outdir, f"{model_name}-seed-summary.tsv")
    summary_df.to_csv(summary_tsv, sep="\t", index=False)

    logger.info(
        "[%s] Seed-run summary saved: %s (median run: %d)",
        model_name,
        os.path.basename(summary_tsv),
        int(median_run_index),
    )
    return int(median_run_index)


def run_dtw_plots(
    X_all,
    y_all,
    run_ids_all,
    scen_out,
    scenario_name,
    testing,
    seed,
    num_processes,
    fixed_length,
):
    from utils_dtw_alignment import dtw_plots_figure

    logger.info("[%s] Plotting DTW distances", scenario_name)
    sep_scenarios = (
        "ßCAT_binary_D",
        "ßCAT_binary_W",
        "ßCAT_alone",
        "ßCAT_single_variants",
    )
    runs = [(False, "dtw-analysis")]
    if scenario_name in sep_scenarios:
        runs.append((True, "dtw-analysis-run-rep-separate"))
    for run_rep_separate, subdir in runs:
        out = os.path.join(scen_out, subdir)
        os.makedirs(out, exist_ok=True)
        dtw_plots_figure(
            X_all,
            y_all,
            selected_classes=None,
            per_class=testing,
            seed=seed,
            n_jobs=num_processes,
            outdir=out,
            resamp_len=fixed_length,
            run_rep_separate=run_rep_separate,
            run_ids=run_ids_all,
        )


def run_model(
    mname,
    model_out,
    ds,
    ds_test,
    *,
    class_weights_array,
    n_epochs,
    num_processes,
    scenario_name,
    padding_mode,
    fixed_length,
    seed,
    feature_test=False,
    feature_set="all",
    feature_backend="auto",
    feature_chunk_size=0,
    inception_weighted_loss_mode="auto",
    inception_imbalance_threshold=2.0,
    inception_batch_size=0,
    inception_large_batch_size=0,
    timesnet_batch_size=0,
    inception_early_stopping=True,
    inception_early_stopping_patience=14,
    inception_early_stopping_min_delta=0.0,
):
    random.seed(seed)
    np.random.seed(seed)

    if mname in ("InceptionTime", "InceptionTime-large", "TimesNet"):
        imbalance_threshold = float(inception_imbalance_threshold)
        if str(mname) == "TimesNet":
            imbalance_threshold = min(imbalance_threshold, 1.35)
        use_weighted_loss = should_use_weighted_inception_loss(
            ds["y_train_IT"],
            imbalance_threshold=imbalance_threshold,
            weighted_loss_mode=inception_weighted_loss_mode,
        )
        loss_fn = (
            weighted_categorical_crossentropy(class_weights_array)
            if use_weighted_loss
            else None
        )
        logger.info(
            "[%s] %s loss mode: %s",
            scenario_name,
            mname,
            "weighted_categorical_crossentropy" if use_weighted_loss else "default",
        )
        model = get_model(
            mname,
            n_epochs=n_epochs,
            n_jobs=num_processes,
            loss_fn=loss_fn,
            n_samples=len(ds["y_train_IT"]),
            inception_batch_size=inception_batch_size,
            inception_large_batch_size=inception_large_batch_size,
            timesnet_batch_size=timesnet_batch_size,
            inception_early_stopping=inception_early_stopping,
            inception_early_stopping_patience=inception_early_stopping_patience,
            inception_early_stopping_min_delta=inception_early_stopping_min_delta,
            random_state=seed,
        )
        model = fit_model_safe(
            model,
            ds["X_train_IT"],
            ds["y_train_IT"],
            X_val=ds["X_val_IT"],
            y_val=ds["y_val_IT"],
        )
        X_t, y_t = ds_test.get("X_IT"), ds_test.get("y_IT")
        val_m = evaluate_model(
            model, ds["X_val_IT"], ds["y_val_IT"], "Validation", model_out
        )
        test_m = (
            evaluate_model(model, X_t, y_t, "Test", model_out)
            if X_t is not None
            else None
        )

        interp_scenarios = {
            "ALL_classes": ["ßCAT", "ßCATW", "ßCATWW", "ßCATWWW"],
            "ßCAT_single_variants": ["ßCAT", "ßCATD", "ßCATL", "ßCATW"],
        }
        if scenario_name in interp_scenarios and mname in (
            "InceptionTime",
            "InceptionTime-large",
        ):
            try:
                from utils_interpretability import (
                    run_inception_interpretability_pipeline,
                )

                y_pred_val = model.predict(ds["X_val_IT"])
                run_inception_interpretability_pipeline(
                    model=model,
                    model_name=mname,
                    X_val_raw=ds.get("X_val_raw", ds["X_val"]),
                    y_val=ds["y_val_IT"],
                    y_pred=y_pred_val,
                    val_run_ids=ds.get("y_val_runID"),
                    output_dir=model_out,
                    scenario_name=scenario_name,
                    seed=seed,
                    specific_peptides=interp_scenarios[scenario_name],
                    resample_len=fixed_length,
                )
            except Exception as e:
                logger.warning(
                    "[%s] InceptionTime interpretability failed: %s",
                    scenario_name,
                    str(e),
                )

    elif mname.lower().startswith("feature") or "rf" in mname.lower():
        aliases = {
            "lgbm": "LGBM",
            "rf": "RandomForest",
            "svm": "SVM",
            "rc": "RC",
            "xgb": "XGBoost",
            "mlp": "MLP",
            "knn": "KNN",
            "tabpfn": "TabPFN",
        }
        base_name = re.sub(r"^feature[_-]?", "", mname, flags=re.IGNORECASE)
        match = re.search(r"([A-Za-z]+)", base_name)
        model_type = aliases.get((match.group(1).lower() if match else "lgbm"), "LGBM")
        logger.info("[model] %s -> %s", mname, model_type)

        feat_kw = {
            **DEFAULT_FEATURE_KWARGS,
            "n_jobs": num_processes,
            "feature_set": str(feature_set).lower(),
            "parallel_backend": str(feature_backend).lower(),
            "chunk_size": int(feature_chunk_size),
        }
        if feat_kw["use_dtw_medoid"]:
            from utils_dtw_alignment import compute_class_medoids

            templates, template_names, _ = compute_class_medoids(
                ds["X_train_raw"],
                ds["y_train"],
                n_medoids=1,
                window_frac=0.10,
                n_jobs=num_processes,
                norm="path",
                total_target=2000,
            )
            feat_kw.update(dtw_templates=templates, dtw_template_names=template_names)

        feature_test_enabled = feature_test and model_type == "LGBM"
        train_result = train_on_features(
            ds["X_train_raw"],
            ds["y_train"],
            signals_padded=ds["X_train"],
            padding_mode=padding_mode,
            target_length=fixed_length,
            X_valid_raw=ds["X_val"],
            y_valid=ds["y_val"],
            output_path=model_out,
            n_jobs=num_processes,
            model_type=model_type,
            feature_kwargs=feat_kw,
            random_state=seed,
            return_feature_data=feature_test_enabled,
        )

        feature_train_data = None
        if isinstance(train_result, tuple):
            model, feature_train_data = train_result
        else:
            model = train_result

        val_m = evaluate_model(
            model, ds["X_val_raw"], ds["y_val"], "Validation", model_out
        )
        X_t_raw, y_t = ds_test.get("X_raw"), ds_test.get("y")
        test_m = (
            evaluate_model(model, X_t_raw, y_t, "Test", model_out)
            if X_t_raw is not None
            else None
        )

        if feature_test_enabled:
            feature_test_out = os.path.join(model_out, "feature-test")
            try:
                run_feature_reduction_protocol(
                    X_train_raw=ds["X_train_raw"],
                    y_train=ds["y_train"],
                    X_val_raw=ds["X_val_raw"],
                    y_val=ds["y_val"],
                    output_dir=feature_test_out,
                    feature_kwargs=feat_kw,
                    model_type=model_type,
                    n_jobs=num_processes,
                    random_state=seed,
                    signals_padded_train=ds["X_train"],
                    signals_padded_val=ds["X_val"],
                    top_ks=(50, 40, 30, 25, 20, 10),
                    precomputed_train_df=(feature_train_data or {}).get(
                        "train_features_df"
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "[%s] feature-test failed for %s: %s", scenario_name, mname, exc
                )

        interp = {
            "ALL_classes": ["ßCAT", "ßCATW", "ßCATWW", "ßCATWWW"],
            "ßCAT_single_variants": ["ßCAT", "ßCATD", "ßCATL", "ßCATW"],
        }
        if scenario_name in interp:
            from utils_interpretability import run_interpretability_pipeline

            run_interpretability_pipeline(
                model=model,
                model_name=mname,
                X_train_raw=ds["X_train"],
                y_train=ds["y_train"],
                X_val_raw=ds["X_val"],
                y_val=ds["y_val"],
                val_metrics=val_m,
                output_dir=model_out,
                scenario_name=scenario_name,
                seed=seed,
                val_run_ids=ds.get("y_val_runID"),
                feature_kwargs=feat_kw,
                specific_peptides=interp[scenario_name],
                n_jobs=num_processes,
            )

    else:
        X_t, y_t = ds_test.get("X"), ds_test.get("y")
        model = get_model(
            mname,
            n_epochs=n_epochs,
            n_jobs=num_processes,
            n_samples=len(ds["y_train"]),
            random_state=seed,
        )
        model = fit_model_safe(
            model, ds["X_train"], ds["y_train"], X_val=ds["X_val"], y_val=ds["y_val"]
        )
        val_m = evaluate_model(model, ds["X_val"], ds["y_val"], "Validation", model_out)
        test_m = (
            evaluate_model(model, X_t, y_t, "Test", model_out)
            if X_t is not None
            else None
        )

    if val_m is not None and test_m is not None:
        plot_confusion_matrices(
            model_out, val_m, test_m, outfile_name="confusion_matrices",
            scenario_name=scenario_name
        )
    return {"model": model, "val": val_m, "test": test_m}


def all_classes_postprocessing(
    results_all_scenarios, X_all, y_all, train_sub, val_sub, test_df, scen_out
):
    from utils_plot_physchem import (
        create_feature_vs_physchem_supplement,
        create_physchem_vs_perf_supplement,
    )

    def _counts(df):
        return (
            df["Class"].value_counts().sort_index()
            if (df is not None and "Class" in df.columns)
            else pd.Series(dtype=int)
        )

    train_c, val_c, test_c = _counts(train_sub), _counts(val_sub), _counts(test_df)
    trainval_c = train_c.add(val_c, fill_value=0).astype(int)
    counts_dict = dict(
        train=train_c.to_dict(),
        val=val_c.to_dict(),
        test=test_c.to_dict(),
        trainval=trainval_c.to_dict(),
        total=trainval_c.add(test_c, fill_value=0).astype(int).to_dict(),
    )

    fig_out = os.path.join(scen_out, "PhysChemFigures")
    os.makedirs(fig_out, exist_ok=True)
    try:
        create_physchem_vs_perf_supplement(
            results_all_scenarios,
            output_dir=fig_out,
            dataset_name="Validation",
            counts=counts_dict,
        )
    except RuntimeError as exc:
        logger.warning("Skipping physchem-vs-performance supplement: %s", exc)

    create_feature_vs_physchem_supplement(
        X=X_all,
        y=y_all,
        n_features=4,
        outdir=os.path.join(scen_out, "PhysChemFigures-Features"),
    )


def _resolve_feature_backend(n_signals, requested_backend=None):
    """Return an effective parallel backend for feature extraction."""
    if requested_backend:
        backend = str(requested_backend).strip().lower()
        if backend in {"threads", "processes"}:
            return backend
    return "processes" if n_signals >= 20000 else "threads"


def _prepare_feature_options(feature_kwargs, n_jobs, n_signals):
    """Merge feature kwargs with sensible performance defaults."""
    options = dict(feature_kwargs or {})
    options.setdefault("n_jobs", n_jobs)
    adaptive_chunk = max(512, int(n_signals // max(1, n_jobs * 16)))
    adaptive_chunk = min(adaptive_chunk, 8192)

    chunk_size = options.get("chunk_size", None)
    if chunk_size is None or int(chunk_size) <= 0:
        options["chunk_size"] = adaptive_chunk
    else:
        options["chunk_size"] = int(chunk_size)

    requested_backend = options.get("parallel_backend", None)
    if (
        requested_backend is not None
        and str(requested_backend).strip().lower() == "auto"
    ):
        requested_backend = None
    options["parallel_backend"] = _resolve_feature_backend(n_signals, requested_backend)
    return options


def _to_float_feature_matrix(features_df):
    """Convert extracted features to float matrix, coercing only when necessary."""
    try:
        return features_df.to_numpy(dtype=float, copy=False)
    except Exception:
        numeric_df = features_df.apply(pd.to_numeric, errors="coerce")
        return numeric_df.to_numpy(dtype=float, copy=False)


def _impute_nan_columns(X, fill_values=None):
    """Impute NaNs column-wise and return (imputed_matrix, used_fill_values)."""
    if fill_values is not None and len(fill_values) == X.shape[1]:
        col_fill = np.asarray(fill_values, dtype=float)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_fill = np.nanmedian(X, axis=0)
        col_fill = np.where(np.isfinite(col_fill), col_fill, 0.0)

    nan_idx = np.where(np.isnan(X))
    if nan_idx[0].size:
        X[nan_idx] = np.take(col_fill, nan_idx[1])
    return X, col_fill


def _build_feature_estimator(model_type, n_estimators, max_depth, n_jobs, random_state):
    if model_type == "LGBM":
        return LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=n_jobs,
            verbosity=-1,
            random_state=random_state,
            deterministic=True,
            force_col_wise=True,
        )
    if model_type == "RC":
        return RidgeClassifierCV(class_weight="balanced", alphas=np.logspace(-3, 3, 10))
    if model_type == "SVM":
        return SGDClassifier(
            loss="hinge",
            max_iter=2000,
            tol=1e-3,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_type == "KNN":
        return KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=n_jobs, p=2
        )

    logger.warning("Unknown model_type '%s', defaulting to RandomForest", model_type)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=random_state,
    )


def _export_feature_importance(model, feature_names, output_path):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(output_path, "feature_importances.csv"), index=False)

    top_k = min(30, fi_df.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=(8, max(4, 0.18 * top_k)))
    sns.barplot(x="importance", y="feature", data=fi_df.head(top_k), ax=ax)
    ax.set_title(f"Feature importances (top {top_k})")
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_path, "feature_importances.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


class FeatureModelWrapper:

    def __init__(
        self,
        estimator,
        scaler,
        feature_kwargs=None,
        feature_names=None,
        padding_mode=None,
        target_length=0,
        impute_values=None,
    ):
        self.model = estimator
        self.scaler = scaler
        self.feature_kwargs = feature_kwargs or {}
        self.feature_names = feature_names
        self.padding_mode = padding_mode
        self.target_length = target_length
        self.impute_values = impute_values
        self._feature_cache = {}

    @property
    def classes_(self):
        """Expose underlying estimator classes_ (if available)."""
        return getattr(self.model, "classes_", None)

    @staticmethod
    def _restore_vector_order(values, kept_idx, total_n):
        if kept_idx is None or len(kept_idx) == total_n:
            return values
        out = np.array([None] * total_n, dtype=object)
        out[kept_idx] = values
        return out

    @staticmethod
    def _restore_matrix_order(values, kept_idx, total_n):
        if kept_idx is None or len(kept_idx) == total_n:
            return values
        out = np.full((total_n, values.shape[1]), np.nan, dtype=float)
        out[kept_idx, :] = values
        return out

    def _features_from_raw(self, raw_signals):
        signals = list(raw_signals)
        padded_signals = None
        kept_idx = None
        if self.padding_mode is not None and self.target_length > 0:
            padded_signals, kept_idx, _ = pad_trim_or_resample(
                signals,
                mode=self.padding_mode,
                target_length=self.target_length,
            )

        feature_options = dict(self.feature_kwargs or {})
        # Avoid process startup overhead for small inference batches.
        if len(signals) < 5000 and str(
            feature_options.get("parallel_backend", "threads")
        ).startswith("process"):
            feature_options["parallel_backend"] = "threads"

        features_df, _ = extract_interpretable_features(
            signals, padded_signals, **feature_options
        )

        # Ensure columns follow training order
        if self.feature_names is not None:
            features_df = features_df.reindex(
                columns=self.feature_names, fill_value=np.nan
            )

        Xf = _to_float_feature_matrix(features_df)
        Xf, _ = _impute_nan_columns(Xf, fill_values=self.impute_values)

        return Xf, (None if kept_idx is None else np.asarray(kept_idx, dtype=int))

    def _get_cached_or_extract_features(self, raw_signals):
        key = (id(raw_signals), len(raw_signals))
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached

        Xf, kept_idx = self._features_from_raw(raw_signals)
        self._feature_cache[key] = (Xf, kept_idx)
        if len(self._feature_cache) > 4:
            self._feature_cache.pop(next(iter(self._feature_cache)))
        return Xf, kept_idx

    def _apply_scaler(self, Xf):
        if self.scaler is None:
            return Xf
        return self.scaler.transform(Xf)

    def predict(self, raw_signals):
        Xf, kept_idx = self._get_cached_or_extract_features(raw_signals)
        Xs = self._apply_scaler(Xf)
        preds = self.model.predict(Xs)
        return self._restore_vector_order(preds, kept_idx, len(raw_signals))

    def predict_proba(self, raw_signals):
        Xf, kept_idx = self._get_cached_or_extract_features(raw_signals)
        Xs = self._apply_scaler(Xf)
        probs = self.model.predict_proba(Xs)
        return self._restore_matrix_order(probs, kept_idx, len(raw_signals))


class IdentityScaler:
    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X


def train_on_features(
    X_train_raw,
    y_train,
    signals_padded=None,
    X_valid_raw=None,
    y_valid=None,
    n_estimators=250,
    max_depth=None,
    random_state=385,
    n_jobs=-1,
    feature_kwargs=None,
    output_path=None,
    save_name="rf_features_model.joblib",
    model_type="LGBM",
    padding_mode=None,
    target_length=0,
    return_feature_data=False,
):
    train_signals = list(X_train_raw)
    if n_jobs is None or n_jobs == 0:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    feature_options = _prepare_feature_options(
        feature_kwargs, n_jobs=n_jobs, n_signals=len(train_signals)
    )

    logger.info(
        "[features] Training on %d samples, classes: %s",
        len(train_signals),
        np.unique(y_train),
    )

    signals_padded, kept_idx, _ = pad_trim_or_resample(
        train_signals,
        mode=padding_mode,
        target_length=target_length,
    )

    features_df, _ = extract_interpretable_features(
        train_signals,
        signals_padded=signals_padded,
        **feature_options,
    )

    feature_names = list(features_df.columns)
    logger.info(
        "[features] Extracted %d features for %d samples",
        len(feature_names),
        features_df.shape[0],
    )

    if features_df.shape[1] == 0:
        raise ValueError(
            "No numeric features extracted — check feature extraction settings"
        )

    Xf = _to_float_feature_matrix(features_df)
    Xf, col_medians = _impute_nan_columns(Xf)

    model_type_key = str(model_type)
    use_identity_scaler = model_type_key in {"LGBM", "RandomForest"}
    if use_identity_scaler:
        scaler = IdentityScaler()
        Xs = np.asarray(Xf, dtype=np.float32)
    else:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(Xf)

    model = _build_feature_estimator(
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(Xs, y_train)

    wrapper = FeatureModelWrapper(
        model,
        scaler,
        feature_options,
        feature_names,
        padding_mode=padding_mode,
        target_length=target_length,
        impute_values=col_medians,
    )

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        try:
            _export_feature_importance(model, feature_names, output_path)
        except Exception as e:
            logger.warning("Could not export feature importances: %s", e)

    if return_feature_data:
        return wrapper, {
            "train_features_df": features_df,
            "feature_names": feature_names,
            "feature_options": feature_options,
        }

    return wrapper


def _filter_correlated_features(features_df, corr_threshold=0.95, priority=None):
    if features_df.shape[1] <= 1:
        return list(features_df.columns), pd.DataFrame(
            columns=["dropped", "kept", "corr_abs"]
        )

    corr = features_df.corr().abs().fillna(0.0)
    order = list(features_df.columns)
    if priority is not None:
        order = list(
            priority.reindex(features_df.columns)
            .fillna(0.0)
            .sort_values(ascending=False)
            .index
        )

    keep = []
    dropped_rows = []
    for col in order:
        if not keep:
            keep.append(col)
            continue
        rel = corr.loc[col, keep]
        max_corr = float(rel.max()) if rel.size else 0.0
        if max_corr > corr_threshold:
            partner = rel.idxmax()
            dropped_rows.append({"dropped": col, "kept": partner, "corr_abs": max_corr})
            continue
        keep.append(col)

    dropped_df = pd.DataFrame(dropped_rows)
    return keep, dropped_df


def _fit_and_eval_feature_subset(
    model_type,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    n_jobs,
    random_state,
    n_estimators=300,
    max_depth=None,
):
    X_train_use = np.asarray(X_train, dtype=float)
    X_val_use = np.asarray(X_val, dtype=float)

    tree_like = str(model_type).upper() in {"LGBM", "RANDOMFOREST"}
    if not tree_like:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_use = scaler.fit_transform(X_train_use)
        X_val_use = scaler.transform(X_val_use)

    est = _build_feature_estimator(
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    est.fit(X_train_use, y_train)
    pred = est.predict(X_val_use)

    return {
        "Accuracy": float(accuracy_score(y_val, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_val, pred)),
        "Weighted_F1": float(
            f1_score(y_val, pred, average="weighted", zero_division=0)
        ),
        "estimator": est,
    }


def run_feature_reduction_protocol(
    X_train_raw,
    y_train,
    X_val_raw,
    y_val,
    *,
    output_dir,
    feature_kwargs=None,
    model_type="LGBM",
    n_jobs=-1,
    random_state=385,
    signals_padded_train=None,
    signals_padded_val=None,
    corr_threshold=0.95,
    var_threshold=1e-10,
    top_ks=(50, 25, 20, 10),
    precomputed_train_df=None,
):
    """Run feature reduction protocol and save scenario-level artifacts.

    Steps:
      1) Variance filter
      2) Correlation filter
      3) Ranking via Mutual Information + LGBM gain importance
      4) Refit/evaluate candidate subsets (full, top-k)
    """
    os.makedirs(output_dir, exist_ok=True)

    val_signals = list(X_val_raw)
    if n_jobs is None or n_jobs == 0:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    n_train_signals = (
        len(precomputed_train_df)
        if precomputed_train_df is not None
        else len(X_train_raw)
    )
    feature_options = _prepare_feature_options(
        feature_kwargs, n_jobs=n_jobs, n_signals=n_train_signals
    )

    if precomputed_train_df is None:
        train_signals = list(X_train_raw)
        train_df, _ = extract_interpretable_features(
            train_signals, signals_padded_train, **feature_options
        )
    else:
        train_df = precomputed_train_df.copy(deep=False)

    val_df, _ = extract_interpretable_features(
        val_signals, signals_padded_val, **feature_options
    )
    val_df = val_df.reindex(columns=train_df.columns, fill_value=np.nan)

    if train_df.shape[1] == 0:
        raise ValueError("Feature reduction protocol: no features extracted.")

    X_train_all = _to_float_feature_matrix(train_df)
    X_train_all, fill_values = _impute_nan_columns(X_train_all)

    X_val_all = _to_float_feature_matrix(val_df)
    X_val_all, _ = _impute_nan_columns(X_val_all, fill_values=fill_values)

    feature_names = list(train_df.columns)
    train_df_num = pd.DataFrame(X_train_all, columns=feature_names)
    val_df_num = pd.DataFrame(X_val_all, columns=feature_names)

    # Variance filter
    var_series = train_df_num.var(axis=0)
    keep_var = var_series[var_series > var_threshold].index.tolist()
    if not keep_var:
        keep_var = feature_names
    dropped_var = sorted(set(feature_names) - set(keep_var))

    variance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "variance": var_series.reindex(feature_names).to_numpy(),
            "kept_after_variance": [f in set(keep_var) for f in feature_names],
        }
    ).sort_values(["kept_after_variance", "variance"], ascending=[False, False])
    variance_df.to_csv(
        os.path.join(output_dir, "variance_analysis.tsv"), sep="\t", index=False
    )

    train_var = train_df_num[keep_var]
    val_var = val_df_num[keep_var]

    # Correlation filter
    keep_corr, dropped_corr_df = _filter_correlated_features(
        train_var,
        corr_threshold=corr_threshold,
        priority=var_series.reindex(keep_var),
    )
    if not keep_corr:
        keep_corr = keep_var

    if dropped_corr_df.empty:
        dropped_corr_df = pd.DataFrame(columns=["dropped", "kept", "corr_abs"])
    dropped_corr_df.to_csv(
        os.path.join(output_dir, "correlation_dropped_pairs.tsv"), sep="\t", index=False
    )

    # Ranking via MI + LGBM gain
    y_enc, _ = pd.factorize(np.asarray(y_train))
    X_rank = train_var[keep_corr].to_numpy(dtype=float)

    mi = mutual_info_classif(X_rank, y_enc, random_state=random_state)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)

    rank_est = _build_feature_estimator(
        model_type="LGBM",
        n_estimators=350,
        max_depth=None,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    rank_est.fit(X_rank, y_train)
    gain = np.asarray(
        getattr(rank_est, "feature_importances_", np.zeros(len(keep_corr))), dtype=float
    )

    mi_norm = mi / (mi.max() + 1e-12)
    gain_norm = gain / (gain.max() + 1e-12) if gain.size else np.zeros_like(mi_norm)
    combined = 0.6 * mi_norm + 0.4 * gain_norm

    score_df = pd.DataFrame(
        {
            "feature": keep_corr,
            "mi": mi,
            "gain_importance": gain,
            "mi_norm": mi_norm,
            "gain_norm": gain_norm,
            "combined_score": combined,
        }
    ).sort_values("combined_score", ascending=False)
    score_df.to_csv(
        os.path.join(output_dir, "feature_scores.tsv"), sep="\t", index=False
    )

    ranked_features = score_df["feature"].tolist()

    # Evaluate subsets
    perf_rows = []
    subset_defs = [("all_features", feature_names), ("variance+correlation", keep_corr)]
    for k in top_ks:
        kk = min(int(k), len(ranked_features))
        subset_defs.append((f"top{kk}", ranked_features[:kk]))

    seen = set()
    for subset_name, feats in subset_defs:
        if subset_name in seen or not feats:
            continue
        seen.add(subset_name)

        Xtr = train_df_num[feats].to_numpy(dtype=float)
        Xva = val_df_num[feats].to_numpy(dtype=float)

        result = _fit_and_eval_feature_subset(
            model_type=model_type,
            X_train=Xtr,
            y_train=y_train,
            X_val=Xva,
            y_val=y_val,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        perf_rows.append(
            {
                "subset": subset_name,
                "n_features": len(feats),
                "Accuracy": result["Accuracy"],
                "Balanced_Accuracy": result["Balanced_Accuracy"],
                "Weighted_F1": result["Weighted_F1"],
            }
        )

        pd.Series(feats, name="feature").to_csv(
            os.path.join(output_dir, f"{subset_name}_features.tsv"),
            sep="\t",
            index=False,
            header=True,
        )

    perf_df = pd.DataFrame(perf_rows).sort_values("Balanced_Accuracy", ascending=False)
    perf_df.to_csv(
        os.path.join(output_dir, "subset_performance.tsv"), sep="\t", index=False
    )

    summary = pd.DataFrame(
        [
            {"metric": "total_features_extracted", "value": len(feature_names)},
            {"metric": "dropped_low_variance", "value": len(dropped_var)},
            {"metric": "remaining_after_variance", "value": len(keep_var)},
            {"metric": "dropped_high_correlation", "value": len(dropped_corr_df)},
            {"metric": "remaining_after_correlation", "value": len(keep_corr)},
            {"metric": "corr_threshold", "value": corr_threshold},
            {"metric": "var_threshold", "value": var_threshold},
        ]
    )
    summary.to_csv(
        os.path.join(output_dir, "feature_reduction_summary.tsv"), sep="\t", index=False
    )

    logger.info(
        "[feature-test] saved protocol outputs to %s (raw=%d, post-var=%d, post-corr=%d)",
        output_dir,
        len(feature_names),
        len(keep_var),
        len(keep_corr),
    )

    return {
        "n_raw": len(feature_names),
        "n_after_variance": len(keep_var),
        "n_after_correlation": len(keep_corr),
        "ranked_features": ranked_features,
        "performance": perf_df,
    }


def save_time_series_samples(X, y, predictions, output_dir, max_plots=5):
    classes = np.unique(y)
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        correct_indices = cls_indices[predictions[cls_indices] == cls]
        incorrect_indices = cls_indices[predictions[cls_indices] != cls]

        # Save a few samples for each class
        for category, indices in [
            ("correct", correct_indices),
            ("incorrect", incorrect_indices),
        ]:
            sampled_indices = np.random.choice(
                indices, size=min(max_plots, len(indices)), replace=False
            )
            for i, idx in enumerate(sampled_indices):
                sample = X[idx].squeeze()
                plt.figure(figsize=(10, 4))
                plt.plot(sample)
                plt.title(f"Class {cls} - {category.capitalize()} Sample {i + 1}")
                plt.xlabel("Time Steps")
                plt.ylabel("Value")
                plt.savefig(f"{output_dir}/class_{cls}_{category}_sample_{i + 1}.png")
                plt.close()


class LossAccLogger(Callback):
    """Logs train/val loss and accuracy per epoch; safe if val_* not present."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def fmt(k):
            v = logs.get(k)
            return f"{v:.4f}" if (v is not None) else "n/a"

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={fmt('loss')}, val_loss={fmt('val_loss')}, "
            f"train_acc={fmt('accuracy')}, val_acc={fmt('val_accuracy')}"
        )


def weighted_categorical_crossentropy(class_weights):
    """
    class_weights: 1D array-like of shape (num_classes,)
    Expects y_true one-hot.
    Returns a loss function compatible with Keras.
    """
    class_weights = tf.constant(np.asarray(class_weights, dtype=np.float32))

    def loss(y_true, y_pred):
        # y_true: one-hot
        y_true = tf.cast(y_true, tf.float32)
        # compute unweighted categorical crossentropy per sample
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        # get per-sample weight from one-hot labels
        sample_weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        return ce * sample_weights

    return loss


def _build_inception_callbacks(
    early_stopping=True,
    early_stopping_patience=14,
    early_stopping_min_delta=0.0,
    early_stopping_monitor="loss",
):
    monitor = str(early_stopping_monitor or "val_loss")
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=6,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=max(int(early_stopping_patience), 1),
                min_delta=float(early_stopping_min_delta),
                restore_best_weights=True,
                verbose=1,
            )
        )
    return callbacks


def _set_reproducible_seed(seed, enable_tf_determinism=True):
    if seed is None:
        return
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    try:
        tf.keras.utils.set_random_seed(s)
    except Exception:
        pass

    if enable_tf_determinism:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        # Optional strict mode to reduce run-to-run CPU scheduling variance.
        single_thread = os.environ.get("NP_TF_SINGLE_THREAD", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if single_thread:
            try:
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except Exception:
                pass
            try:
                tf.config.threading.set_intra_op_parallelism_threads(1)
            except Exception:
                pass


def _add_random_state_if_supported(constructor, kwargs, random_state):
    out = dict(kwargs)
    if random_state is None:
        return out
    try:
        params = inspect.signature(constructor).parameters
        if "random_state" in params and "random_state" not in out:
            out["random_state"] = int(random_state)
    except Exception:
        # If signature introspection fails, keep kwargs unchanged.
        pass
    return out


def _add_supported_kwargs(constructor, kwargs, candidate_kwargs):
    out = dict(kwargs)
    if not candidate_kwargs:
        return out
    try:
        params = inspect.signature(constructor).parameters
    except Exception:
        return out

    for key, value in candidate_kwargs.items():
        if key in params and key not in out:
            out[key] = value
    return out


def _inception_config(
    model_name,
    n_samples=None,
    batch_size_override=None,
    large_batch_size_override=None,
):
    n_samples = int(n_samples or 0)
    if model_name == "InceptionTime-large":
        default_batch = 256
        if large_batch_size_override is not None and int(large_batch_size_override) > 0:
            default_batch = int(large_batch_size_override)
        return {
            "batch_size": int(default_batch),
            "n_classifiers": 5,
            "n_conv_per_layer": 3,
            "n_filters": 64,
            "kernel_size": 30,
            "depth": 4,
            "learning_rate": 1.0e-3,
        }

    default_batch = 128
    if batch_size_override is not None and int(batch_size_override) > 0:
        default_batch = int(batch_size_override)
    return {
        "batch_size": int(default_batch),
        "n_classifiers": 5,
        "n_conv_per_layer": 3,
        "n_filters": 64,
        "kernel_size": 32,
        "depth": 4,
        "learning_rate": 1.5e-3,
    }


def _minirocket_config(n_samples=None):
    return {"n_kernels": 2000}


def _timesnet_config(n_samples=None, batch_size_override=None):
    n_samples = int(n_samples or 0)
    default_batch = 128
    if batch_size_override is not None and int(batch_size_override) > 0:
        default_batch = int(batch_size_override)
    return {
        "batch_size": int(default_batch),
        "d_model": 64,
        "d_ff": 96,
        "e_layers": 2,
        "top_k": 3,
        "num_kernels": 4,
        "dropout": 0.10,
        "learning_rate": 1.0e-3,
    }


def _build_inception_optimizer(n_classifiers, learning_rate):
    lr = float(learning_rate)
    if int(n_classifiers or 1) > 1:
        legacy_optimizers = getattr(tf.keras.optimizers, "legacy", None)
        if legacy_optimizers is not None and hasattr(legacy_optimizers, "Adam"):
            return legacy_optimizers.Adam(learning_rate=lr, clipnorm=1.0)

        logger.warning(
            "Legacy Adam not available for InceptionTime ensemble; falling back to 'adam'."
        )
        return "adam"

    return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)


def get_model(
    model_name,
    n_epochs=25,
    n_jobs=4,
    loss_fn=None,
    n_samples=None,
    inception_batch_size=None,
    inception_large_batch_size=None,
    timesnet_batch_size=None,
    inception_early_stopping=True,
    inception_early_stopping_patience=14,
    inception_early_stopping_min_delta=0.0,
    random_state=385,
):
    model_name = str(model_name)
    _set_reproducible_seed(random_state)

    if model_name in ("InceptionTime", "InceptionTime-large"):
        InceptionTimeClassifier, _ = _import_aeon_deep_learning_silenced()
        cfg = _inception_config(
            model_name,
            n_samples=n_samples,
            batch_size_override=inception_batch_size,
            large_batch_size_override=inception_large_batch_size,
        )
        force_native = True

        native_kwargs = {
            "n_epochs": n_epochs,
            "batch_size": cfg["batch_size"],
            "n_classifiers": cfg["n_classifiers"],
            "n_conv_per_layer": cfg["n_conv_per_layer"],
            "n_filters": cfg["n_filters"],
            "kernel_size": cfg["kernel_size"],
            "depth": cfg["depth"],
            "verbose": 1,
            "optimizer": _build_inception_optimizer(
                n_classifiers=cfg["n_classifiers"],
                learning_rate=cfg["learning_rate"],
            ),
            "callbacks": _build_inception_callbacks(
                early_stopping=inception_early_stopping,
                early_stopping_patience=inception_early_stopping_patience,
                early_stopping_min_delta=inception_early_stopping_min_delta,
                early_stopping_monitor="val_loss",
            ),
            "random_state": random_state,
        }
        if loss_fn is not None:
            native_kwargs["loss"] = loss_fn

        if force_native:
            logger.info(
                "Using native TensorFlow InceptionTime fallback (forced by env)."
            )
            return NativeInceptionTimeClassifier(**native_kwargs)

        logger.info(
            "Building %s (epochs=%d, batch=%d, filters=%d, depth=%d, kernel=%d)",
            model_name,
            n_epochs,
            cfg["batch_size"],
            cfg["n_filters"],
            cfg["depth"],
            cfg["kernel_size"],
        )
        inception_kwargs = {
            "n_epochs": n_epochs,
            "batch_size": cfg["batch_size"],
            "n_classifiers": cfg["n_classifiers"],
            "n_conv_per_layer": cfg["n_conv_per_layer"],
            "n_filters": cfg["n_filters"],
            "kernel_size": cfg["kernel_size"],
            "depth": cfg["depth"],
            "verbose": 1,
            "optimizer": _build_inception_optimizer(
                n_classifiers=cfg["n_classifiers"],
                learning_rate=cfg["learning_rate"],
            ),
            "callbacks": _build_inception_callbacks(
                early_stopping=inception_early_stopping,
                early_stopping_patience=inception_early_stopping_patience,
                early_stopping_min_delta=inception_early_stopping_min_delta,
                early_stopping_monitor="val_loss",
            ),
        }
        if loss_fn is not None:
            inception_kwargs["loss"] = loss_fn
        inception_kwargs = _add_random_state_if_supported(
            InceptionTimeClassifier,
            inception_kwargs,
            random_state,
        )
        inception_kwargs = _add_supported_kwargs(
            InceptionTimeClassifier,
            inception_kwargs,
            {
                "save_best_model": False,
                "save_last_model": False,
                "save_init_model": False,
            },
        )
        model = InceptionTimeClassifier(**inception_kwargs)
        try:
            fit_sig = inspect.signature(model.fit)
            aeon_supports_val = (
                "X_val" in fit_sig.parameters and "y_val" in fit_sig.parameters
            )
        except Exception:
            aeon_supports_val = False

        if not aeon_supports_val:
            logger.warning(
                "aeon %s does not support X_val/y_val in fit; using native TensorFlow InceptionTime for val_loss early stopping.",
                model_name,
            )
            return NativeInceptionTimeClassifier(**native_kwargs)

        model._native_fallback_builder = lambda: NativeInceptionTimeClassifier(
            **native_kwargs
        )
        if not getattr(model, "callbacks", None):
            try:
                model.callbacks = list(inception_kwargs["callbacks"])
            except Exception:
                pass
        return model

    if model_name == "TimesNet":
        cfg = _timesnet_config(
            n_samples=n_samples,
            batch_size_override=timesnet_batch_size,
        )
        logger.info(
            "Building TimesNet (epochs=%d, batch=%d, d_model=%d, layers=%d, top_k=%d)",
            n_epochs,
            cfg["batch_size"],
            cfg["d_model"],
            cfg["e_layers"],
            cfg["top_k"],
        )
        return TimesNetClassifier(
            n_epochs=n_epochs,
            batch_size=cfg["batch_size"],
            d_model=cfg["d_model"],
            d_ff=cfg["d_ff"],
            e_layers=cfg["e_layers"],
            top_k=cfg["top_k"],
            num_kernels=cfg["num_kernels"],
            dropout=cfg["dropout"],
            learning_rate=cfg["learning_rate"],
            verbose=1,
            loss=loss_fn,
            callbacks=_build_inception_callbacks(
                early_stopping=inception_early_stopping,
                early_stopping_patience=inception_early_stopping_patience,
                early_stopping_min_delta=inception_early_stopping_min_delta,
                early_stopping_monitor="val_loss",
            ),
            random_state=random_state,
        )

    mini_cfg = _minirocket_config(n_samples=n_samples)
    if model_name == "MiniRocket":
        logger.info(
            "Building MiniRocket (n_kernels=%d, estimator=default)",
            mini_cfg["n_kernels"],
        )

    def _build_resnet():
        _, ResNetClassifier = _import_aeon_deep_learning_silenced()
        kwargs = {
            "n_epochs": n_epochs,
            "verbose": True,
            "batch_size": 128,
            "n_conv_per_residual_block": 3,
            "n_residual_blocks": 2,
            "n_filters": [32, 16],
        }
        kwargs = _add_random_state_if_supported(ResNetClassifier, kwargs, random_state)
        kwargs = _add_supported_kwargs(
            ResNetClassifier,
            kwargs,
            {
                "save_best_model": False,
                "save_last_model": False,
                "save_init_model": False,
            },
        )
        return ResNetClassifier(**kwargs)

    model_builders = {
        "ResNet": _build_resnet,
        "MiniRocket": lambda: MiniRocketClassifier(
            **_add_random_state_if_supported(
                MiniRocketClassifier,
                {"n_kernels": 1000, "n_jobs": n_jobs},
                random_state,
            )
        ),
        "MultiRocket-huge": lambda: MultiRocketClassifier(
            **_add_random_state_if_supported(
                MultiRocketClassifier,
                {
                    "n_kernels": 50000,
                    "n_jobs": n_jobs,
                    "class_weight": "balanced",
                },
                random_state,
            )
        ),
        "MultiRocket-large": lambda: MultiRocketClassifier(
            **_add_random_state_if_supported(
                MultiRocketClassifier,
                {
                    "n_kernels": 5000,
                    "n_jobs": n_jobs,
                    "estimator": make_pipeline(
                        StandardScaler(with_mean=False),
                        SGDClassifier(
                            loss="modified_huber",
                            penalty="l2",
                            alpha=1e-4,
                            max_iter=1000,
                            tol=None,
                            learning_rate="optimal",
                            early_stopping=True,
                            validation_fraction=0.05,
                            n_iter_no_change=10,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                },
                random_state,
            )
        ),
        "MultiRocket": lambda: MultiRocketClassifier(
            **_add_random_state_if_supported(
                MultiRocketClassifier,
                {
                    "n_kernels": 1000,
                    "n_jobs": n_jobs,
                    "class_weight": "balanced",
                },
                random_state,
            )
        ),
        "MultiRocketHydra": lambda: MultiRocketHydraClassifier(
            **_add_random_state_if_supported(
                MultiRocketHydraClassifier,
                {"class_weight": "balanced", "n_jobs": n_jobs},
                random_state,
            )
        ),
        "HIVCOTEV2": lambda: HIVECOTEV2(
            **_add_random_state_if_supported(HIVECOTEV2, {}, random_state)
        ),
        "QUANT": lambda: QUANTClassifier(
            **_add_random_state_if_supported(
                QUANTClassifier,
                {"interval_depth": 6, "quantile_divisor": 4},
                random_state,
            )
        ),
        "MUSE": lambda: MUSE(
            **_add_random_state_if_supported(
                MUSE,
                {
                    "window_inc": 4,
                    "use_first_order_differences": False,
                    "n_jobs": n_jobs,
                },
                random_state,
            )
        ),
        "WEASEL": lambda: WEASEL(
            **_add_random_state_if_supported(
                WEASEL,
                {"window_inc": 4, "n_jobs": n_jobs, "support_probabilities": True},
                random_state,
            )
        ),
    }
    if model_name not in model_builders:
        raise ValueError(f"Unknown model '{model_name}'")
    return model_builders[model_name]()


def is_keras_model(model):
    return hasattr(model, "training_model_") or hasattr(model, "model_")


def fit_model_safe(model, X_train, y_train, X_val=None, y_val=None, **fit_kwargs):
    """
    Fits an aeon model safely.
    """
    import inspect

    sig = inspect.signature(model.fit)
    supports_val = "X_val" in sig.parameters and "y_val" in sig.parameters

    if is_keras_model(model):
        # Merge existing and fit-time callbacks, then attach LossAccLogger.
        cb_list = list(getattr(model, "callbacks", []) or [])
        user_callbacks = fit_kwargs.pop("callbacks", None)
        if user_callbacks:
            cb_list.extend(list(user_callbacks))
        if not any(isinstance(cb, LossAccLogger) for cb in cb_list):
            cb_list.append(LossAccLogger())
        fit_kwargs["callbacks"] = cb_list

        try:
            if supports_val and X_val is not None and y_val is not None:
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_kwargs)
            else:
                model.fit(X_train, y_train, **fit_kwargs)
            return model
        except Exception as exc:
            fallback_builder = getattr(model, "_native_fallback_builder", None)
            if fallback_builder is None:
                raise

            emsg = str(exc).lower()
            can_fallback = (X_val is not None and y_val is not None) and (
                "val_loss" in emsg
                or "x_val" in emsg
                or "y_val" in emsg
                or "callback" in emsg
            )
            if not can_fallback:
                raise

            logger.warning(
                "Validation-based aeon fit failed (%s). Falling back to native TensorFlow InceptionTime.",
                exc,
            )
            native_model = fallback_builder()
            fit_model_safe(
                native_model, X_train, y_train, X_val=X_val, y_val=y_val, **fit_kwargs
            )

            model.__class__ = native_model.__class__
            model.__dict__ = native_model.__dict__
            return model
    else:
        model.fit(X_train, y_train)
        return model


UNKNOWN_PATTERNS = (
    r"^unknown[\w-]*$",
    r"^ßCATnoC$",
    r"^YLDSnoC$",
    r"^InfluenzanoC$",
    r"^Influenza$",
)

CRUDE_PATTERNS = (
    r"^CHGnegG$",
    r"^βCATins1$",
    r"^βCATins2$",
)


def _safe_metric(func, y_true, y_pred, **kwargs):
    if y_true.size == 0:
        return np.nan
    try:
        return func(y_true, y_pred, **kwargs)
    except Exception:
        return np.nan


def _build_label_mask(y_labels, patterns):
    return np.array(
        [any(re.match(pat, str(lbl)) for pat in patterns) for lbl in y_labels],
        dtype=bool,
    )


def _resolve_labels_for_cm_all(y_all, preds_all, all_class_labels=None):
    if all_class_labels is None:
        return np.unique(np.concatenate([y_all, preds_all]))

    labels = list(all_class_labels)
    for lbl in np.unique(y_all):
        if lbl not in labels:
            labels.append(lbl)
            logger.info("Adding new class '%s' to confusion matrix labels", lbl)
    return np.asarray(labels)


def _write_metrics_summary_tsv(out_dir, dataset_name, metrics):
    important_keys = [
        "Accuracy (all)",
        "Accuracy (w/o unknown)",
        "Accuracy (w/o unknown & w/o crude)",
        "Balanced Accuracy (w/o unknown)",
        "MCC (all)",
        "MCC (w/o unknown)",
        "Micro Precision",
        "Micro Recall",
        "Micro F1-Score",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-Score",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1-Score",
        "Top-1 (all)",
        "Top-2 (all)",
        "Top-3 (all)",
    ]
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"{dataset_name}_metrics.tsv")
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for key in important_keys:
            if key in metrics:
                writer.writerow([key, metrics[key]])


def _write_per_class_metrics_tsv(out_dir, dataset_name, report_dict):
    if not report_dict:
        return
    per_class_file = os.path.join(out_dir, f"{dataset_name}_metrics_per_class.tsv")
    class_labels = [
        label
        for label in report_dict.keys()
        if label not in ("accuracy", "macro avg", "weighted avg", "micro avg")
    ]

    with open(per_class_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for label in sorted(class_labels):
            row = report_dict[label]
            writer.writerow(
                [
                    label,
                    f"{row['precision']:.6f}",
                    f"{row['recall']:.6f}",
                    f"{row['f1-score']:.6f}",
                    int(row["support"]),
                ]
            )
        for avg_type in ("macro avg", "weighted avg"):
            if avg_type in report_dict:
                row = report_dict[avg_type]
                writer.writerow(
                    [
                        avg_type,
                        f"{row['precision']:.6f}",
                        f"{row['recall']:.6f}",
                        f"{row['f1-score']:.6f}",
                        int(row["support"]),
                    ]
                )


def _log_evaluation_snapshot(dataset_name, metrics):
    logger.info("\n%s Set Evaluation:", dataset_name)
    for key, fmt in [
        ("Accuracy (all)", ".3f"),
        ("Balanced Accuracy (w/o unknown)", ".3f"),
        ("Weighted F1-Score", ".3f"),
        ("MCC (w/o unknown)", ".3f"),
        ("Top-2 (w/o unknown)", ".3f"),
        ("Top-3 (w/o unknown)", ".3f"),
    ]:
        val = metrics.get(key, np.nan)
        if not np.isnan(val):
            logger.info("%s: %s", key, format(val, fmt))


def evaluate_model(
    model,
    X,
    y,
    dataset_name="Validation",
    out_dir=None,
    all_class_labels=None,
    predict_fn=None,
):
    y = np.asarray(y)
    preds, probs = _get_predictions(model, X, predict_fn)
    unknown_mask = _build_label_mask(y, UNKNOWN_PATTERNS)
    crude_mask = _build_label_mask(y, CRUDE_PATTERNS)
    known_mask = ~unknown_mask
    known_no_crude_mask = known_mask & (~crude_mask)

    subsets = {
        "all": (y, preds, probs),
        "known": (
            y[known_mask],
            preds[known_mask],
            probs[known_mask] if probs is not None else None,
        ),
        "known_no_crude": (
            y[known_no_crude_mask],
            preds[known_no_crude_mask],
            probs[known_no_crude_mask] if probs is not None else None,
        ),
        "unknown": (y[unknown_mask], preds[unknown_mask], None),
    }

    y_all, preds_all, _ = subsets["all"]
    labels_for_cm_all = _resolve_labels_for_cm_all(
        y_all, preds_all, all_class_labels=all_class_labels
    )

    present_known = (
        np.unique(np.concatenate(subsets["known"][:2]))
        if subsets["known"][0].size
        else np.array([])
    )
    present_unknown = (
        np.unique(np.concatenate(subsets["unknown"][:2]))
        if subsets["unknown"][0].size
        else np.array([])
    )

    metrics = {}

    for key, (y_true, y_pred, _) in subsets.items():
        if key == "unknown":
            continue
        suffix = (
            ""
            if key == "all"
            else f" (w/o unknown)" if key == "known" else " (w/o unknown & w/o crude)"
        )
        metrics[f"Accuracy{suffix}"] = _safe_metric(accuracy_score, y_true, y_pred)

    metrics["Balanced Accuracy (w/o unknown)"] = _safe_metric(
        balanced_accuracy_score, *subsets["known"][:2]
    )
    metrics["MCC (all)"] = _safe_metric(matthews_corrcoef, *subsets["all"][:2])
    metrics["MCC (w/o unknown)"] = _safe_metric(
        matthews_corrcoef, *subsets["known"][:2]
    )

    y_known, preds_known, _ = subsets["known"]
    # Reporting metrics should be computed only over evaluated ground-truth classes.
    labels_for_report = list(np.unique(y_known)) if y_known.size else None
    if labels_for_report:
        for avg in ["micro", "macro", "weighted"]:
            kwargs = {"average": avg, "zero_division": 0, "labels": labels_for_report}
            metrics[f"{avg.capitalize()} Precision"] = _safe_metric(
                precision_score, y_known, preds_known, **kwargs
            )
            metrics[f"{avg.capitalize()} Recall"] = _safe_metric(
                recall_score, y_known, preds_known, **kwargs
            )
            metrics[f"{avg.capitalize()} F1-Score"] = _safe_metric(
                f1_score, y_known, preds_known, **kwargs
            )
    else:
        for name in ["Precision", "Recall", "F1-Score"]:
            metrics[f"{name} (w/o unknown)"] = np.nan

    metrics["Confusion Matrix (all)"] = confusion_matrix(
        y_all, preds_all, labels=labels_for_cm_all
    )
    metrics["Labels (confusion_matrix_all)"] = labels_for_cm_all

    for key, (y_true, y_pred, _) in [
        ("known", subsets["known"]),
        ("unknown", subsets["unknown"]),
    ]:
        present = present_known if key == "known" else present_unknown
        if present.size:
            cm = confusion_matrix(y_true, y_pred, labels=present)
            metrics[
                (
                    f"Confusion Matrix (only {key})"
                    if key == "unknown"
                    else f"Confusion Matrix (w/o unknown)"
                )
            ] = cm
            metrics[f"Labels (confusion_matrix_{key})"] = present
        else:
            metrics[
                (
                    f"Confusion Matrix (only {key})"
                    if key == "unknown"
                    else f"Confusion Matrix (w/o unknown)"
                )
            ] = np.array([[]])
            metrics[f"Labels (confusion_matrix_{key})"] = np.array([])

    if labels_for_report:
        report = classification_report(
            y_known,
            preds_known,
            labels=labels_for_report,
            output_dict=True,
            zero_division=0,
        )
        metrics["Classification Report (w/o unknown)"] = report
        metrics["Classification Report"] = report
    else:
        metrics["Classification Report (w/o unknown)"] = {}

    classes_for_probs = _infer_class_ordering(
        model, probs, all_class_labels, labels_for_cm_all, y_all, preds_all
    )

    topk_metrics = {}
    for key in ["all", "known", "known_no_crude"]:
        y_true, _, probs_subset = subsets[key]
        topk = _compute_topk_metrics(
            probs_subset, y_true, classes_for_probs, ks=(1, 2, 3)
        )
        suffix = (
            ""
            if key == "all"
            else " (w/o unknown)" if key == "known" else " (w/o unknown & w/o crude)"
        )
        for k in [1, 2, 3]:
            topk_metrics[f"Top-{k}{suffix}"] = topk.get(f"Top-{k}", np.nan)

    metrics.update(topk_metrics)

    metrics.update(
        {
            "y_true": y_all,
            "y_pred": preds_all,
            "y_known": subsets["known"][0],
            "y_unknown": subsets["unknown"][0],
            "mask_unknown": unknown_mask,
            "mask_crude": crude_mask,
            "Predictions": preds_all,
            "Probabilities": probs,
        }
    )

    if out_dir:
        _write_metrics_summary_tsv(out_dir, dataset_name, metrics)
        if present_known.size:
            _write_per_class_metrics_tsv(
                out_dir, dataset_name, metrics["Classification Report (w/o unknown)"]
            )

    _log_evaluation_snapshot(dataset_name, metrics)

    return metrics


def _sanitize_probabilities(probs):
    """Return a finite probability-like matrix safe for metric/plot computation."""
    if probs is None:
        return None

    arr = np.asarray(probs, dtype=float)
    if arr.size == 0:
        return arr

    non_finite = ~np.isfinite(arr)
    if np.any(non_finite):
        logger.warning(
            "Detected %d non-finite probability values; replacing and renormalizing.",
            int(non_finite.sum()),
        )
        arr = np.where(np.isfinite(arr), arr, 0.0)

    if arr.ndim == 1:
        return np.clip(arr, 0.0, 1.0)

    arr = np.clip(arr, 0.0, None)
    if arr.shape[1] == 0:
        return arr

    row_sum = arr.sum(axis=1, keepdims=True)
    bad_rows = (~np.isfinite(row_sum[:, 0])) | (row_sum[:, 0] <= 0.0)
    if np.any(bad_rows):
        arr[bad_rows, :] = 1.0 / float(arr.shape[1])
        row_sum = arr.sum(axis=1, keepdims=True)

    return arr / row_sum


def _get_predictions(model, X, predict_fn):
    if predict_fn is not None:
        try:
            out = predict_fn(model, X)
        except TypeError:
            out = predict_fn(X)
        if isinstance(out, (tuple, list)):
            preds = np.asarray(out[0])
            probs = np.asarray(out[1]) if len(out) > 1 and out[1] is not None else None
        else:
            preds = np.asarray(out)
            probs = None
        return preds, _sanitize_probabilities(probs)

    if model is None:
        raise ValueError("No model provided and no predict_fn; cannot predict")
    if not hasattr(model, "predict"):
        raise ValueError("Model has no .predict and no predict_fn supplied")

    preds = np.asarray(model.predict(X))
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = np.asarray(model.predict_proba(X))
        except Exception:
            pass
    return preds, _sanitize_probabilities(probs)


def _infer_class_ordering(
    model, probs, all_class_labels, labels_for_cm_all, y_all, preds_all
):
    if probs is None:
        return None

    if hasattr(model, "classes_"):
        classes = np.asarray(model.classes_)
        if not np.array_equal(classes, np.sort(classes)):
            pass
        else:
            pass
        return classes
    if all_class_labels is not None:
        return np.asarray(all_class_labels)
    if len(labels_for_cm_all) == probs.shape[1]:
        logger.warning(
            "Using labels_for_cm_all as class ordering for probability columns."
        )
        return np.asarray(labels_for_cm_all)

    uniq = np.unique(np.concatenate([y_all, preds_all]))
    if probs.shape[1] == len(uniq):
        logger.warning(
            "Inferred class ordering from unique labels - this may be incorrect."
        )
        return uniq

    logger.warning(
        "Cannot infer class ordering for probability columns; top-k metrics will be NaN."
    )
    return None


def _compute_topk_metrics(probs_array, y_true, classes, ks=(1, 2, 3)):
    out = {f"Top-{k}": np.nan for k in ks}

    if (
        probs_array is None
        or np.ndim(probs_array) != 2
        or classes is None
        or len(classes) != probs_array.shape[1]
    ):
        return out

    order = np.argsort(probs_array, axis=1)[:, ::-1]
    top_labels = np.asarray(classes)[order]
    n = len(y_true)

    for k in ks:
        topk = top_labels[:, :k]
        hits = np.array([y_true[i] in topk[i] for i in range(n)], dtype=float)
        out[f"Top-{k}"] = float(np.nanmean(hits))

    return out


def prepare_scenario_datasets(
    train_sub,
    val_sub,
    *,
    train_stats,
    smoothing,
    norm_method,
    fixed_length,
    padding_mode,
    cutoff,
    order,
    trimming,
    trim_fraction,
    outdir,
):
    common = dict(
        outdir=outdir,
        train_stats=train_stats,
        smoothing=smoothing,
        norm_method=norm_method,
        padding_method="fixed",
        fixed_length=fixed_length,
        cutoff=cutoff,
        order=order,
        trimming=trimming,
        trim_fraction=trim_fraction,
    )
    X_train, y_train, X_train_raw, y_train_runID, info_train = prepare_dataset_from_df(
        train_sub, **common, padding_mode=padding_mode
    )
    X_val, y_val, X_val_raw, y_val_runID, _ = prepare_dataset_from_df(
        val_sub, **common, padding_mode=padding_mode
    )
    X_train_IT, y_train_IT, X_train_raw_IT, y_train_runID_IT, _ = (
        prepare_dataset_from_df(train_sub, **common, padding_mode="truncate_or_pad")
    )
    X_val_IT, y_val_IT, X_val_raw_IT, y_val_runID_IT, _ = prepare_dataset_from_df(
        val_sub, **common, padding_mode="truncate_or_pad"
    )
    return dict(
        X_train=X_train,
        y_train=y_train,
        X_train_raw=X_train_raw,
        y_train_runID=y_train_runID,
        X_val=X_val,
        y_val=y_val,
        X_val_raw=X_val_raw,
        y_val_runID=y_val_runID,
        X_train_IT=X_train_IT,
        y_train_IT=y_train_IT,
        X_train_raw_IT=X_train_raw_IT,
        y_train_runID_IT=y_train_runID_IT,
        X_val_IT=X_val_IT,
        y_val_IT=y_val_IT,
        X_val_raw_IT=X_val_raw_IT,
        y_val_runID_IT=y_val_runID_IT,
        info_train=info_train,
    )


def safe_concat(arrays):
    valid_arrays = []
    for a in arrays:
        if a is None:
            continue
        a = np.asarray(a)
        # Promote scalars 0D to 1D
        if a.ndim == 0:
            a = a[np.newaxis]  # or np.atleast_1d(a)
        valid_arrays.append(a)

    if not valid_arrays:
        return None
    return np.concatenate(valid_arrays, axis=0)
