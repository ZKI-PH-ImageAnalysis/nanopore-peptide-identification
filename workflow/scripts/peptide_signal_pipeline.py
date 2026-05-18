#!/usr/bin/env python

import os
import random
import time
from collections import OrderedDict

# Keep noisy TensorFlow/XLA startup logs minimal
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np
import pandas as pd
import rich_click as click
import logging
from remora import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from utils_segmentation import (
    collect_records_multi_process,
    collect_records_single_process,
    export_filtering_summary,
    group_records_by_category,
    format_elapsed_time,
    get_read_ids,
    plot_grouped_records,
    resolve_category_outputs,
    summarize_and_export,
)
from utils_classification import (
    all_classes_postprocessing,
    is_inception_model,
    prepare_scenario_datasets,
    resolve_inception_preprocessing,
    run_dtw_plots,
    run_model,
    safe_concat,
    summarize_model_seed_runs,
)
from utils_data_preprocessing import (
    compute_train_stats,
    log_length_distributions,
    prepare_dataset_from_df,
    process_parse_inputfiles,
    resample_signals_to_length,
    trim_padding,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nanopore-peptide-classifier")


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main():
    """
    # Nanopore-Peptide-Classifier
    """


@main.command()
@click.argument(
    "pod5",
    required=True,
    type=click.Path(exists=True, dir_okay=True),
)
@click.argument(
    "outdir",
    required=True,
    type=click.Path(dir_okay=True, file_okay=False),
)
@click.option(
    "--reads",
    "-r",
    required=False,
    type=click.Path(dir_okay=False, file_okay=True),
    help="Path to read ID text file",
)
@click.option(
    "--max-plots",
    "-m",
    type=int,
    default=150,
    help="Maximum number of reads you want to plot",
)
@click.option(
    "--sam",
    "-s",
    required=False,
    type=click.Path(dir_okay=False, file_okay=True),
    help="Path to SAM file for special segmentation purposes",
)
@click.option(
    "--extend_by",
    default='0',
    type=int,
    help="Amount of signal points to extend the detected PLR region by (default: 0)",
)
@click.option(
    "--signal_type",
    default='norm',
    required=False,
    type=click.Choice(["norm", "pa"], case_sensitive=False),
    help="Signal type used only for segmentation plots ('norm' or 'pa'); peptide_signals.tsv is always exported in normalized values.",
)
@click.option(
    "--max_valid",
    type=int,
    default=10000000,
    help="Maximum number of reads you want to plot",
)
@click.option(
    "--min_alignment_position",
    default=None,
    type=int,
    help="Minimum alignment position for PLR detection",
)
@click.option(
    "--time-limit",
    default=None,
    type=float,
    help="Optional time limit (minutes) for worker processing. Workers stop after this time.",
)
@click.option(
    "--max-reads",
    default=None,
    type=int,
    help="Optional limit on total number of reads to process (trims input list).",
)
@click.option(
    "--chunksize",
    default=100,
    type=int,
    help="Number of reads per chunk for multiprocessing.",
)
@click.option(
    "--seed",
    default=1234,
    type=int,
    show_default=True,
    help="Random seed for sampling and worker reproducibility.",
)
def find_region(
    pod5,
    outdir,
    reads,
    max_plots,
    sam,
    extend_by,
    signal_type,
    max_valid,
    min_alignment_position,
    time_limit,
    max_reads,
    chunksize,
    seed,
):
    """
    Identifies and plots the peptide linker region
    """
    logger.info("Starting peptide linker region detection...")
    start_total = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Using seed: {seed}")
    logger.info(f"Maximum valid samples: {max_valid}")

    num_processes = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    os.makedirs(outdir, exist_ok=True)  

    read_ids = get_read_ids(pod5, reads)
    read_ids = list(OrderedDict.fromkeys(read_ids))

    if max_reads is not None:
        logger.info(f"Trimming read list to first {max_reads} entries (was {len(read_ids)})")
        read_ids = read_ids[:int(max_reads)]

    time_limit_seconds = None if time_limit is None else float(time_limit) * 60.0

    bam_fh = io.ReadIndexedBam(sam) if sam else None
    start_processing = time.perf_counter()    

    DEBUG_SINGLE_PROCESS = False
    if DEBUG_SINGLE_PROCESS:
        all_rec = collect_records_single_process(
            read_ids,
            pod5,
            bam_fh,
            extend_by,
            signal_type,
            min_alignment_position,
        )
    else:
        all_rec = collect_records_multi_process(
            read_ids,
            pod5,
            bam_fh,
            seed,
            extend_by,
            signal_type,
            max_valid,
            min_alignment_position,
            chunksize,
            num_processes,
            time_limit_seconds,
        )

    end_processing = time.perf_counter()
    groups = group_records_by_category(all_rec)
    subdirs, plot_limits = resolve_category_outputs(groups, outdir, max_plots)

    start_plotting = time.perf_counter()
    plot_grouped_records(groups, plot_limits, subdirs, pod5, bam_fh, signal_type, num_processes, max_plots)

    end_plotting = time.perf_counter()
    logger.info(f"Plotting completed in {end_plotting - start_plotting:.2f} seconds")
    
    # always write the TSV, even when empty
    valid_reads = groups.get("valid", [])
    summarize_and_export(valid_reads, outdir)
    
    export_filtering_summary(all_rec, groups, outdir, len(read_ids))

    end_total = time.perf_counter()
    logger.info("Finished peptide linker region detection.")
    logger.info(f"Chunk processing took     {format_elapsed_time(end_processing - start_processing)}")
    logger.info(f"Plotting took             {format_elapsed_time(end_plotting - start_plotting)}")
    logger.info(f"Total runtime             {format_elapsed_time(end_total - start_total)}")

    
@main.command()
@click.argument('metadata', type=click.Path(exists=True))
@click.option('--output', required=True, type=click.Path())
@click.option('--models', default='InceptionTime,TimesNet,MiniRocket,featureLGBM',
              help='Comma-separated model names to run')
@click.option('--dataset_stats', is_flag=True, default=False)
@click.option('--split_size', default=0.1, type=float)
@click.option('--smoothing', default="low-pass-bessel",
              type=click.Choice(['None', 'savitzky-golay', 'moving-average', 'low-pass-butter', 'low-pass-bessel']))
@click.option('--cutoff', default=0.2, type=float, help="Cutoff frequency for low-pass filters")
@click.option('--order', default=2, type=float, help="Order for low-pass filters")
@click.option('--testing', default=0, type=int)
@click.option('--seed', default=385, type=int)
@click.option('--plot_dtw', is_flag=True, default=False)
@click.option('--norm_method', default='None',
              type=click.Choice(['None', 'MAD-local', 'MAD-global', 'MAD-before-plr',
                                 'min-max-before-plr', 'median-before-plr', 'minmax-global',
                                 'quantile-global', 'minmax-decile-median']))
@click.option('--padding_method', default='percentile',
              type=click.Choice(['percentile', 'median', 'mean', 'fixed']))
@click.option('--fixed_length', default=300, type=int)
@click.option('--padding_mode', default='resample',
              type=click.Choice(['truncate_or_pad', 'pad_only', 'trim_only', 'strict',
                                 'resample', 'resample_if_longer']))
@click.option('--trimming', default=False, type=bool,
              help="Whether to trim signals before classification")
@click.option('--trim_fraction', default=0.03, type=float,
              help="Fraction to trim from signal edges before classification")
@click.option('--min_alignment_position', default=None, type=int,
              help="Minimum alignment position for classification. None to take all reads")
@click.option('--feature-test', is_flag=True, default=False,
        help="Run feature reduction protocol for featureLGBM and save subset results per scenario")
@click.option(
    '--feature-set',
    default='top50',
    type=click.Choice(['all', 'top50'], case_sensitive=False),
    show_default=True,
    help="Feature subset for feature-based models. 'top50' uses a curated fixed top-50 list.",
)
@click.option(
    '--model-seed-runs',
    default=1,
    type=click.IntRange(min=0),
    show_default=True,
    help="Repeat model training/evaluation with incremented model seeds; 0 or 1 keeps single standard run.",
)

def classify_signals(metadata, output, models, dataset_stats, split_size, smoothing, cutoff,
                     order, testing, seed, plot_dtw, norm_method,
                     padding_method, fixed_length, padding_mode, trimming, trim_fraction,
            min_alignment_position, feature_test, feature_set, model_seed_runs):
    """Classify peptide signals from different samples."""
    os.makedirs(output, exist_ok=True)
    num_processes = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    feature_set = str(feature_set).lower()

    feature_backend = "auto"
    feature_chunk_size = 0
    read_engine = "auto"
    inception_weighted_loss = "auto"
    inception_imbalance_threshold = 2.0
    inception_batch_size = 0
    inception_large_batch_size = 0
    timesnet_batch_size = 0
    n_model_seed_runs = max(1, int(model_seed_runs or 1))
    logger.info("Using %d processes; seed=%d", num_processes, seed)
    logger.info("Model seed runs per model: %d", n_model_seed_runs)
    inception_early_stopping = True
    inception_early_stopping_patience = 50
    inception_early_stopping_min_delta = 0.0
    random.seed(seed)
    np.random.seed(seed)

    df_meta = pd.read_csv(metadata)
    df_train_all = df_meta[df_meta["dataset"] == "train"].reset_index(drop=True)
    df_test_all  = df_meta[df_meta["dataset"] == "test"].reset_index(drop=True)
    logger.info("Train samples: %d, Test samples: %d", len(df_train_all), len(df_test_all))

    scenarios = {
        "ßCAT_binary_WW":        {"label": ["ßCAT", "ßCATWW"]},
        "ßCAT_binary_W":        {"label": ["ßCAT", "ßCATW"]},
        "ßCAT_single_variants": {"label": ["ßCAT", "ßCATD", "ßCATL", "ßCATW"]},
        "ALL_classes":          {"label": None},
        "ßCAT_single_double_variants": {"label": ["ßCAT", "ßCATD", "ßCATW", 
                                       "ßCATL", "ßCATWW", "ßCATWWW", "ßCATGG"]},
        
    }
    models_list = [m.strip() for m in models.split(',') if m.strip()] \
                  or ["MiniRocket", "featureLGBM", "InceptionTime", "TimesNet"]
    n_epochs = 1000
    has_inception_model = any(is_inception_model(m) for m in models_list)

    ds_kwargs = dict(smoothing=smoothing, norm_method=norm_method, fixed_length=fixed_length,
                     padding_mode=padding_mode, cutoff=cutoff, order=order,
                     trimming=trimming, trim_fraction=trim_fraction)
    inception_ds_kwargs, inception_overrides = resolve_inception_preprocessing(ds_kwargs, models_list)
    if inception_overrides:
        logger.info("Inception-like preprocessing overrides: %s", inception_overrides)

    preloaded_train_df = None
    preloaded_test_df = None
    if not dataset_stats:
        logger.info("Preloading parsed train/test signals once for all scenarios...")
        train_sources = df_train_all[["file_path", "class"]].drop_duplicates().reset_index(drop=True)
        test_sources = df_test_all[["file_path", "class"]].drop_duplicates().reset_index(drop=True)

        preloaded_train_df = process_parse_inputfiles(
            train_sources["file_path"].tolist(),
            train_sources["class"].tolist(),
            output,
            dataset_stats=False,
            testing=testing,
            seed=seed,
            min_alignment_position=min_alignment_position,
            read_engine=read_engine,
        )
        preloaded_test_df = process_parse_inputfiles(
            test_sources["file_path"].tolist(),
            test_sources["class"].tolist(),
            output,
            dataset_stats=False,
            testing=testing,
            seed=seed,
            min_alignment_position=min_alignment_position,
            read_engine=read_engine,
        )

        if preloaded_train_df is not None:
            logger.info("Preloaded train signals: %d", len(preloaded_train_df))
        if preloaded_test_df is not None:
            logger.info("Preloaded test signals: %d", len(preloaded_test_df))

    results_all_scenarios = {}
    for scenario_name, scenario_info in scenarios.items():
        scen_out = os.path.join(output, scenario_name)
        os.makedirs(scen_out, exist_ok=True)
        logger.info("=== Scenario: %s ===", scenario_name)

        # Filter metadata by scenario labels
        labels = scenario_info["label"]
        included = (lambda lbl: True) if labels is None else (lambda lbl: lbl in labels)
        df_train = df_train_all[df_train_all["class"].apply(included)].reset_index(drop=True)
        df_test  = df_test_all [df_test_all ["class"].apply(included)].reset_index(drop=True)
        logger.info("  train %d/%d, test %d/%d",
                    len(df_train), len(df_train_all), len(df_test), len(df_test_all))
        if df_train.empty:
            logger.warning("No training rows → skipping")
            continue
        if df_test.empty:
            logger.warning("Test set empty → running without test evaluation")

        # Load and split train/val
        if preloaded_train_df is not None:
            train_val_df = preloaded_train_df if labels is None else preloaded_train_df[preloaded_train_df["Class"].isin(labels)]
        else:
            train_val_df = process_parse_inputfiles(
                df_train["file_path"].tolist(), df_train["class"].tolist(), scen_out,
                dataset_stats=dataset_stats, testing=testing, seed=seed,
                min_alignment_position=min_alignment_position,
                read_engine=read_engine)
        if not (train_val_df is not None and len(train_val_df)):
            logger.warning("No processed data → skipping")
            continue

        train_val_df = train_val_df.reset_index(drop=True)
        valid_cls = train_val_df['Class'].value_counts()
        valid_cls = valid_cls[valid_cls >= 2].index
        n_dropped = len(train_val_df) - train_val_df['Class'].isin(valid_cls).sum()
        if n_dropped:
            logger.warning("Removed %d samples from classes with <2 members", n_dropped)
        train_val_df = train_val_df[train_val_df['Class'].isin(valid_cls)].reset_index(drop=True)
        train_sub, val_sub = train_test_split(
            train_val_df, test_size=split_size,
            stratify=train_val_df['Class'], random_state=seed)

        train_stats = compute_train_stats(
            train_sub,
            [trim_padding(np.asarray(s)) for s in train_sub['Signal'].values],
            method=norm_method)

        ds = prepare_scenario_datasets(
            train_sub, val_sub, train_stats=train_stats, outdir=scen_out, **ds_kwargs)
        fixed_len = ds["info_train"]["target_length"]

        # InceptionTiime-specific preprocessing branch.
        inception_train_stats = train_stats
        ds_inception = ds
        if has_inception_model:
            inception_norm_method = inception_ds_kwargs.get("norm_method", norm_method)
            if inception_norm_method != norm_method:
                inception_train_stats = compute_train_stats(
                    train_sub,
                    [trim_padding(np.asarray(s)) for s in train_sub['Signal'].values],
                    method=inception_norm_method,
                )
            if inception_ds_kwargs != ds_kwargs or inception_train_stats is not train_stats:
                ds_inception = prepare_scenario_datasets(
                    train_sub,
                    val_sub,
                    train_stats=inception_train_stats,
                    outdir=scen_out,
                    **inception_ds_kwargs,
                )
        fixed_len_inception = ds_inception["info_train"]["target_length"]

        test_df, ds_test, ds_test_inception = None, {}, {}
        if not df_test.empty:
            if preloaded_test_df is not None:
                test_df = preloaded_test_df if labels is None else preloaded_test_df[preloaded_test_df["Class"].isin(labels)]
            else:
                test_df = process_parse_inputfiles(
                    df_test["file_path"].tolist(), df_test["class"].tolist(), scen_out,
                    dataset_stats=dataset_stats, testing=testing, seed=seed,
                    min_alignment_position=min_alignment_position,
                    read_engine=read_engine)
            if test_df is not None and len(test_df):
                test_kw = {**ds_kwargs, "fixed_length": fixed_len,
                           "train_stats": train_stats, "outdir": scen_out,
                           "padding_method": 'fixed'}
                X_test, y_test, X_test_raw, y_test_runID, _ = prepare_dataset_from_df(
                    test_df, **{**test_kw, "padding_mode": padding_mode})
                X_test_IT, y_test_IT, _, _, _ = prepare_dataset_from_df(
                    test_df, **{**test_kw, "padding_mode": "truncate_or_pad"})
                ds_test = dict(X=X_test, y=y_test, X_raw=X_test_raw,
                               X_IT=X_test_IT, y_IT=y_test_IT, runID=y_test_runID)

                if has_inception_model:
                    if ds_inception is ds:
                        ds_test_inception = ds_test
                    else:
                        test_kw_it = {
                            **inception_ds_kwargs,
                            "fixed_length": fixed_len_inception,
                            "train_stats": inception_train_stats,
                            "outdir": scen_out,
                            "padding_method": 'fixed',
                        }
                        X_test_IT_i, y_test_IT_i, _, y_test_runID_i, _ = prepare_dataset_from_df(
                            test_df,
                            **{**test_kw_it, "padding_mode": "truncate_or_pad"},
                        )
                        ds_test_inception = dict(X_IT=X_test_IT_i, y_IT=y_test_IT_i, runID=y_test_runID_i)
            else:
                test_df = None

        log_length_directory = os.path.join(scen_out, "length_distributions")
        os.makedirs(log_length_directory, exist_ok=True)
        log_length_distributions(train_sub, df_name='train_sub',
                                 out_csv=os.path.join(log_length_directory, 'train_length_stats.csv'))
        log_length_distributions(val_sub, df_name='val_sub',
                                 out_csv=os.path.join(log_length_directory, 'val_length_stats.csv'))
        if test_df is not None:
            log_length_distributions(test_df, df_name='test_metadata_filtered',
                                     out_csv=os.path.join(log_length_directory, 'test_metadata_filtered_length_stats.csv'))

        logger.info("[%s] shapes — train: %s  val: %s  test: %s",
                    scenario_name, ds["X_train"].shape, ds["X_val"].shape,
                    ds_test["X"].shape if ds_test.get("X") is not None else "None")

        le = LabelEncoder()
        y_train_int = le.fit_transform(np.asarray(ds["y_train"]))
        n_classes = len(le.classes_)
        class_weights_array = compute_class_weight('balanced', classes=np.arange(n_classes), y=y_train_int)
        logger.info("[%s] %d classes, counts: %s",
                    scenario_name, n_classes, dict(zip(*np.unique(y_train_int, return_counts=True))))

        try:
            X_all = safe_concat([ds["X_train"], ds["X_val"], ds_test.get("X")])
        except ValueError:
            logger.warning(
                "[%s] Mixed signal lengths across splits; using truncate_or_pad arrays for DTW plotting.",
                scenario_name,
            )
            try:
                X_all = safe_concat([ds["X_train_IT"], ds["X_val_IT"], ds_test.get("X_IT")])
            except ValueError:
                logger.warning(
                    "[%s] truncate_or_pad arrays still mixed; resampling all DTW inputs to fixed_length=%d.",
                    scenario_name,
                    fixed_length,
                )
                sigs_for_dtw = []
                for arr in (ds["X_train_IT"], ds["X_val_IT"], ds_test.get("X_IT")):
                    if arr is not None:
                        sigs_for_dtw.extend([np.asarray(s) for s in arr])
                X_all = resample_signals_to_length(sigs_for_dtw, target_len=fixed_length, n_jobs=num_processes)
        y_all = safe_concat([np.asarray(ds["y_train"]), np.asarray(ds["y_val"]),
                             np.asarray(ds_test.get("y")) if ds_test.get("y") is not None else None])
        run_ids_all = safe_concat([np.asarray(ds["y_train_runID"]), np.asarray(ds["y_val_runID"]),
                                   np.asarray(ds_test["runID"]) if ds_test.get("runID") is not None else None])


        if plot_dtw and "ßCAT" in scenario_name:
            run_dtw_plots(X_all, y_all, run_ids_all, scen_out, scenario_name,
                           testing, seed, num_processes, fixed_length)

        # Per-model training & evaluation
        scenario_results = {}
        for mname in models_list:
            model_ds = ds_inception if is_inception_model(mname) else ds
            model_ds_test = ds_test_inception if is_inception_model(mname) else ds_test
            run_records = []

            for run_idx in range(1, n_model_seed_runs + 1):
                run_seed = int(seed) + (run_idx - 1)
                model_dir_name = mname if n_model_seed_runs == 1 else f"{mname}-{run_idx}"
                model_out = os.path.join(scen_out, model_dir_name)
                os.makedirs(model_out, exist_ok=True)

                logger.info(
                    "[%s] Running model: %s (run %d/%d, model_seed=%d)",
                    scenario_name,
                    mname,
                    run_idx,
                    n_model_seed_runs,
                    run_seed,
                )

                run_result = run_model(
                    mname, model_out, model_ds, model_ds_test,
                    class_weights_array=class_weights_array, n_epochs=n_epochs,
                    num_processes=num_processes, scenario_name=scenario_name,
                    padding_mode=padding_mode, fixed_length=fixed_length, seed=run_seed,
                    feature_test=feature_test, feature_set=feature_set,
                    feature_backend=feature_backend, feature_chunk_size=feature_chunk_size,
                    inception_weighted_loss_mode=inception_weighted_loss,
                    inception_imbalance_threshold=inception_imbalance_threshold,
                    inception_batch_size=inception_batch_size,
                    inception_large_batch_size=inception_large_batch_size,
                    timesnet_batch_size=timesnet_batch_size,
                    inception_early_stopping=inception_early_stopping,
                    inception_early_stopping_patience=inception_early_stopping_patience,
                    inception_early_stopping_min_delta=inception_early_stopping_min_delta,
                )

                run_records.append({
                    "run_index": run_idx,
                    "run_seed": run_seed,
                    "run_dir": model_dir_name,
                    "result": run_result,
                    "val": run_result.get("val"),
                    "test": run_result.get("test"),
                })

            if n_model_seed_runs <= 1:
                scenario_results[mname] = run_records[0]["result"]
            else:
                median_run_index = summarize_model_seed_runs(mname, run_records, scen_out)
                representative = next((r for r in run_records if int(r["run_index"]) == int(median_run_index)), run_records[0])
                scenario_results[mname] = representative["result"]
                logger.info(
                    "[%s] Selected median run for %s: run %d (%s)",
                    scenario_name,
                    mname,
                    int(representative["run_index"]),
                    str(representative["run_dir"]),
                )

        results_all_scenarios[scenario_name] = scenario_results

        if scenario_name == "ALL_classes":
            all_classes_postprocessing(results_all_scenarios, X_all, y_all,
                                        train_sub, val_sub, test_df, scen_out)

    return results_all_scenarios


if __name__ == "__main__":
    main()
    