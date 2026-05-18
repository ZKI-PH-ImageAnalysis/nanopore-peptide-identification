#!/usr/bin/env python
"""
Panels:
  - 3B: confusion_matrices_B_diagacc_validation (ßCAT_single_variants)
  - 3C: confusion_matrices_B_diagacc_test (ßCAT_single_variants)
  - 3D: confusion_matrices_B_diagacc_validation (ALL_classes)
  - 3E: Coverage_accuracy_test (ALL_classes)
  - 3F: confusion_matrices_unknown_mixtures_onlyPredictedPortions_predicted_only (test)
"""
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "workflow" / "scripts"))

import click
import numpy as np

from figure_bundle_io import get_repo_figures_dir, load_pickle_bundle
from utils_plot_classification import (
    plot_confidence_accuracy_curve,
    plot_confusion_matrices,
    plot_unknown_mixtures,
    plot_unknown_predicted_only,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to exported Figure 3 data bundle (e.g., results/classification/)",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output directory for figures. Defaults to Figures/Figure3.",
)
def replay_figure_3(data_dir, output):
    data_dir = Path(data_dir)
    if output is None:
        output = get_repo_figures_dir() / "Figure3"
    else:
        output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Figure 3 data from {data_dir}")

    scenarios_and_panels = [
        ("ßCAT_single_variants/InceptionTime", "fig3_bcat_single_val_test_metrics.pkl"),
        ("ALL_classes/InceptionTime", "fig3_all_classes_val_test_metrics.pkl"),
    ]

    for scenario_path, metrics_file in scenarios_and_panels:
        metrics_pkl = data_dir / scenario_path / metrics_file
        if not metrics_pkl.exists():
            logger.warning(f"Missing {metrics_pkl}")
            continue

        try:
            bundle = load_pickle_bundle(metrics_pkl)
            val_metrics = bundle.get("val_metrics")
            test_metrics = bundle.get("test_metrics")
            scenario_name = bundle.get("scenario_name", scenario_path)

            if val_metrics and test_metrics:
                logger.info(f"Replaying confusion matrices for {scenario_name}")
                plot_confusion_matrices(
                    str(output / scenario_name), val_metrics, test_metrics
                )
        except Exception as e:
            logger.error(f"Failed to replay metrics for {scenario_path}: {e}")

    unknown_pkl = data_dir / "ALL_classes/InceptionTime/fig3_unknown_test_metrics.pkl"
    if unknown_pkl.exists():
        try:
            bundle = load_pickle_bundle(unknown_pkl)
            test_metrics = bundle.get("test_metrics")
            if test_metrics:
                logger.info("Replaying unknown mixtures panels")
                out_unknown = output / "unknown_mixtures"
                out_unknown.mkdir(exist_ok=True)
                plot_unknown_predicted_only(test_metrics, str(out_unknown))
                plot_unknown_mixtures(test_metrics, str(out_unknown))
        except Exception as e:
            logger.error(f"Failed to replay unknown mixtures: {e}")
    else:
        logger.warning(f"Missing {unknown_pkl}; skipping unknown mixtures panels")

    logger.info(f"Figure 3 replay complete. Output: {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    replay_figure_3()
