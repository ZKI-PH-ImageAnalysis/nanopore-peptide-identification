#!/usr/bin/env python

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "workflow" / "scripts"))

import logging
import click

from figure_bundle_io import (
    get_repo_figures_dir,
    load_pickle_bundle,
)

from utils_dtw_alignment import (
    plot_embeddings_panel,
    plot_feature_distributions_panel,
    plot_peptide_medoids_panel,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to exported Figure 2 data bundle (e.g., results/ßCAT_single_variants/dtw-analysis/)",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output directory for figures. Defaults to same as data-dir.",
)
def replay_figure_2(data_dir, output):
    data_dir = Path(data_dir)
    if output is None:
        output = data_dir
    else:
        output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Figure 2 data from {data_dir}")

    # Load bundled data
    medoids_pkl = data_dir / "fig2_medoids_bundle.pkl"
    features_pkl = data_dir / "fig2_features_bundle.pkl"
    embeddings_pkl = data_dir / "fig2_embeddings_bundle.pkl"

    if not medoids_pkl.exists():
        logger.error(f"Missing {medoids_pkl}; skipping Figure 2A")
        return

    try:
        medoids_bundle = load_pickle_bundle(medoids_pkl)
        logger.info("Loaded Figure 2A bundle")
        plot_peptide_medoids_panel(**medoids_bundle, outdir=output)
    except Exception as e:
        logger.error(f"Failed to replay Figure 2A: {e}")

    if features_pkl.exists():
        try:
            features_bundle = load_pickle_bundle(features_pkl)
            logger.info("Loaded Figure 2B bundle")
            plot_feature_distributions_panel(**features_bundle, outdir=output)
        except Exception as e:
            logger.error(f"Failed to replay Figure 2B: {e}")
    else:
        logger.warning(f"Missing {features_pkl}; skipping Figure 2B")

    if embeddings_pkl.exists():
        try:
            embeddings_bundle = load_pickle_bundle(embeddings_pkl)
            logger.info("Loaded Figure 2C bundle")
            plot_embeddings_panel(**embeddings_bundle, outdir=output)
        except Exception as e:
            logger.error(f"Failed to replay Figure 2C: {e}")
    else:
        logger.warning(f"Missing {embeddings_pkl}; skipping Figure 2C")

    logger.info(f"Figure 2 replay complete. Output: {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    replay_figure_2()
