#!/usr/bin/env python
import logging
import pickle
from pathlib import Path

import click
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "workflow" / "scripts"))

from figure_bundle_io import get_repo_figures_dir
from utils_interpretability import replay_figure_4_from_bundle

logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to exported Figure 4 data bundle (e.g., results/classification-featuresLGBM/ßCAT_single_variants/featuresLGBM/)",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output directory for figures. Defaults to Figures/Figure4.",
)
def replay_figure_4(data_dir, output):
    logging.basicConfig(level=logging.INFO)
    
    data_dir = Path(data_dir)
    if output is None:
        output = get_repo_figures_dir() / "Figure4"
    else:
        output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Figure 4 data from {data_dir}")

    # load the figure bundle
    bundle_pkl = data_dir / "fig4_interpretability_bundle.pkl"
    if not bundle_pkl.exists():
        logger.error(f"Missing {bundle_pkl}")
        logger.error(f"Looked in: {bundle_pkl}")
        return

    try:
        with open(bundle_pkl, "rb") as f:
            bundle = pickle.load(f)
        logger.info("Loaded Figure 4 interpretability bundle")

        png_path, svg_path = replay_figure_4_from_bundle(bundle, str(output))
        logger.info(f"Figure 4 saved to {png_path}")

    except Exception as e:
        logger.error(f"Failed to load/replay Figure 4: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    replay_figure_4()

