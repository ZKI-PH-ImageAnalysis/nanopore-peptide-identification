#!/usr/bin/env python

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def get_repo_figures_dir():
    return Path(__file__).resolve().parents[2] / "Figures"


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_pickle_bundle(path, payload):
    ensure_parent_dir(path)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle_bundle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_json(path, payload):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="list")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
