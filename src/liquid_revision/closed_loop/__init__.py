"""Auditable training and response-surface evaluation for the closed-loop task."""

from .manifests import create_trajectory_manifest, prepare_split_datasets
from .parameter_matching import create_parameter_registry
from .training import train_registered_model

__all__ = [
    "create_parameter_registry",
    "create_trajectory_manifest",
    "prepare_split_datasets",
    "train_registered_model",
]
