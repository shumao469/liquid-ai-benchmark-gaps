"""Leakage-safe offline sequence benchmarks for the major revision."""

from .models import OfflineSequenceClassifier, build_offline_model

__all__ = ["OfflineSequenceClassifier", "build_offline_model"]
