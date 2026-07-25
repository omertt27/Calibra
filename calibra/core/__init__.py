"""
calibra.core — public API layer for dataset reading and schema normalization.

Exposes:
  LazyDatasetReader  : SQL-level zero-copy interface for local v2 LeRobot datasets.
  SchemaNormalizer   : YAML-configurable column name normalization.
"""

from calibra.core.normalizer import SchemaNormalizer
from calibra.core.reader import LazyDatasetReader

__all__ = ["LazyDatasetReader", "SchemaNormalizer"]
