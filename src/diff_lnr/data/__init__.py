"""Data loading and splitting.

Torch-dependent dataset objects are imported lazily so that data curation and
configuration checks can run in lightweight environments.
"""

from .records import DeletionRecord, LoadResult, load_deletion_records
from .splits import SplitIndices, stratified_holdout_split

__all__ = [
    "DeletionRecord",
    "LoadResult",
    "SplitIndices",
    "load_deletion_records",
    "stratified_holdout_split",
]
