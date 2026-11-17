"""Training and evaluation entry points; implementation is in ``tourplan.nn``."""
from tourplan.nn import duration_metrics, next_metrics, train_all

__all__ = ["duration_metrics", "next_metrics", "train_all"]
