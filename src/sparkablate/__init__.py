"""sparkablate: component-ablation experiments for open-weight causal LMs."""

from sparkablate.hooks import AblationManager, AblationSpec
from sparkablate.eval import evaluate_perplexity, iter_eval_batches

__version__ = "0.1.0"

__all__ = [
    "AblationManager",
    "AblationSpec",
    "evaluate_perplexity",
    "iter_eval_batches",
    "__version__",
]
