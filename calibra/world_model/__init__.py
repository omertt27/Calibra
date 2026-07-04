"""calibra.world_model — lightweight world-model data curation (v1 baseline)."""

from calibra.world_model.surprise import (
    LatentEncoder,
    LinearLatentPredictor,
    WorldModelCurationResult,
    compute_surprise_scores,
    curate_for_world_model,
)

__all__ = [
    "LatentEncoder",
    "LinearLatentPredictor",
    "WorldModelCurationResult",
    "compute_surprise_scores",
    "curate_for_world_model",
]