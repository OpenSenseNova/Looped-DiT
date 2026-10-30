"""Adaptive looping: a small controller decides after each loop pass whether to run another.

    controller.py  LoopController, checkpoint loading, threshold lookup
    labels.py      per-exit errors of the frozen model and the exit targets
    sampler.py     adaptive prediction, Euler sampler, generate_adaptive
    data.py        held-out caption split of the training data
    train.py       python -m looped_dit.adaptive.train      (train a controller)
    calibrate.py   python -m looped_dit.adaptive.calibrate  (thresholds for target average loop counts)
    eval.py        adaptive options of looped_dit.sample and looped_dit.eval
"""

from .controller import LoopController, load_controller, resolve_threshold
from .labels import exit_rollout, exit_targets
from .sampler import adaptive_euler_sample, adaptive_prediction, generate_adaptive

__all__ = [
    "LoopController",
    "adaptive_euler_sample",
    "adaptive_prediction",
    "exit_rollout",
    "exit_targets",
    "generate_adaptive",
    "load_controller",
    "resolve_threshold",
]
