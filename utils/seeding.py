"""Global random seeding utilities for deterministic execution across experiments."""

import random
import os
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set global random seeds for Python random, NumPy, and TensorFlow.

    Args:
        seed (int): The integer seed value for random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
