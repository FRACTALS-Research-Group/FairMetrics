"""
    Binarization utilities for target vectors in fairness experiments.

    This module provides functions to binarize target vectors according to different modes.
    The available modes include:
    - 'mul': Keep the original multiclass target.
    - 'bin0', 'bin1', 'bin2': Create binary targets for the respective classes.
"""
import numpy as np


# Available binarization modes for target vectors.
# This set contains all the valid modes that can be used for binarizing the target vector. The 'mul' mode keeps the original multiclass target, while 'bin0', 'bin1', and 'bin2' create binary targets for the respective classes.
_AVAILABLE_MODES = {"mul", "bin0", "bin1", "bin2"}


def binarize_target(y, mode, class_order=None):
    """
        Return a target for the selected fairness strategy.

        Parameters:
        - y: Original target vector.
        - mode: Binarization mode.
        - class_order: Optional order of classes for binarization.

        Returns:
        - Binarized target vector according to the specified mode.

        Supported modes:
        - mul: keep the original multiclass target
        - bin0: label 0 vs rest
        - bin1: label 1 vs rest
        - bin2: label 2 vs rest
    """
    if mode not in _AVAILABLE_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {sorted(_AVAILABLE_MODES)}")

    y = np.asarray(y)
    if mode == "mul":
        return y.copy()

    if class_order is None:
        class_order = np.unique(y)
    else:
        class_order = np.asarray(class_order)

    positive_class = int(mode.replace("bin", ""))
    if positive_class >= len(class_order):
        raise ValueError(
            f"Mode '{mode}' requires a class index in the range [0, {len(class_order)-1}] "
            f"but received {positive_class}."
        )

    positive_value = class_order[positive_class]
    return (y == positive_value).astype(int)


__all__ = ["binarize_target", "_AVAILABLE_MODES"]
