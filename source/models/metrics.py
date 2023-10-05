r"""
Losses used for quantum classifier training.
"""

import jax
from jax import Array
import jax.numpy as jnp

from typing import Callable


@jax.jit
def BCE_loss(labels: Array, x: Array) -> Array:
    r"""Measures the Binary Cross Entropy (BCE) loss between each element in the one-hot
    encoded target :math:`x` and the input :math:`y` given by the equations :


    .. math::
        \ell(x, y) = - \sum_{n=1}^N_\mathbf{x}_n \cdot \log (\mathbf{y}_n),

    where :math:`N` is the batch size.

    Args:
        x (Array): Targets of shape ``(N, L)``
        y (Array): Targets of shape ``(N, L)``

    Returns:
        Array: BCE loss value.
    """
    num_classes = x.shape[1]

    return -jnp.mean(jnp.sum(jax.nn.one_hot(labels, num_classes) * jnp.log(x), axis=-1))


@jax.jit
def MSE_loss(x: Array, y: Array) -> Array:
    r"""Measures the Mean Squared Error (MSE) loss between each element in the target
    :math:`x` and the input :math:`y` given by the equation :

    .. math::
        \ell_{\text{MSE}}(x, y) = \frac{1}{N}\sum_{n=1}^N\sqrt{\left( x_n - y_n \right)^2},

    where :math:`N` is the number of elements in :math:`x` and :math:`y`.

    Args:
        x (Array): Targets of shape ``(N, 1)``
        y (Array): Inputs of shape ``(N, 1)``.

    Returns:
        Array: MSE loss value.
    """
    return jnp.mean((x - y) ** 2)


@jax.jit
def accuracy(target: Array, pred: Array) -> Array:
    r"""Measures accuracy between the target labels ``target`` and the predicted labels
    ``pred``.

    Args:
        target (Array): The target labels.
        pred (Array): The predicted labels.

    Returns:
        Array: Accuracy caculated between ``target`` and ``pred``.
    """
    accuracy = jnp.sum(jnp.argmax(pred, axis=1) == target) / len(pred)

    return accuracy


def get_metrics(loss_type: str) -> Callable:
    r"""

    Args:
        loss_type (List[str]):

    Returns:

    """

    r"""Function to return Callable corresponding the given loss types.

    Args:
        loss_type (str): String representing the loss types to be returned.

    Raises:
        TypeError: Return error if the given loss type is not implemented.

    Returns:
        Callable: The loss function corresponding to the given ``loss_type``.
    """

    switcher = {"MSE_loss": MSE_loss, "BCE_loss": BCE_loss, "accuracy": accuracy}
    loss = switcher.get(loss_type, lambda: None)
    if loss is None:
        raise TypeError("Specified loss does not exist!")

    return loss
