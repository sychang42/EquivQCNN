r"""
Utility methods for QCNN training. 
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))


import flax.linen as nn
from flax.training.train_state import TrainState

import jax.numpy as jnp

import optax

from qcnn_classifier import QCNNClassifier
from circuits.quantum_circuit import get_quantum_circuit

from typing import Optional, Tuple, Dict, Union, List, Callable, Any
import pandas as pd

PRNGKey = jnp.ndarray


def choose_model(model_type: str, model_args: Dict[str, Any]) -> Callable:
    r"""Picks and loads one of the implemented classifier model classes with the given
    hyperparmaeters.

    Args
        model_type (str) : Model type. Currently, only the quantum classifier is
            implemented.
        model_args (Dict[str, Any]) : Model parameters.
    Returns:
        Callable: The loaded quantum classifier model with the given configuration.
    """
    model_cls = None
    if model_type == "quantum_classifier":
        kwargs = {}
        if "ver" in model_args.keys():
            kwargs["qnn_ver"] = model_args["ver"]

        if "qnn_config" in model_args.keys():
            kwargs["qnn_config"] = model_args["qnn_config"]

        if "symmetry_breaking" in model_args.keys():
            kwargs["sym_break"] = model_args["symmetry_breaking"]

        circuit, num_params = get_quantum_circuit(
            model_args["num_wires"],
            model_args["num_measured"],
            equiv=model_args["equiv"],
            trans_inv=model_args["trans_inv"],
            **kwargs,
        )

        model_cls = QCNNClassifier
        model_args = {
            "circuit": circuit,
            "num_params": num_params,
            "equiv": model_args["equiv"],
            "delta": model_args["delta"],
        }

    return model_cls, model_args


def create_state(
    rng: PRNGKey,
    model_cls: nn.Module,
    input_shape: Union[Tuple[int], List[Tuple[int]]],
    input_args: Optional[Dict] = None,
    opt_args: Optional[Dict] = None,
) -> TrainState:
    r"""Function to create train state of input class

    Args:
        rng (PRNGKey): Random number generator key
        model_cls (nn.Module): Flax class to create trainstate
        input_shape (Tuple[int]): Input data shape
        input_args (Dict): Input argument for trainstate class
        opt_args (Dict): Optimizer arguments

    Returns:
        TrainState: Initial training state.
    """

    if opt_args is None:
        opt_args = {"lr": 0.01, "b1": 0.9, "b2": 0.999}

    model = model_cls(**input_args)
    tx = optax.adam(opt_args["lr"], b1=opt_args["b1"], b2=opt_args["b2"])
    #     tx = optax.amsgrad(opt_args['lr'], b1=opt_args['b1'], b2=opt_args['b2'])

    # In case we add regularization
    if "weight_decay" in opt_args.keys():
        tx = optax.adamw(
            opt_args["lr"],
            b1=opt_args["b1"],
            b2=opt_args["b2"],
            weight_decay=opt_args["weight_decay"],
        )

    variables = model.init(rng, jnp.ones(input_shape))

    state = TrainState.create(apply_fn=model.apply, tx=tx, params=variables["params"])

    return state


def init_trainstate(
    model_args: Tuple[Dict], opt_args: Tuple[Dict], input_shape: Tuple, key: PRNGKey
) -> Tuple[TrainState, PRNGKey]:
    model_cls, model_args = choose_model("quantum_classifier", model_args)

    model_state = create_state(key, model_cls, input_shape, model_args, opt_args)

    return model_state, key


def save_outputs(
    epoch: int, snapshot_dir: str, outputs: Dict[str, jnp.ndarray], labels: jnp.ndarray
) -> None:
    df = pd.DataFrame({"preds": outputs["preds"]})
    df["labels"] = labels

    df.to_csv(os.path.join(snapshot_dir, "classification_epoch" + str(epoch) + ".csv"))


def print_losses(
    epoch: int,
    epochs: int,
    train_loss: Dict[str, jnp.ndarray],
    valid_loss: Dict[str, jnp.ndarray],
):
    """
    Print the training and validation losses.

    Args :
        epoch (int) : Current epoch
        epochs (int) : Total number of epohcs
        train_loss (dict) : Training loss
        valid_loss (dict) : Validation loss
    """
    print(
        f"Epoch : {epoch + 1}/{epochs}, Train loss (average) = "
        f", ".join("{}: {}".format(k, v) for k, v in train_loss.items())
    )
    print(
        f"Epoch : {epoch + 1}/{epochs}, Valid loss = "
        f", ".join("{}: {}".format(k, v) for k, v in valid_loss.items())
    )
