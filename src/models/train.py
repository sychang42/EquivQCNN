import sys
import os

sys.path.append(os.path.dirname(__file__))


import jax

import jax.numpy as jnp

import numpy as np
from metrics import get_metrics

from flax.training.train_state import TrainState
from typing import Tuple, Optional, Dict, Any

import csv
from tqdm import tqdm
from time import time

from model_utils import init_trainstate, print_losses, save_outputs


@jax.jit
def compute_metrics(
    pred: jnp.ndarray, labels: jnp.ndarray
) -> Tuple[float, Dict[str, float]]:
    r"""Compute training metrics (Only BCE loss and accuracy implemented for the moment)

    Args:
        pred (jnp.ndarray) : Classifier outputs
        labels (jnp.ndarray) : Data labels (Ground truth)

    Returns:
        Tuple[float, Dict[str, float]]: Tuple conatining the total loss, with respect
        to which we take the gradient and the dictionary of each losses
    """
    loss_type = ["BCE_loss", "accuracy"]

    loss = 0.0
    losses = {}

    for l in loss_type:
        losses[l] = get_metrics(l)(labels, pred)
        if "loss" in l:
            loss += losses[l]

    return loss, losses


@jax.jit
def train_batch(
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    model_state: TrainState,
) -> Tuple[TrainState, Dict[str, float], jnp.ndarray]:
    r"""
    Train the model on a single batch.

    Args:
        x_batch (jnp.ndarray) : Classical input data of shape :math:`(batchsize,
            img_size[0],img_size[1], img_size[2])`.
        y_batch (jnp.ndarray) : Input data labels of shape :math:`(batchsize, )`.
        model_state (TrainState) : Quantum Classifier model train state.

    Returns:
        Tuple[TrainState, Dict[str, float], jnp.ndarray]: Tuple containing the updated
        model train state, the dictionary of losses computed on ``x_batch``, ``y_batch``,
        and the quantum classifier outputs.
    """

    def loss_fn(params):
        class_outputs = model_state.apply_fn({"params": params}, x_batch)
        loss, losses = compute_metrics(class_outputs, y_batch)

        return loss, (losses, class_outputs)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (losses, class_outputs)), grads = grad_fn(model_state.params)

    # Update the generator parameters.
    new_model_state = model_state.apply_gradients(grads=grads)

    return new_model_state, losses, class_outputs


@jax.jit
def validate(
    x_batch: jnp.ndarray, y_batch: jnp.ndarray, model_state: TrainState
) -> Tuple[Dict, jnp.ndarray]:
    r"""Validate the model on a single batch.

    Args:
        x_batch (jnp.ndarray) :
        y_batch (jnp.ndarray) :
        model_state (TrainState) :

    Returns:
        losses (Dict[str, float]) : Dictionary of losses computed on x_batch, y_batch.
        preds (jnp.ndarray) :

    Args:
        x_batch (jnp.ndarray): Classical input data of shape :math:`(batchsize,
            img_size[0], img_size[1], img_size[2])`.
        y_batch (jnp.ndarray): Input data labels of shape :math:`(batchsize, ).
        model_state (TrainState): Quantum Classifier model train state (Not trained).

    Returns:
        Tuple[Dict, jnp.ndarray]: Tuple containing the dictionary of losses computed on
        ``x_batch``, ``y_batch``, and the predicted labels of shape :math:`(batchsize,
        )`.
    """

    class_outputs = model_state.apply_fn({"params": model_state.params}, x_batch)

    _, losses = compute_metrics(class_outputs, y_batch)
    if class_outputs.shape[1] == 1:
        preds = {"preds": class_outputs}
    else:
        preds = {"preds": jnp.argmax(class_outputs, axis=1)}
    return losses, preds


def train_model(
    train_ds: Dict[str, jnp.ndarray],
    test_ds: Dict[str, jnp.ndarray],
    train_args: Dict[str, Any],
    model_args: Dict[str, Any],
    opt_args: Dict[str, float],
    seed: int,
    snapshot_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    r"""Train the non-equivariant and equivaraint Quantum Convolutional Neural Network
    for image classification using the provided datasets and hyperparameters.


    Args:
        train_ds (Dict[str, jnp.ndarray]): Dictionary with training data and labels.
        test_ds (Dict[str, jnp.ndarray]): Dictionary with test data and labels.
        train_args (Dict[str, Any]): Hyperparameters needed for classifier training.
            The dictionary should contain the following points:

            * *num_epochs*: Integer indicating the number of training epochs.
            * *batch_size*: Integer indicating the training batch size.
            * *loss_type*: List of strings representing the loss types used in the
              training. Currently, only :func:`metrics.BCE_loss` and
              :func:`metrics.accuracy` are supported.

        model_args (Dict[str, Any]): Arguments required to constructed the QCNN.

            * *num_wires*: Number of qubits in the quantum classifier.
            * *num_measured*: Number of measured qubits at the end of the circuit. For
              L classes, we measure :math:`\lceil (\log_2(L))\rceil` qubits in case of
              the non-equivariant QCNN, and :math:`\lceil (\log_2(L))\rceil`
              qubits in case of the equivariant QCNN.
            * *equiv*: Boolean to indicate whether an equivariant neural network is used.
              If True, the quantum classifier loads :class:`circuits.qcnn.EquivQCNN`,
              otherwise, it loads :class:`circuits.qcnn.QCNN`.
            * *trans_inv*: Boolean to indicate whether the QCNN is
              translational invariant or not. If True, all filters in a layer share
              identical parameters; otherwise, different parameters are used. (To be
              implemented)
            * *ver*: Quantum circuit architecture version.
            * *symmetry_breaking*: Boolean to indicate whether we use :math:`RZ` gates
              at the end of the quantum circuit in case of the EquivQCNN.

        optim_args (Dict[str, float]): :class:`optax.adam` optimizer hyperparameters.

            * *learning_rate*: Learning rate for the optimizer.
            * *b1*: :math:`\beta_1` value of the Adam optimizer.
            * *b2*: :math:`\beta_2` value of the Adam optimizer.

        seed (int): Seed used to generate random values using JAX's random number
            generator (:class:`jax.random`).
        snapshot_dir (str, optional): Directory to store the training result if not
            None. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Tuple of dictionaries containing the
        training and test set loss progression based on the specified ``loss_type``.

    Example:

        >>> train_args = {"num_epochs": 5, "batch_size" : 1024, "loss_type": ['BCE_loss',
        ...               'accuracy]}
        >>> model_args = {"num_wires": 8, "num_measured" : 2,  "equiv": True,
        ...                 "trans_inv": True, "ver": "noisy_equiv",
        ...                  "symmetry_breaking": True}
        >>> opt_args = {"learning_rate": 0.01, "b1" : 0.9, "b2": 0.999}
        >>> train_loss, test_loss = train(train_ds, test_ds, train_args, model_args,
        ...                         opt_args, "Result")
        >>> print(train_loss)
            {'BCE_loss': [0.6449261327584586, 0.600408172607422, 0.5616568744182586,
            0.5470444520314535, 0.537257703145345], 'accuracy': [0.6750000019868214,
            0.7497685273488361, 0.8037037114302319, 0.8217592616875966, 0.8263888935248058]
            }

    """

    epochs = train_args["num_epochs"]
    batch_size = train_args["batchsize"]
    loss_type = train_args["loss_type"]

    # Image shape
    im_shape = train_ds["image"].shape

    key = jax.random.PRNGKey(seed=seed)
    key, init_key = jax.random.split(key)

    input_shape = (batch_size, im_shape[1], im_shape[2], im_shape[3])
    model_state, key = init_trainstate(model_args, opt_args, input_shape, key)

    # If we store the results
    fieldnames = ["epoch"]
    fieldnames.extend([k + "_train" for k in loss_type])
    fieldnames.extend([k + "_test" for k in loss_type])
    fieldnames.append("time_taken")

    if snapshot_dir is not None:
        with open(os.path.join(snapshot_dir, "output.csv"), mode="w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    # Train autoencoder
    for epoch in tqdm(range(1, epochs + 1), desc="Epoch...", position=0, leave=True):
        train_loss_epoch = {k: [] for k in loss_type}

        start_epoch = time()

        key, init_key = jax.random.split(key)
        perms = jax.random.permutation(key, train_ds_size)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))

        with tqdm(
            total=steps_per_epoch, desc="Training...", leave=False
        ) as progress_bar_train:
            for b in range(steps_per_epoch):
                batch_data = {k: train_ds[k][perms[b]] for k in train_ds.keys()}

                # Update Model.
                model_state, loss, class_outputs = train_batch(
                    batch_data["image"], batch_data["label"], model_state
                )
                for k, v in loss.items():
                    train_loss_epoch[k].append(v)

                progress_bar_train.update(1)

            del batch_data

            val_batch_size = 1024
            test_batch_num = len(test_ds["image"]) // val_batch_size

            valid_losses = {k: [] for k in loss_type}
            valid_outputs = []

            for j in range(test_batch_num):
                batch_data = {
                    k: test_ds[k][j * val_batch_size : (j + 1) * val_batch_size]
                    for k in test_ds.keys()
                }
                valid_loss, outputs = validate(
                    batch_data["image"], batch_data["label"], model_state
                )
                for k, v in valid_loss.items():
                    valid_losses[k].append(v)
                valid_outputs.append(outputs)

            if len(test_ds["image"]) > test_batch_num * val_batch_size:
                batch_data = {
                    k: test_ds[k][test_batch_num * val_batch_size :]
                    for k in test_ds.keys()
                }
                valid_loss, outputs = validate(
                    batch_data["image"], batch_data["label"], model_state
                )
                for k, v in valid_loss.items():
                    valid_losses[k].append(v)
                valid_outputs.append(outputs)

            valid_loss = {k: jnp.mean(jnp.array(v)) for k, v in valid_losses.items()}
            train_loss = {
                k: jnp.mean(jnp.array(v)) for k, v in train_loss_epoch.items()
            }
            outputs = {k: valid_outputs[0][k] for k in valid_outputs[0].keys()}
            for output in valid_outputs[1:]:
                for k, v in output.items():
                    outputs[k] = jnp.concatenate((outputs[k], v), axis=0)

            print_losses(epoch, epochs, train_loss, valid_loss)

            # Save results.
            if snapshot_dir is not None:
                # Store output
                with open(
                    os.path.join(snapshot_dir, "output.csv"), mode="a"
                ) as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                    to_write = {"epoch": epoch}
                    for k, v in train_loss.items():
                        to_write[k + "_train"] = np.mean(np.array(v))

                    for k, v in valid_loss.items():
                        to_write[k + "_test"] = v

                    to_write["time_taken"] = (time() - start_epoch) / 60.0

                    writer.writerow(to_write)

                with open(
                    os.path.join(snapshot_dir, "train_parameters.txt"), mode="a"
                ) as f:
                    for x in model_state.params["qparams"]:
                        f.write(str(x) + " ")
                    f.write("\n")
                if epoch == 1 or epoch % 10 == 0:
                    save_outputs(epoch, snapshot_dir, outputs, test_ds["label"])
    return train_loss, valid_loss
