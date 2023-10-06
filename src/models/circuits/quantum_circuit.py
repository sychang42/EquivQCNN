r"""
Construct the quantum circuit for the quantum classifier
"""

import os
import sys
sys.path.append(os.path.dirname(__file__)) 

import numpy as np

import jax 
import pennylane as qml 

from typing import Tuple, Callable, Optional

from qcnn import *
from data_embedding import * 

def get_quantum_circuit(num_qubits : int, 
                         num_measured : int, 
                          equiv : Optional[bool] = False,
                         trans_inv: Optional[bool] = True, 
                        **kwargs
                         ) -> Tuple[Callable, int]:
    r"""Load quantum classifier circuit constructed with the given configuration including the final measurement.

    Args: 
        num_qubits (int): Number of qubits in the quantum generator.
        num_measured (int): Number of qubits measured at the end of the quantum circuit. 
            In case of the non-equiv QCNN, ``num_measured = ceil(log2(num_classes))``.
            In case of the EquivQCNN, ``num_measured = 2*ceil(log2(num_classes))``. 
        qnn_config (Union[str, Dict[str, Any]]): Quantum Circuit configuration for the learning layers.
        equiv (Optional[bool]): Boolean to indicate whether an equivariant neural network is used.
        trans_inv (Optional[bool]): Boolean to indicate whether the model is constructed in a translational invariant way.

    Returns: 
        Tuple[Callable, int]: Tuple of a Callable representing the quantum classifier circuit, and an int representing 
        the total number of parameters required in the quantum circuit.
    """
    
    if equiv: 
        qcnn = EquivQCNN(num_qubits, num_measured, trans_inv, **kwargs)     
    else : 
        qcnn = QCNN(num_qubits, num_measured, trans_inv, **kwargs) 
    
    #Load quantum circuit
    qcnn_circuit, meas_wires = qcnn.get_circuit() 
    
    dev = qml.device("default.qubit", wires = num_qubits) 
    
    
    if equiv : 
        # Use Equivariant QCNN
        @jax.jit 
        @qml.qnode(dev, interface = "jax") 
        def circuit(X, params) : 
            
            embed_image(X, np.array([i for i in range(num_qubits)])) 
            qcnn_circuit(params)
            for i in meas_wires : 
                qml.Hadamard(i)
            return qml.probs(wires = meas_wires[:len(meas_wires)//2]), qml.probs(wires = meas_wires[len(meas_wires)//2:])

        return circuit, qcnn._num_params 
    else : 
        @jax.jit 
        @qml.qnode(dev, interface = "jax") 
        def circuit(X, params) : 
            embed_image(X, np.array([i for i in range(num_qubits)])) 
            qcnn_circuit(params)
            
            return qml.probs(wires = meas_wires) 
            
        return circuit, qcnn._num_params
    
    


