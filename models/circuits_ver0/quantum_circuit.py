"""
Construct Quantum circuit for generator
"""

import os
import sys
sys.path.append(os.path.dirname(__file__)) 

import pennylane as qml 

import numpy as np

import jax 
from jax import numpy as jnp 

from typing import Tuple, Callable, Optional, List, Any, Dict, Union

from qcnn import *
from data_embedding import * 

def get_quantum_circuit(num_qubits : int, 
                         num_measured : int, 
                          equiv : Optional[bool] = False,
                         trans_inv: Optional[bool] = True, 
                        **kwargs
                         ) -> Tuple[Callable, int]:
    """
    
    Args : 
        num_qubits (int) : Number of qubits in the quantum generator.
        qnn_config (Union[str, Dict[str, Any]]) : Quantum Circuit configuration for the learning layers.
        style_based Optional[bool] : Boolean to indicate whether we use style based-architecture. 
        num_final_qubits (int) : Number of qubits at the end of the circuit in case we use pooling layers in the quantum circuit. 
                            num_qubits > num_final_qubits. 
    Return : 
        circuit (Callable) : Quantum generator circuit.
        total_num_params (int) : Total number of parameters required for the generator. 
    """
    
    if equiv: 
        qcnn = EquivQCNN(num_qubits, num_measured, trans_inv, **kwargs)     
    else : 
        qcnn = QCNN(num_qubits, num_measured, trans_inv, **kwargs) 
    
    #Load quantum circuit
    qcnn_circuit, meas_wires = qcnn.get_circuit() 
    
    dev = qml.device("default.qubit", wires = num_qubits) 
    
    
    # In case we use style-based architecture
    if equiv : 
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
    
    


