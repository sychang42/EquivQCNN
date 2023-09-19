"""
This module contains the set of unitary circuit ansatz used to benchmark the performances of 
non-equivariant Quantum Convolutional Neural Network (QCNN) 
""" 

import pennylane as qml
   
class U_TTN(qml.operation.Operation):
    num_wires = 2
    num_params = 2 # Number of trainable parameters.

    ndim_params = tuple(0 for _ in range(num_params))
    

    grad_method = "A" # Analytic Differentiation method to use. 
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires): 
        # Defines the decomposition of the new gate (called by Operator.decomposition())
        op_list = []
            
        op_list.append(qml.RY(angle[0], wires=wires[0])) 
        op_list.append(qml.RY(angle[1], wires=wires[1])) 
        op_list.append(qml.CNOT(wires=[wires[0], wires[1]]))
        
        return op_list

    
class U_6(qml.operation.Operation):
    num_wires = 2
    num_params = 10 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    grad_method = "A" # Analytic Differentiation method to use. 
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires): 
        # Defines the decomposition of the new gate (called by Operator.decomposition())
        op_list = []
            
        op_list.append(qml.RX(angle[0], wires=wires[0])) 
        op_list.append(qml.RX(angle[1], wires=wires[1])) 
        op_list.append(qml.RZ(angle[2], wires=wires[0]))
        op_list.append(qml.RZ(angle[3], wires=wires[1]))
        op_list.append(qml.CRX(angle[4], wires=[wires[1], wires[0]]))
        op_list.append(qml.CRX(angle[5], wires=[wires[0], wires[1]]))
        op_list.append(qml.RX(angle[6], wires=wires[0]))
        op_list.append(qml.RX(angle[7], wires=wires[1]))
        op_list.append(qml.RZ(angle[8], wires=wires[0]))
        op_list.append(qml.RZ(angle[9], wires=wires[1]))
        
    
class U_SO4(qml.operation.Operation):
    num_wires = 2
    num_params = 6 # Number of trainable parameters.

    ndim_params = tuple(0 for _ in range(num_params))
    
    grad_method = "A" # Analytic Differentiation method to use. 
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires): 
        # Defines the decomposition of the new gate (called by Operator.decomposition())
        op_list = []
            
        op_list.append(qml.RY(angle[0], wires=wires[0])) 
        op_list.append(qml.RY(angle[1], wires=wires[1])) 
        op_list.append(qml.CNOT(wires=[wires[0], wires[1]]))
        op_list.append(qml.RY(angle[2], wires=wires[0]))
        op_list.append(qml.RY(angle[3], wires=wires[1]))
        op_list.append(qml.CNOT(wires=[wires[0], wires[1]]))
        op_list.append(qml.RY(angle[4], wires=wires[0])) 
        op_list.append(qml.RY(angle[5], wires=wires[1])) 

        return op_list
    
    
class Pooling_ansatz(qml.operation.Operation):
    num_wires = 2
    num_params = 2 # Number of trainable parameters.

    ndim_params = tuple(0 for _ in range(num_params))
    
  
    grad_method = "A" # Analytic Differentiation method to use. 
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  
        # Defines the decomposition of the new gate (called by Operator.decomposition())
        op_list = []
            
        op_list.append(qml.CRZ(angle[0], wires=[wires[1], wires[0]])) 
        op_list.append(qml.PauliX(wires=wires[1])) 
        op_list.append(qml.CRX(angle[1], wires=[wires[1], wires[0]]))

        return op_list
       
