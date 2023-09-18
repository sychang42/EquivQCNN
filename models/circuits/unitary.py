# This module contains the set of unitary ansatze that will be used to benchmark the performances of Quantum Convolutional Neural Network (QCNN) in QCNN.ipynb module
import pennylane as qml
   
class U_TTN(qml.operation.Operation):
    num_wires = 2
    num_params = 2 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        op_list = []
            
        op_list.append(qml.RY(angle[0], wires=wires[0])) 
        op_list.append(qml.RY(angle[1], wires=wires[1])) 
        op_list.append(qml.CNOT(wires=[wires[0], wires[1]]))
        
        return op_list

class U_5(qml.operation.Operation):
    num_wires = 2
    num_params = 10 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        op_list = []
            
        op_list.append(qml.RX(angle[0], wires=wires[0])) 
        op_list.append(qml.RX(angle[1], wires=wires[1])) 
        op_list.append(qml.RZ(angle[2], wires=wires[0]))
        op_list.append(qml.RZ(angle[3], wires=wires[1]))
        op_list.append(qml.CRZ(angle[4], wires=[wires[1], wires[0]]))
        op_list.append(qml.CRZ(angle[5], wires=[wires[0], wires[1]]))
        op_list.append(qml.RX(angle[6], wires=wires[0]))
        op_list.append(qml.RX(angle[7], wires=wires[1]))
        op_list.append(qml.RZ(angle[8], wires=wires[0]))
        op_list.append(qml.RZ(angle[9], wires=wires[1]))
        
        return op_list
    
class U_6(qml.operation.Operation):
    num_wires = 2
    num_params = 10 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
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
        
        
        return op_list
    
    

class U_9(qml.operation.Operation):
    num_wires = 2
    num_params = 2 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        op_list = []
            
        op_list.append(qml.Hadamard(wires=wires[0])) 
        op_list.append(qml.Hadamard(wires=wires[1])) 
        op_list.append(qml.CZ(wires=[wires[0], wires[1]]))
        op_list.append(qml.RX(angle[0], wires=wires[0]))
        op_list.append(qml.RX(angle[1], wires=wires[1]))

        return op_list

class U_13(qml.operation.Operation):
    num_wires = 2
    num_params = 6 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        op_list = []
            
        op_list.append(qml.RY(angle[0], wires=wires[0])) 
        op_list.append(qml.RY(angle[1], wires=wires[1])) 
        op_list.append(qml.CRZ(angle[2], wires=[wires[1], wires[0]]))
        op_list.append(qml.RY(angle[2], wires=wires[0]))
        op_list.append(qml.RY(angle[3], wires=wires[1]))
        op_list.append(qml.CRZ(angle[5], wires=[wires[0], wires[1]]))

        return op_list


class U_14(qml.operation.Operation):
    num_wires = 2
    num_params = 6 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        op_list = []
            
        op_list.append(qml.RY(angle[0], wires=wires[0])) 
        op_list.append(qml.RY(angle[1], wires=wires[1])) 
        op_list.append(qml.CRX(angle[2], wires=[wires[1], wires[0]]))
        op_list.append(qml.RY(angle[2], wires=wires[0]))
        op_list.append(qml.RY(angle[3], wires=wires[1]))
        op_list.append(qml.CRX(angle[5], wires=[wires[0], wires[1]]))

        return op_list
    
    
class U_15(qml.operation.Operation):
    num_wires = 2
    num_params = 4 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        op_list = []
            
        op_list.append(qml.RY(angle[0], wires=wires[0])) 
        op_list.append(qml.RY(angle[1], wires=wires[1])) 
        op_list.append(qml.CNOT(wires=[wires[1], wires[0]]))
        op_list.append(qml.RY(angle[2], wires=wires[0]))
        op_list.append(qml.RY(angle[3], wires=wires[1]))
        op_list.append(qml.CNOT(wires=[wires[0], wires[1]]))

        return op_list

    
class U_SO4(qml.operation.Operation):
    num_wires = 2
    num_params = 6 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
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
    
class U_SU4(qml.operation.Operation):
    num_wires = 2
    num_params = 15 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        op_list = []
            
        op_list.append(qml.U3(angle[0], angle[1], angle[2], wires=wires[0])) 
        op_list.append(qml.U3(angle[3], angle[4], angle[5], wires=wires[1])) 
        op_list.append(qml.CNOT(wires=[wires[0], wires[1]]))
        op_list.append(qml.RY(angle[6], wires=wires[0]))
        op_list.append(qml.RZ(angle[7], wires=wires[1]))
        op_list.append(qml.CNOT(wires=[wires[1], wires[0]]))
        op_list.append(qml.RY(angle[8], wires=wires[0])) 
        op_list.append(qml.CNOT(wires=[wires[0], wires[1]])) 
        op_list.append(qml.U3(angle[9], angle[10], angle[11], wires=wires[0])) 
        op_list.append(qml.U3(angle[12], angle[13], angle[14], wires=wires[1]))

        return op_list
    
class Pooling_ansatz1(qml.operation.Operation):
    num_wires = 2
    num_params = 2 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        op_list = []
            
        op_list.append(qml.CRZ(angle[0], wires=[wires[1], wires[0]])) 
        op_list.append(qml.PauliX(wires=wires[1])) 
        op_list.append(qml.CRX(angle[1], wires=[wires[1], wires[0]]))

        return op_list
       
class Pooling_ansatz2(qml.operation.Operation):
    num_wires = 2
    num_params = 0 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        op_list = []
            
        op_list.append(qml.CZ(wires=[wires[0], wires[1]])) 

        return op_list
    
    
class Pooling_ansatz3(qml.operation.Operation):
    num_wires = 2
    num_params = 3 #int: Number of trainable parameters that the operator depends on.

    ndim_params = tuple(0 for _ in range(num_params))
    
    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        op_list = []
            
        op_list.append(qml.CRot(*angle, wires=[wires[0], wires[1]])) 

        return op_list
    
        

