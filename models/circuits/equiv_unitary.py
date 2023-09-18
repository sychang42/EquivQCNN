import jax.numpy as jnp
import pennylane as qml

def IsingZ4(phi : float) -> float: 
    """
    ZZZZ rotation. U(\phi) = (e^(i*phi*Z^\otimes4)) 
    
    Argument : 
        phi (float) : Rotation angle 
        
    Return : 
        Diagonal matrix for IsingZ4 rotation 
        
    """ 
    Z = jnp.array([[1, 0], [0, -1]])
    Z4 = jnp.kron(Z, jnp.kron(Z, jnp.kron(Z, Z)))
    
    Z4_diag = jnp.diag(jnp.exp(1j*phi*Z4))
    
    return Z4_diag

class equiv_U2(qml.operation.Operation):
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
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.IsingZZ(angle[2], wires = wires)) 
        op_list.append(qml.RX(angle[3], wires = wires[0])) 
        op_list.append(qml.RX(angle[4], wires = wires[1])) 
        op_list.append(qml.IsingYY(angle[5], wires = wires)) 
        
        return op_list
   
    
class equiv_U4(qml.operation.Operation):
    num_wires = 4
    num_params = 3 # Number of trainable parameters.

    ndim_params = tuple(0 for _ in range(num_params))
    

    grad_method = "A" # Analytic Differentiation method to use. 
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
    @staticmethod
    def compute_decomposition(*angle, wires):  
        # Defines the decomposition of the new gate (called by Operator.decomposition())
        op_list = []
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.RX(angle[0], wires = wires[2])) 
        op_list.append(qml.RX(angle[1], wires = wires[3])) 
        op_list.append(qml.DiagonalQubitUnitary(IsingZ4(angle[2]), wires = wires)) 
        
        return op_list
    
        
class Pooling_ansatz(qml.operation.Operation):
    num_wires = 2
    num_params = 5 # Number of trainable parameters. 

    ndim_params = tuple(0 for _ in range(num_params))
    
    grad_method = "A" # Analytic Differentiation method to use. 
    
    def __init__(self, *phi, wires, do_queue=True, id=None):
        super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
    
    @staticmethod
    def compute_decomposition(*angle, wires):  
        # Defines the decomposition of the new gate (called by Operator.decomposition())
        op_list = []
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.RY(angle[2], wires = wires[1])) 
        op_list.append(qml.RZ(angle[3], wires = wires[1])) 
        op_list.append(qml.CRX(angle[4], wires = [wires[1], wires[0]]))
        return op_list    
    