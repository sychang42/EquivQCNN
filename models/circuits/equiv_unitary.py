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

def RXXYYZZ(phi1, phi2, phi3) : 
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    XYZ = phi1*np.kron(X, X) + phi2*np.kron(Y,Y) + phi3*np.kron(Z,Z)
    return np.diag(np.exp(-1j*XYZ))

class U2_ver1(qml.operation.Operation):
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
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.IsingZZ(angle[2], wires = wires)) 
        op_list.append(qml.RX(angle[3], wires = wires[0])) 
        op_list.append(qml.RX(angle[4], wires = wires[1])) 
        op_list.append(qml.IsingYY(angle[5], wires = wires)) 
        
        return op_list
    
class U2_ver1_2(qml.operation.Operation):
    num_wires = 2
    num_params = 8 #int: Number of trainable parameters that the operator depends on.

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
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.IsingZZ(angle[2], wires = wires)) 
        op_list.append(qml.RX(angle[3], wires = wires[0])) 
        op_list.append(qml.RX(angle[4], wires = wires[1])) 
        op_list.append(qml.IsingYY(angle[5], wires = wires)) 
        op_list.append(qml.RX(angle[6], wires = wires[0])) 
        op_list.append(qml.RX(angle[7], wires = wires[1])) 
        
        return op_list

class U2_ver1_3(qml.operation.Operation):
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
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.CNOT(wires = wires)) 
        op_list.append(qml.RX(angle[2], wires = wires[0])) 
        op_list.append(qml.RX(angle[3], wires = wires[1])) 
        op_list.append(qml.CNOT(wires = wires)) 
        op_list.append(qml.RX(angle[4], wires = wires[0])) 
        op_list.append(qml.RX(angle[5], wires = wires[1])) 
        
        return op_list

    
class U2_ver2(qml.operation.Operation):
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
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.IsingZZ(angle[2], wires = wires)) 
        
        return op_list

class U2_ver3(qml.operation.Operation):
    num_wires = 2
    num_params = 5 #int: Number of trainable parameters that the operator depends on.

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
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.DiagonalQubitUnitary(RXXYYZZ(angles[2], angles[3], angles[4]), wires = wires)) 
        
        return op_list

    
    
class U2_ver4(qml.operation.Operation):
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
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.IsingYY(angle[2], wires = wires)) 
        op_list.append(qml.IsingZZ(angle[3], wires = wires)) 
        
        return op_list     
    
class U2_ver3(qml.operation.Operation):
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
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.IsingYY(angle[2], wires = wires)) 
        
        return op_list     
    
    
# class U2_ver2(qml.operation.Operation):
#     num_wires = 2
#     num_params = 14 #int: Number of trainable parameters that the operator depends on.

#     ndim_params = tuple(0 for _ in range(num_params))
    
#     # This attribute tells PennyLane what differentiation method to use. Here
#     # we request parameter-shift (or "analytic") differentiation.
#     grad_method = "A"
    
#     def __init__(self, *phi, wires, do_queue=True, id=None):
#         super().__init__(*phi, wires=wires, do_queue=do_queue, id=id)
        
#     @staticmethod
#     def compute_decomposition(*angle, wires):  # pylint: disable=arguments-differ
#         # Overwriting this method defines the decomposition of the new gate, as it is
#         # called by Operator.decomposition().
#         # The general signature of this function is (*parameters, wires, **hyperparameters).
#         op_list = []
            
#         op_list.append(qml.RX(angle[0], wires = wires[0])) 
#         op_list.append(qml.RX(angle[1], wires = wires[1])) 
#         op_list.append(qml.RY(angle[2], wires = wires[0])) 
#         op_list.append(qml.RY(angle[3], wires = wires[1])) 
#         op_list.append(qml.RZ(angle[4], wires = wires[0])) 
#         op_list.append(qml.RZ(angle[5], wires = wires[1])) 
#         op_list.append(qml.IsingZZ(angle[6], wires = wires)) 
#         op_list.append(qml.RX(angle[7], wires = wires[0])) 
#         op_list.append(qml.RX(angle[8], wires = wires[1])) 
#         op_list.append(qml.RY(angle[9], wires = wires[0])) 
#         op_list.append(qml.RY(angle[10], wires = wires[1])) 
#         op_list.append(qml.RZ(angle[11], wires = wires[0])) 
#         op_list.append(qml.RZ(angle[12], wires = wires[1])) 
#         op_list.append(qml.IsingYY(angle[13], wires = wires)) 
        
#         return op_list
   
    
class U4_ver1(qml.operation.Operation):
    num_wires = 4
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
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.RX(angle[0], wires = wires[2])) 
        op_list.append(qml.RX(angle[1], wires = wires[3])) 
        op_list.append(qml.DiagonalQubitUnitary(IsingZ4(angle[2]), wires = wires)) 
        
        return op_list
    
class U4_ver1_2(qml.operation.Operation):
    num_wires = 4
    num_params = 5 #int: Number of trainable parameters that the operator depends on.

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
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.RX(angle[0], wires = wires[2])) 
        op_list.append(qml.RX(angle[1], wires = wires[3])) 
        op_list.append(qml.DiagonalQubitUnitary(IsingZ4(angle[2]), wires = wires)) 
        op_list.append(qml.RX(angle[3], wires = wires[0])) 
        op_list.append(qml.RX(angle[4], wires = wires[1])) 
        op_list.append(qml.RX(angle[3], wires = wires[2])) 
        op_list.append(qml.RX(angle[4], wires = wires[3])) 
        
        return op_list
    
    
class Pooling_ansatz1(qml.operation.Operation):
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
        def cond0(x, wires):
            qml.RX(x[0], wires)

        def cond1(x, wires):
            qml.RX(x[1], wires)
        
        op_list = []
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.cond(qml.measure(wires[1]), cond0, cond1)(angle[2:], wires = wires[0])) 
        return op_list    

    
    
class Pooling_ansatz2(qml.operation.Operation):
    num_wires = 2
    num_params = 5 #int: Number of trainable parameters that the operator depends on.

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
        def cond0(x, wires):
            qml.RX(x[0], wires)

        def cond1(x, wires):
            qml.RX(x[1], wires)
        
        op_list = []
            
        op_list.append(qml.RX(angle[0], wires = wires[0])) 
        op_list.append(qml.RX(angle[1], wires = wires[1])) 
        op_list.append(qml.RY(angle[2], wires = wires[1])) 
        op_list.append(qml.RZ(angle[3], wires = wires[1])) 
        op_list.append(qml.CRX(angle[4], wires = [wires[1], wires[0]]))
        return op_list    
    
class Pooling_ansatz4(qml.operation.Operation):
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
            
        op_list.append(qml.CRX(angle[0], wires=[wires[1], wires[0]])) 
        op_list.append(qml.PauliX(wires=wires[1])) 
        op_list.append(qml.CRX(angle[1], wires=[wires[1], wires[0]]))

        return op_list   
    
def conv2(angle, wires) : 
    
    for i in range(0, len(wires), 2):
        U2(angle, [wires[i], wires[i+1]])
    
    for i in range(1, len(wires)-1, 2):
        U2(angle, [wires[i], wires[i+1]])
    
    if len(wires) > 2 : 
        U2(angle, [wires[-1], wires[0]])
               
        
def conv4(angle, wires) : 
    
    for i in range(0, len(wires), 4):
        U4(angle, [wires[i + j] for j in range(4)])
    
    for i in range(2, len(wires)-2, 4):
        U4(angle, [wires[i + j] for j in range(4)])
    
    if len(wires) > 4 : 
        U4(angle, [wires[0], wires[1], wires[-2], wires[-1]])
        


#def U(angle : jnp.ndarray, 
#       wires: jnp.ndarray) -> None : 
#     """
#     Convolutional filter with RX and ZZZZ rotation 
    
#     """ 
#     for i in range(len(wires)) : 
#         qml.RX(angle[i%2], wires = wires[i])

#     qml.DiagonalQubitUnitary(IsingZ4(angle[2]), wires = wires)
    

# def conv_layers(angle : float, 
#       wires: jnp.ndarray) -> None : 
#     """
#     Convolutional layer
    
#     """ 
#     for i in range(0, len(wires), 4):
#         U(angle, [wires[i + j] for j in range(4)])
    
#     for i in range(2, len(wires)-2, 4):
#         U(angle, [wires[i + j] for j in range(4)])
    
#     if len(wires) > 4 : 
#         U(angle, [wires[0], wires[1], wires[-2], wires[-1]])
        

# def pooling(angle : jnp.ndarray, 
#       wires: jnp.ndarray) -> None : 
#     """
#     Pooling layer 
    
#     """ 
#     for i in range(0, len(wires), 2) :
#         qml.CRX(angle, wires = [wires[i+1], wires[i]])


# def pooling(angle: jnp.ndarray,
#             wires: jnp.ndarray) :
#     """
#     Pooling layer 

#     """     

#     for i in range(0, len(wires), 2) :
#         qml.RX(angle[0], wires = wires[i])
#         qml.RX(angle[1], wires = wires[i+1])
#         qml.RY(angle[2], wires = wires[i+1])
#         qml.RZ(angle[3], wires = wires[i+1])
        
#         m_1 = qml.measure(wires[i+1])
#         qml.cond(m_1 == 1, qml.X)(wires = wires[i])

# def conv2(angle, wires) : 
    
#     for i in range(0, len(wires), 2):
#         U2(angle[:6], [wires[i], wires[i+1]])
    
#     for i in range(1, len(wires)-1, 2):
#         U2(angle[6:12], [wires[i], wires[i+1]])
    
#     if len(wires) > 2 : 
#         U2(angle[12:18], [wires[-1], wires[0]])


