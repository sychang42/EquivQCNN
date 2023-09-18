import sys
import os 
sys.path.append(os.path.dirname(__file__)) 


import jax 
import jax.numpy as jnp 

import flax 
import flax.linen as nn 
from flax.training.train_state import TrainState 

import pennylane as qml 


from typing import Tuple, List, Dict, Callable 

PRNGKey = jnp.ndarray 

class QCNNClassifier(nn.Module):
    """
    Image classifier with Equivariant Quantum Convolutional Neural Network
    
    Args : 
        num_wires: Number of wires in quantum circuit
        depth: Number of convolutional layers in qcnn
        
    """ 
    circuit: Callable
    num_params: int
    equiv: bool
    delta: float
    
    def init_params(self, 
                    rng: PRNGKey, 
                    num_qparams: int) -> jnp.ndarray : 
#         return jax.random.normal(rng, (num_qparams, ))

        """
        Function to initialize quantum circuit parameters
        
        Args : 
            rng (PRNGKey) : Random Number Generator Key
            num_qparams (int) : Number of quantum circuit parameters
        
        Return : 
            params (jnp.ndarray) : Initial parameters 
            
        """ 
        return jax.random.uniform(rng, (num_qparams, ), minval = -self.delta, maxval = self.delta)
        
    
    @nn.compact
    def __call__(self,  
             X_batch: jnp.ndarray) -> jnp.ndarray:      
                
        qparams = self.param('qparams', self.init_params, self.num_params)
        
        if self.equiv : 
            classifier_vmap = jax.vmap(lambda z : jnp.sum(self.circuit(z, qparams), axis = 0)/2.) 
        else : 
            classifier_vmap = jax.vmap(lambda z : self.circuit(z, qparams)) 
        class_outputs = classifier_vmap(X_batch) 
        
        return class_outputs

    

    