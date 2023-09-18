import sys
import os 
sys.path.append(os.path.dirname(__file__)) 

import numpy as np

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
    hybrid: bool
    
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
#         return jax.random.normal(rng, (num_qparams, ))*self.delta
    
    @nn.compact
    def __call__(self,  
             X_batch: jnp.ndarray) -> jnp.ndarray:      
                
        qparams = self.param('qparams', self.init_params, self.num_params)
        
#         if self.sym_break > 0: 
#             sym_break_terms = self.param("epsilon", self, init_params, self.sym_break) 
#         if self.equiv : 
#             classifier_vmap = jax.vmap(lambda z : jnp.sum(self.circuit(z, qparams), axis = 0)/2.) 
#         else : 
#         classifier_vmap = jax.vmap(lambda z : self.circuit(z, qparams)) 
        
        if self.equiv : 
#             classifier_vmap = jax.vmap(lambda z : jnp.sum(self.circuit(z, qparams), axis = 0)/2.) 
#             class_outputs = classifier_vmap(X_batch) 
            cvmap1 = jax.vmap(lambda z : self.circuit[0](z, qparams))  
            cvmap2 = jax.vmap(lambda z : self.circuit[1](z, qparams)) 
        
#             class_outputs = jnp.concatenate([cvmap1(X_batch[:len(X_batch)//2]), cvmap2(X_batch[len(X_batch)//2:])])
            class_outputs = (cvmap1(X_batch) + cvmap2(X_batch))/2.
        else : 
            classifier_vmap = jax.vmap(lambda z : self.circuit(z, qparams)) 
            class_outputs = classifier_vmap(X_batch) 
            
        if self.hybrid : 
            class_outputs = nn.Dense(features = 2)(class_outputs)
            class_outputs = nn.softmax(class_outputs)
        
        
        return class_outputs

    

    