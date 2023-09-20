import sys
import os 
sys.path.append(os.path.dirname(__file__)) 


import jax 
import jax.numpy as jnp 

import flax.linen as nn 
from flax.training.train_state import TrainState 

import pennylane as qml 


from typing import Tuple, List, Dict, Callable 

PRNGKey = jnp.ndarray 

class QCNNClassifier(nn.Module):
    """
    Image classifier with Equivariant Quantum Convolutional Neural Network
    
    Args : 
        circuit (Callable) : Quantum Circuit. 
        num_params (int) : Number of trainable parameters. 
        equiv (Bool) : Boolean to indicate whether an equivariant neural network is used.
        delta (float) : Range of uniform distribution from which the initial parameters are sampled.  
        
    """ 
    circuit: Callable
    num_params: int
    equiv: bool
    delta: float
    # hybrid: bool (Boolean to indicate whether we add a classical layer at the end. Deprecated.)
        
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

        # Uniform initialization of initial weights
        return jax.random.uniform(rng, (num_qparams, ), minval = -self.delta, maxval = self.delta)
#         return jax.random.normal(rng, (num_qparams, ))*self.delta
    
    @nn.compact
    def __call__(self,  
             X_batch: jnp.ndarray) -> jnp.ndarray:      
        """
        Forward function to return classifier output. 
        
        Args : 
            X_batch (jnp.ndarray) : Classical input data of shape (batch_size, img_size[0], img_size[1], img_size[2])
        
        Return : 
            class_outputs (jnp.ndarray) : Quantum Classifier output of shape (batch_size, ceil(log2(n_class)))
            
        """         
        qparams = self.param('qparams', self.init_params, self.num_params)
        
        if self.equiv : 
            classifier_vmap = jax.vmap(lambda z : jnp.sum(jnp.array(self.circuit(z, qparams)), axis = 0)/2.) 
        else : 
            classifier_vmap = jax.vmap(lambda z : self.circuit(z, qparams)) 
            
        class_outputs = classifier_vmap(X_batch) 

        # if self.hybrid : 
        #     class_outputs = nn.Dense(features = 2)(class_outputs)
        #     class_outputs = nn.softmax(class_outputs)
            
        return class_outputs

    

    