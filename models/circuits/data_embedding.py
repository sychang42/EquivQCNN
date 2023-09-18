"""Embedding classical image into quantum circuit"""

import pennylane as qml 
import jax.numpy as jnp

from typing import List


def embed_image(img : jnp.ndarray, 
               wires: List[int]) -> None: 
    """
    Embed classical image into quantum circuit in reflection and rotation equivariant way 
    with Coordinate-Aware Amplitude Embedding. 
    
    U(x)|0\ket = \sum_i=1^n x_i,j 
    
    Args: 
        img (jnp.ndarray) : Image to embed into quantum circuit
        wires (List[int]) : Ordered list of quantum circuit wires to embed the image. 
        
    Return : 
        None 
    """
    n = len(wires)//2
    if img.shape[2] == 1 : 
        img = img.reshape(img.shape[0], img.shape[1]) 
        
    # Rewrite the images as amplitudes of the input quantum state
    features = jnp.zeros(2**(2*n))
    
    for i in range(2**n) : 
        for j in range(2**n) :
            
            features = features.at[2**n*i + j].set(jnp.sin(jnp.pi/2*(2*img[i,j]-1)))
        
    features = features/jnp.sqrt(jnp.sum(features**2))   

    # Encode data with amplitude embedding
    qml.AmplitudeEmbedding(features, wires = wires)    
