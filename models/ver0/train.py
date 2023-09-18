import sys
import os 
sys.path.append(os.path.dirname(__file__)) 

import jax 
import jax.numpy as jnp 

import numpy as np 
from metrics import get_metrics 


from typing import Tuple, Optional, List, Dict

import csv
from tqdm import tqdm
from time import time 

from .utils import *

from functools import partial

@jax.jit
def compute_metrics(pred, labels):
    """ 
    """
    loss_type = ['BCE_loss', 'accuracy']
    
    loss = 0.
    losses = {}
    for l in loss_type : 
        losses[l] = get_metrics(l)(labels, pred)   
        if "loss" in l : 
            loss += losses[l] 
   
        
    return loss, losses


@jax.jit
def train_batch(x_batch : jnp.ndarray,
                y_batch : jnp.ndarray,
               model_state: TrainState, 
               ) -> Tuple[TrainState, Dict]:
    
    def loss_fn(params) : 
        class_outputs = model_state.apply_fn(
                {'params': params}, x_batch)
        loss, losses = compute_metrics(class_outputs, y_batch) 

        return loss + jnp.sqrt(jnp.sum(params**2)), (losses, class_outputs)

    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, (losses, class_outputs)), grads = grad_fn(model_state.params)
    
    # Update the generator parameters.
    new_model_state = model_state.apply_gradients(
      grads=grads)

    return new_model_state, losses, class_outputs

@jax.jit
def validate(x_batch: jnp.ndarray, 
             y_batch: jnp.ndarray, 
            model_state : TrainState) -> Tuple[Dict, jnp.ndarray]:
    
    class_outputs = model_state.apply_fn(
                {'params': model_state.params}, x_batch)
        
    _, losses = compute_metrics(class_outputs, y_batch) 
    if class_outputs.shape[1] == 1 : 
        preds = {"preds": class_outputs}
    else : 
        preds = {"preds" : jnp.argmax(class_outputs, axis = 1)}
    return losses, preds


def train_model(train_ds : jnp.ndarray,
               test_ds : jnp.ndarray,
                train_args : Dict,
                model_args : Dict, 
                opt_args : Dict, 
                snapshot_dir : Optional[str] = None) -> None :
    
    epochs = train_args['num_epochs']
    batch_size = train_args['batchsize'] 
    loss_type = train_args['loss_type']
    
    print(train_ds['image'].shape) 
    print(test_ds['image'].shape)     
    # Image shape    
    im_shape = train_ds['image'].shape
    label_shape = train_ds['label'].shape
    
    seed = np.random.randint(1000)
    
    key = jax.random.PRNGKey(seed=seed)
    key, init_key = jax.random.split(key)
    
    input_shape = (batch_size, im_shape[1], im_shape[2], im_shape[3])
    model_state, key = init_trainstate(model_args, opt_args, input_shape, key)
    
    # If we store the results
    fieldnames = ['epoch']
    fieldnames.extend([k + "_train" for k in loss_type]) 
    fieldnames.extend([k + "_test" for k in loss_type]) 
    fieldnames.append('time_taken')
    
        
    if snapshot_dir is not None :                 
        with open(os.path.join(snapshot_dir, 'output.csv'), mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
    
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size//batch_size

    
    # Train autoencoder 
    for epoch in tqdm(range(1, epochs + 1), desc="Epoch...",
                    position=0, leave=True):
        
        
        train_loss_epoch = {k : [] for k in loss_type}
        
        start_epoch = time() 
        
        key, init_key = jax.random.split(key)
        perms = jax.random.permutation(key, train_ds_size)
        perms = perms[:steps_per_epoch*batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))

        
        with tqdm(total=steps_per_epoch, desc="Training...",
                leave=False) as progress_bar_train:
                
            for b in range(steps_per_epoch):
                batch_data = {k : train_ds[k][perms[b]] for k in train_ds.keys()} 
                        
                # Update Model. 
                model_state, loss,class_outputs = train_batch(batch_data['image'], batch_data['label'], model_state)        
                for k,v in loss.items() : 
                    train_loss_epoch[k].append(v)

                progress_bar_train.update(1)
                
            del batch_data 
            
            test_batch_num = len(test_ds['image'])//batch_size
            
            valid_losses = {k : [] for k in loss_type}
            valid_outputs = []
            
            for j in range(test_batch_num) :   
                batch_data = {k : test_ds[k][j*batch_size : (j+1)*batch_size] for k in test_ds.keys()} 
                valid_loss, outputs = validate(batch_data['image'], batch_data['label'], model_state)
                for k,v in valid_loss.items() : 
                    valid_losses[k].append(v)
                valid_outputs.append(outputs) 
                
            batch_data = {k : test_ds[k][test_batch_num*batch_size:] for k in test_ds.keys()} 
            valid_loss, outputs = validate(batch_data['image'], batch_data['label'] , model_state)
            for k,v in valid_loss.items() : 
                valid_losses[k].append(v)
            valid_outputs.append(outputs) 
                
                
            valid_loss = {k : jnp.mean(jnp.array(v)) for k, v in valid_losses.items()}        
            train_loss = {k : jnp.mean(jnp.array(v)) for k, v in train_loss_epoch.items()}   
            outputs = {k: valid_outputs[0][k] for k in valid_outputs[0].keys()}
            for output in valid_outputs[1:] : 
                for k, v in output.items() : 
                    outputs[k] = jnp.concatenate((outputs[k], v), axis = 0)
                
            print_losses(epoch, epochs, train_loss, valid_loss)

            # Save results.
            if snapshot_dir is not None :
                # Store output
                with open(os.path.join(snapshot_dir, 'output.csv'), mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                    to_write = {'epoch' : epoch} 
                    for k, v in train_loss.items()  : 
                        to_write[k + '_train'] = np.mean(np.array(v)) 
                        
                    for k, v in valid_loss.items()  : 
                        to_write[k + '_test'] = v
                        
                    to_write['time_taken'] = (time() - start_epoch)/60.0

                    writer.writerow(to_write) 
                    
                
                with open(os.path.join(snapshot_dir, "train_parameters.txt"), mode = "a") as f : 
                    for x in model_state.params['qparams'] : 
                        f.write(str(x) + " ") 
                    f.write("\n") 
                if epoch == 1 or epoch%10 == 0 :

                    save_outputs(epoch, snapshot_dir, outputs, test_ds['label'])
