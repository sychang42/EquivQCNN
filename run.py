"""
Run Autoencoder
"""

import os 
import yaml 
import argparse

from dataset import *
from utils import *

from models.train import train_model

import jax
from jax.config import config
config.update("jax_enable_x64", True)

from math import ceil

if __name__ == "__main__":
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".50"
    os.environ["XLA_GPU_STRICT_CONV_ALGORITHM_PICKER"]="false"

    # Retrieve argument from yaml file
    parser = argparse.ArgumentParser(
        description="Run qGAN experiments on simulator")
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/training.yaml')

    parser.add_argument('--gpu',  '-g',
                        dest = "gpu_num",
                        help='GPU number',
                        default="2")
    
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            
    # Model parameters 
    config['model_params']['num_measured'] = int(ceil(jnp.log2(len(config['dataset_params']['classes']))))
            
    if config['model_params']['equiv'] : 
        config['model_params']['num_measured'] = config['model_params']['num_measured']*2
    
    # Load MNIST data
    load_dir = "/data/suchang/shared/Data/"
    X_train, Y_train, X_test, Y_test = get_data(config['dataset_params']['data'], 
                                                load_dir, config['dataset_params']['img_size'],
                                                config['dataset_params']['classes'])
    
    if config['dataset_params']['data'] == "Ising" : 
        X_train = (X_train + 1) /2.0 
        X_test = (X_test + 1) / 2.0
    
    seed = np.random.randint(1000) 
    config['training_params']['seed'] = seed 
    
    save_dir = config['logging_params']['save_dir']
    snapshot_dir = None
    if save_dir is not None : 
        snapshot_dir = initialize_logging(save_dir, config) 
      
    num_classes = jnp.max(Y_train) + 1
    n_data = len(X_train) 
    if 'n_data' in config['dataset_params'] and config['dataset_params']['n_data'] is not None : 
        n_data = config['dataset_params']['n_data']
        
#     train_ds = {"image": X_train, "label" : jax.nn.one_hot(Y_train, num_classes)} 
#     test_ds = {"image": X_test, "label" : jax.nn.one_hot(Y_test, num_classes)} 
    train_ds = {"image": X_train[:n_data], "label" :Y_train[:n_data]} 
    test_ds = {"image": X_test, "label" : Y_test} 
    
    
    # Train 
    train_model(train_ds, test_ds, config['training_params'], config['model_params'], config['opt_params'], seed, snapshot_dir)
