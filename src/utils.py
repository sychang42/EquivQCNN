"""
Util functions 
"""
import os 
import json

import numpy as np

from typing import Optional 


def initialize_logging(save_dir, config) :     
    # Create folder to save training results
    path_count = 1
    snapshot_dir = os.path.join(save_dir, 'run' + str(path_count))
    while os.path.exists(snapshot_dir):
        path_count += 1
        snapshot_dir = os.path.join(save_dir, 'run' + str(path_count))

    os.makedirs(snapshot_dir)
    
    
    file_path = os.path.join(snapshot_dir, "summary.json")
    snapshot_file = open(file_path, "w")
    json.dump(config, snapshot_file)
    snapshot_file.close()
        
    return snapshot_dir





