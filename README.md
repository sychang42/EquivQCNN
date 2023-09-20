# Approximately Equivariant QCNN under $p4m$ Group Symmetries for Images 

Codes for equivariant Quantum Convolutional Neural Network for $p4m$ group symmetries for image classification.


```
python run.py --config config/training.yaml --gpu 0
```
 
 
The config file should be structured as follows : 

*dataset_params: 
  -data : Name of training data. 
  -img_size : Input image size
  -classes : Data classes (list of integers) to be trained. Only binary classification implemented for the moment.
  -n_data : Number of training samples. `Null` to use the whole dataset

