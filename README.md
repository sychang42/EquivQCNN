# Approximately Equivariant QCNN under $p4m$ Group Symmetries for Images 

This repository contains code for training an Equivariant Quantum Convolutional Neural Network (QCNN) under $p4m$ group symmetries for image classification tasks. The code is written in Jax[https://github.com/google/jax] and utilizes Pennylane[https://github.com/PennyLaneAI/pennylane] for the quantum operations. 

## Usage

To train the model, use the following command : 

```
python run.py --config config/training.yaml --gpu 0
```
 
The configuration file `configs/training.yaml` should be structured as follows : 

* __dataset_params__: 
  - *data* : Name of training dataset. 
  - *img_size* : Input image size.
  - *classes* : List of integers representing data classes to be trained. Currently, only binary classification is implemented.
  - *n_data* : Number of training samples. Set to `Null` to use the entire dataset.

* __training_params__: 
  - *num_epohcs* : Number of training epochs. 
  - *batchsize* : Training batch size
  - *loss_type* : Type of loss function used to train the model. Currently, only Binary Cross-Entropy (BCE) loss is implemented.
  
* __model_params__: 
  - *num_wires* : Number of qubits in the quantum classifier. 
  - *equiv* : Boolean to indicate whether an equivariant neural network is used. 
  - *trans_inv* : Boolean to indicate whether the model is constructed in a translational invariant way. 
  - *ver* :  Quantum circuit architecture version. 
  - *symmetry_breaking* : Boolean to indicate whether $RZ$ gates at the end of the quantum circuit in case of the *EquivQCNN*. 
  - *delta* : Range of uniform distribution from which the initial parameters are sampled.  
  
* __opt_params__: 
  - *lr* : Learning rate. 
  - *b1* : $\beta_1$ value of the Adam optimizer. 0.9 by default. 
  - *b2* : $\beta_2$ value of the Adam optimizer. 0.999 by default.

* __logging_params__: 
  - *save_dir* : Root directory to store the training results. If `Null`, the results will not stored.



