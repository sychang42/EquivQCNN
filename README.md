<!--Back to the top -->
<a name="readme-top"></a>


<div align="center">
<h3 align="center">Approximately Equivariant Quantum Neural Network for $p4m Group Symmetries in Images
</h3>
  <p align="center">
     Image classification using with an Equivariant Quantum Convolutional Neural Network based on the inherent symmetry of the input dataset. 
    <br />
    <a href="https://github.com/EquivQCNN/docs"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://arxiv.org/abs/2310.02323">Paper</a>
    .
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    ·
    <a href="https://github.com/sychang42/EquivQCNN/issues">Report Bug</a>
    ·
    <a href="https://github.com/sychang42/EquivQCNN/issues">Request Feature</a>
  </p>
</div>

## About the codes

This repository contains code for training an Equivariant Quantum Convolutional Neural Network (QCNN) under $p4m$ group symmetries for image classification tasks. The code is written in [Jax](https://github.com/google/jax) and utilizes [Pennylane](https://github.com/PennyLaneAI/pennylane) for the quantum operations. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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

## Cite

```
@misc{chang2023approximately,
      title={Approximately Equivariant Quantum Neural Network for $p4m$ Group Symmetries in Images}, 
      author={Su Yeon Chang and Michele Grossi and Bertrand Le Saux and Sofia Vallecorsa},
      year={2023},
      eprint={2310.02323},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```


## License

Distributed under the Apache License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Su Yeon Chang - [@SyChang97](https://twitter.com/SyChang97) - su.yeon.chang@cern.ch

Project Link: [https://github.com/sychang42/EquivQCNN](https://github.com/sychang42/EquivQCNN)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

