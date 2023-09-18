import sys 
import os
sys.path.append(os.path.dirname(__file__)) 

import json

import jax.numpy as jnp 

import numpy as np
import pennylane as qml 
import unitary
import equiv_unitary


class QCNN() : 
    _valid_gates = {
        "RZ" : qml.RZ,
        "U_TTN" : unitary.U_TTN,
        "U_5" : unitary.U_5,
        "U_6" : unitary.U_6,
        "U_9" : unitary.U_9,
        "U_13" : unitary.U_13,
        "U_14" : unitary.U_14,
        "U_15" : unitary.U_15,
        "U_SO4" : unitary.U_SO4,
        "U_SU4" : unitary.U_SU4,
        "Pooling_ansatz1" : unitary.Pooling_ansatz1
    } 
    
    qnn_config_path = os.path.join(os.path.dirname(__file__), "qnn_architecture.json")
    
    
    def __init__(self, 
                 num_qubits: int,
                 num_measured: int,
                 trans_inv: bool = True,
                 **kwargs
                ) : 
        
        qnn_architecture = {'conv_filters' : ['U_TTN'],
                            'pooling' : 'Pooling_ansatz1'} 
        if "qnn_ver" in kwargs.keys() : 
            qnn_architecture = json.load(open(self.qnn_config_path))[kwargs['qnn_ver']]
        else : 
            for k in kwargs.keys() : 
                qnn_architecture[k] = kwargs[k]
         
        self._qnn_architecture = qnn_architecture
        self._num_qubits = num_qubits
        self._num_measured = num_measured
        
        self._rotation_gates = []
        if 'conv_filters' in qnn_architecture.keys() : 
            self._conv_filters = [self._choose_gate(gate) for gate in qnn_architecture['conv_filters']]
            
        self._pooling = []
        if 'pooling' in qnn_architecture.keys() : 
            self._pooling = self._choose_gate(qnn_architecture['pooling']) 
        
        depth = int(np.log2(num_qubits//num_measured))
        self._meas_wires = np.array([i for i in range(num_qubits)]) 
        while len(self._meas_wires) > num_measured : 
            self._meas_wires = np.array([self._meas_wires[i] for i in range(0, len(self._meas_wires), 2)])
            
        self._num_params = depth*(sum([gate[2] for gate in self._conv_filters]) + self._pooling[2])  
    
    
    @staticmethod
    def _choose_gate(gate_str : str) : 
        gate = QCNN._valid_gates.get(gate_str, None) 
        
        if gate is None : 
            raise NotImplementedError("Unknown gate.")  
            
        return (gate_str, gate, gate.num_params, gate.num_wires)
                            
        
    def _get_circuit(self, num_qubits, num_measured, conv_filters, pooling) : 

        def circuit(params) : 
            idx = 0
            
            wires = np.array([i for i in range(num_qubits)])
            
            while len(wires) > num_measured : 
                for _, gate, num_params, gate_num_wires in conv_filters : 
                    for i in range(0, len(wires), 2) : 
                        gate(*params[idx : idx + num_params], wires = [wires[i], wires[i+1]])
                    for i in range(1, len(wires)-1, 2) : 
                        gate(*params[idx : idx + num_params], wires = [wires[i], wires[i+1]])

                    gate(*params[idx : idx + num_params], wires = [wires[-1], wires[0]])
                
                    idx = idx + num_params
                    
                _, gate, num_params, gate_num_wires = pooling
                for i in range(0, len(wires), 2) : 
                    gate(*params[idx : idx + num_params], wires = [wires[i], wires[i+1]]) 
                idx = idx + num_params    
                wires = np.array([wires[i] for i in range(0, len(wires), 2)]) 
                        
            
        return circuit
    
    def get_circuit(self) :
                
        return self._get_circuit(self._num_qubits, self._num_measured, self._conv_filters,
                                 self._pooling), self._meas_wires
    
    def __str__(self) : 
        disp = f"Quantum Convolutional Neural Network with architecture :\n" 
        disp += f"  - Convolutional filters : {[gate[0] for gate in self._conv_filters]}\n"
        disp += f"  - Pooling : {self._pooling[0]}\n"
        
        return disp
    

    
class EquivQCNN() : 
    _valid_gates = {
        "U2_ver1" : equiv_unitary.U2_ver1,
        "U2_ver2ZZ" : equiv_unitary.U2_ver2ZZ,
        "U2_ver2YY" : equiv_unitary.U2_ver2YY,
        "U2_ver2YYZZ" : equiv_unitary.U2_ver2YYZZ,
        "U2_ver3" : equiv_unitary.U2_ver3,
        "U4_ver1" : equiv_unitary.U4_ver1,
        "equiv_Pooling_ansatz1" : equiv_unitary.Pooling_ansatz1,
        "equiv_Pooling_ansatz2" : equiv_unitary.Pooling_ansatz2, 
        "equiv_Pooling_ansatz4" : equiv_unitary.Pooling_ansatz4
    } 
    _valid_gates.update(QCNN._valid_gates) 
    
    qnn_config_path = os.path.join(os.path.dirname(__file__), "equiv_qnn_architecture.json")
    
    ccd .
    def __init__(self, 
                 num_qubits: int,
                 num_measured: int,
                 trans_inv: bool,
                 **kwargs
                ) : 
        
        qnn_architecture = {'U2_conv_filters' : ['U2_ver1'],
                            'U4_conv_filters' : ['U4_ver1'],
                            'pooling' : 'equiv_Pooling_ansatz1', 
                             'alternating' : False} 
        if "qnn_ver" in kwargs.keys() : 
            qnn_architecture = json.load(open(self.qnn_config_path))[kwargs['qnn_ver']]
        else : 
            for k in kwargs.keys() : 
                qnn_architecture[k] = kwargs[k]
         
        self._qnn_architecture = qnn_architecture
        self._num_qubits = num_qubits
        self._num_measured = num_measured
        self._trans_inv = trans_inv
        
        self._rotation_gates = []
        if 'U2_conv_filters' in qnn_architecture.keys() : 
            self._U2_conv_filters = [self._choose_gate(gate) for gate in qnn_architecture['U2_conv_filters']]
         
        if 'U4_conv_filters' in qnn_architecture.keys() : 
            self._U4_conv_filters = [self._choose_gate(gate) for gate in qnn_architecture['U4_conv_filters']]
            
        self._pooling = []
        if 'pooling' in qnn_architecture.keys() : 
            self._pooling = self._choose_gate(qnn_architecture['pooling']) 
            
        self._noisy_filters = []
        if 'noisy_filters' in qnn_architecture.keys() : 
            self._noisy_filters = [self._choose_gate(gate) for gate in qnn_architecture['noisy_filters']]
            
            
        if 'alternating' in qnn_architecture.keys() : 
            self._alternating = qnn_architecture['alternating']
            
        depth = int(np.log2(num_qubits//num_measured))
        self._meas_wires = np.array([i for i in range(num_qubits)]) 
        while len(self._meas_wires) > num_measured : 
            self._meas_wires = np.array([self._meas_wires[i] for i in range(0, len(self._meas_wires), 2)])
        
        num_gates = 1
        if not trans_inv : 
            num_gates = num_qubits//2 

        if not self._alternating :  
            self._num_params = num_gates*(sum([gate[2] for gate in self._U2_conv_filters]) + \
                                        depth*(sum([gate[2] for gate in self._U4_conv_filters]) + \
                                               sum([gate[2] for gate in self._noisy_filters]) + \
                                               self._pooling[2]))  
        else : 
            self._num_params = num_gates*depth*(sum([gate[2] for gate in self._U2_conv_filters]) + \
                                      sum([gate[2] for gate in self._U4_conv_filters]) + \
                                                sum([gate[2] for gate in self._noisy_filters]) + \
                                                self._pooling[2])
        # Symmetry breaking
        self._num_params += 1 
    @staticmethod
    def _choose_gate(gate_str : str) : 
        gate = EquivQCNN._valid_gates.get(gate_str, None) 
        
        if gate is None : 
            raise NotImplementedError("Unknown gate.")  
            
        return (gate_str, gate, gate.num_params, gate.num_wires)
                            
    
    def _U2_conv(self, params, gate, wires) : 
        idx  = 0
        for i in range(0, len(wires), 2) : 
            gate(*params[idx], wires = [wires[i], wires[i+1]])
            idx = idx + 1
            
        for i in range(1, len(wires)-1, 2) : 
            gate(*params[idx], wires = [wires[i], wires[i+1]])
            idx = idx + 1
            
        gate(*params[idx], wires = [wires[-1], wires[0]])    
    
                                          
    def _U4_conv(self, params, gate, wires) : 
        idx = 0
        for i in range(0, len(wires), 4):
            gate(*params[idx], wires = [wires[i + j] for j in range(4)])
            idx = idx + 1
            
        for i in range(2, len(wires)-2, 4):
            gate(*params[idx], wires = [wires[i + j] for j in range(4)])
            idx = idx + 1
            
        if len(wires) > 4 : 
            gate(*params[idx], wires = [wires[0], wires[1], wires[-2], wires[-1]])
                                          
                                                     
    def _get_circuit(self, num_qubits, num_measured, U2_conv_filters, U4_conv_filters, pooling, noisy_filters, alternating, trans_inv) : 

        def circuit(params) : 
            idx = 0
            
            wires = np.array([i for i in range(num_qubits)])
            
            if not alternating : 
                for _, gate, num_params, gate_num_wires in U2_conv_filters : 
                    num_gates = len(wires)//2
                    
                    if trans_inv : 
                        conv_params = jnp.repeat(jnp.array([params[idx : idx + num_params]]), num_gates, axis = 0)   
                        num_gates = 1
                    else : 
                        
                        conv_params = params[idx : idx + num_gates*num_params].reshape((num_gates, num_params))  
                        
                    self._U2_conv(conv_params, gate, wires[:len(wires)//2])
                    self._U2_conv(conv_params, gate, wires[len(wires)//2:])
                    
                    idx = idx + num_gates*num_params
            
            while len(wires) > num_measured : 
                if alternating : 
                    for _, gate, num_params, gate_num_wires in U2_conv_filters : 
                        num_gates = len(wires)//2
                    
                        if trans_inv : 
                            conv_params = jnp.repeat(jnp.array([params[idx : idx + num_params]]), num_gates, axis = 0)   
                            num_gates = 1
                        else : 
                            conv_params = params[idx : idx + num_gates*num_params].reshape((num_gates, num_params))  
                        
                        self._U2_conv(conv_params, gate, wires[:len(wires)//2])
                        self._U2_conv(conv_params, gate, wires[len(wires)//2:])

                        idx = idx + num_gates*num_params
                    
                for _, gate, num_params, gate_num_wires in U4_conv_filters : 
                    num_gates = len(wires)//2
                    
                    if trans_inv : 
                        conv_params = jnp.repeat(jnp.array([params[idx : idx + num_params]]), num_gates, axis = 0)   
                        num_gates = 1
                    else : 
                        conv_params = params[idx : idx + num_gates*num_params].reshape((num_gates, num_params))  
                            
                    self._U4_conv(conv_params, gate, wires)
                
                    idx = idx + num_gates*num_params
                
                                          
                for _, gate, num_params, gate_num_wires in noisy_filters : 
                    num_gates = len(wires)
                    
                    if trans_inv : 
                        conv_params = jnp.repeat(jnp.array([params[idx : idx + num_params]]), num_gates, axis = 0)   
                        num_gates = 1
                    else : 
                        conv_params = params[idx : idx + num_gates*num_params].reshape((num_gates, num_params))  
                            
                    self._U2_conv(conv_params, gate, wires)
                
                    idx = idx + num_gates*num_params
                                          
                                          
                _, gate, num_params, gate_num_wires = pooling
                num_gates = len(wires)//2
                
                if trans_inv : 
                    conv_params = jnp.repeat(jnp.array([params[idx : idx + num_params]]), num_gates, axis = 0)  
                    num_gates = 1 
                    
                else : 
                    conv_params = params[idx : idx + num_gates*num_params].reshape((num_gates, num_params))  
                    
                for i in range(0, len(wires), 2) : 
                    gate(*conv_params[i//2], wires = [wires[i], wires[i+1]]) 
                idx = idx + num_params*num_gates
                    
                wires = np.array([wires[i] for i in range(0, len(wires), 2)]) 
                        
            
            #Symmetry breaking term 
            for i in wires : 
                qml.RZ(params[-1], wires = i) 
                
        return circuit
    
    def get_circuit(self) :
                
        return self._get_circuit(self._num_qubits, self._num_measured, self._U2_conv_filters, self._U4_conv_filters,
                                 self._pooling, self._noisy_filters, self._alternating, self._trans_inv), self._meas_wires
    
    def __str__(self) : 
        disp = f"Quantum Convolutional Neural Network with architecture :\n" 
        disp += f"  - Convolutional filters : {[gate[0] for gate in self._conv_filters]}\n"
        disp += f"  - Pooling : {self._pooling[0]}\n"
        
        return disp
    
   