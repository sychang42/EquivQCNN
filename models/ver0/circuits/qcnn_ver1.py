import sys 
import os
sys.path.append(os.path.dirname(__file__)) 

import json

import numpy as np
import pennylane as qml 
import unitary
import equiv_unitary


def get_conv_map(wires) : 
    if 

class QCNN() : 
    _valid_gates = {
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
        "U2_ver2" : equiv_unitary.U2_ver2,
        "U4_ver1" : equiv_unitary.U4_ver1,
        "Pooling_ansatz1" : equiv_unitary.Pooling_ansatz1,
        "Pooling_ansatz2" : equiv_unitary.Pooling_ansatz2
    } 
    
    qnn_config_path = os.path.join(os.path.dirname(__file__), "equiv_qnn_architecture.json")
    
    
    def __init__(self, 
                 num_qubits: int,
                 num_measured: int,
                 **kwargs
                ) : 
        
        qnn_architecture = {'U2_conv_filters' : ['U2_ver1'],
                            'U4_conv_filters' : ['U4_ver1'],
                            'pooling' : 'Pooling_ansatz1', 
                             'alternating' : False} 
        if "qnn_ver" in kwargs.keys() : 
            qnn_architecture = json.load(open(self.qnn_config_path))[kwargs['qnn_ver']]
        else : 
            for k in kwargs.keys() : 
                qnn_architecture[k] = kwargs[k]
         
        self._qnn_architecture = qnn_architecture
        self._num_qubits = num_qubits
        self._num_measured = num_measured
        
        self._rotation_gates = []
        if 'U2_conv_filters' in qnn_architecture.keys() : 
            self._U2_conv_filters = [self._choose_gate(gate) for gate in qnn_architecture['U2_conv_filters']]
         
        if 'U4_conv_filters' in qnn_architecture.keys() : 
            self._U4_conv_filters = [self._choose_gate(gate) for gate in qnn_architecture['U4_conv_filters']]
            
        self._pooling = []
        if 'pooling' in qnn_architecture.keys() : 
            self._pooling = self._choose_gate(qnn_architecture['pooling']) 
        if 'alternating' in qnn_architecture.keys() : 
            self._alternating = qnn_architecture['alternating']
            
        depth = int(np.log2(num_qubits//num_measured))
        self._meas_wires = np.array([i for i in range(num_qubits)]) 
        while len(self._meas_wires) > num_measured : 
            self._meas_wires = np.array([self._meas_wires[i] for i in range(0, len(self._meas_wires), 2)])
        if self._alternating :    
            self._num_params = sum([gate[2] for gate in self._U2_conv_filters]) + \
                                depth*(sum([gate[2] for gate in self._U4_conv_filters]) + self._pooling[2])  
        else : 
            self._num_params = depth*(sum([gate[2] for gate in self._U2_conv_filters]) + \
                                      sum([gate[2] for gate in self._U4_conv_filters]) + self._pooling[2])
    @staticmethod
    def _choose_gate(gate_str : str) : 
        gate = EquivQCNN._valid_gates.get(gate_str, None) 
        
        if gate is None : 
            raise NotImplementedError("Unknown gate.")  
            
        return (gate_str, gate, gate.num_params, gate.num_wires)
                            
    
    def _U2_conv(self, params, gate, wires) : 
        for i in range(0, len(wires), 2) : 
            gate(*params, wires = [wires[i], wires[i+1]])
            
        for i in range(1, len(wires)-1, 2) : 
            gate(*params, wires = [wires[i], wires[i+1]])
        
        gate(*params, wires = [wires[-1], wires[0]])    
        
    def _U4_conv(self, params, gate, wires) : 
        for i in range(0, len(wires), 4):
            gate(*params, wires = [wires[i + j] for j in range(4)])

        for i in range(2, len(wires)-2, 4):
            gate(*params, wires = [wires[i + j] for j in range(4)])

        if len(wires) > 4 : 
            gate(*params, wires = [wires[0], wires[1], wires[-2], wires[-1]])
            
    def _get_circuit(self, num_qubits, num_measured, U2_conv_filters, U4_conv_filters, pooling, alternating) : 

        def circuit(params) : 
            idx = 0
            
            wires = np.array([i for i in range(num_qubits)])
            
            if not alternating : 
                for _, gate, num_params, gate_num_wires in U2_conv_filters : 
                    self._U2_conv(params[idx : idx + num_params], gate, wires[:len(wires)//2])
                    self._U2_conv(params[idx : idx + num_params], gate, wires[len(wires)//2:])
                    
                    idx = idx + num_params
            
            while len(wires) > num_measured : 
                if alternating : 
                    for _, gate, num_params, gate_num_wires in U2_conv_filters : 
                        self._U2_conv(params[idx : idx + num_params], gate, wires[:len(wires)//2])
                        self._U2_conv(params[idx : idx + num_params], gate, wires[len(wires)//2:])

                        idx = idx + num_params
                    
                for _, gate, num_params, gate_num_wires in U4_conv_filters : 
                    self._U4_conv(params[idx : idx + num_params], gate, wires)
                
                    idx = idx + num_params
                    
                _, gate, num_params, gate_num_wires = pooling
                for i in range(0, len(wires), 2) : 
                    gate(*params[idx : idx + num_params], wires = [wires[i], wires[i+1]]) 
                idx = idx + num_params    
                wires = np.array([wires[i] for i in range(0, len(wires), 2)]) 
                        
            
        return circuit
    
    def get_circuit(self) :
                
        return self._get_circuit(self._num_qubits, self._num_measured, self._U2_conv_filters, self._U4_conv_filters,
                                 self._pooling, self._alternating), self._meas_wires
    
    def __str__(self) : 
        disp = f"Quantum Convolutional Neural Network with architecture :\n" 
        disp += f"  - Convolutional filters : {[gate[0] for gate in self._conv_filters]}\n"
        disp += f"  - Pooling : {self._pooling[0]}\n"
        
        return disp


