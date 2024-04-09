import pennylane as qml
from pennylane import numpy as np
import torch

dev = qml.device("lightning.qubit", wires=4)

# circuit used in the paper + encoding
rotation_circuit_shapes = {"weights": (4,)}
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def rotation_circuit(inputs, weights):
    for i in range(4):
        qml.RY(inputs[i] * np.pi, wires=i)
        qml.Hadamard(wires=i)
        
    for i in range(4):
        qml.RZ(weights[i], wires=i)
#        qml.RX(weights[4 + i], wires=i)

        
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


strongly_entangled_circuit_shapes = {"weights": (8,)}
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def strongly_entangled_circuit(inputs, weights):
    for i in range(4):
        qml.RY(inputs[i] * np.pi, wires=i)
        qml.Hadamard(wires=i)
        
    for k in range(3):
        qml.CNOT(wires=[k, k+1])  
        
    for i in range(4):
        qml.RZ(weights[i], wires=i)
        qml.RX(weights[4 + i], wires=i)
        
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

quanv_circuit_shapes = {"weights": (4,)}
@qml.qnode(dev,interface="torch", diff_method="adjoint")
def quanv_circuit(inputs, weights):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi * inputs[j], wires=j)
   
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)

    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[0, 3])
    
    qml.RY(weights[2], wires=0)
    qml.RY(weights[3], wires=3)

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

rot_entangle_loop_circuit_shapes = {"weights": (6,)}
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def rot_entangle_loop_circuit(inputs, weights):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi * inputs[j], wires=j)

    for i in range (2):
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[1, 3])
        qml.RZ(weights[0+i], wires=3 -i)
        qml.RX(weights[1 + i], wires=1- i)
        qml.CNOT(wires=[3-i, 2- i])
        qml.CNOT(wires=[1, 0])
        qml.RZ(weights[2+ i], wires=1-i)
        qml.Hadamard(wires=3-i)

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]