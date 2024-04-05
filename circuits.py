import pennylane as qml
from pennylane import numpy as np

dev = qml.device("lightning.qubit", wires=4)

# circuit used in the paper + encoding
rotation_circuit_shapes = {"weights": (4,)}
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def rotation_circuit(inputs, weights):
    for i in range(4):
        qml.RX(inputs[i] * np.pi, wires=i)
        
    for i in range(4):
        qml.RY(weights[i], wires=i)
        
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


strongly_entangled_circuit_shapes = {"weights": (1, 12)}
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def strongly_entangled_circuit(inputs, weights):
    for i in range(4):
        qml.RY(inputs[i] * np.pi, wires=i)
        
    for W in weights:
        qml.Rot(W[0], W[1], W[2], wires=0)
        qml.Rot(W[3], W[4], W[5], wires=1)
        qml.Rot(W[6], W[7], W[8], wires=2)
        qml.Rot(W[9], W[10], W[11], wires=3)
        
    for k in range(3):
        qml.CNOT(wires=[k, k+1]) 
     
        
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    