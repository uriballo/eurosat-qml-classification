import pennylane as qml
from pennylane import numpy as np

dev = qml.device("lightning.qubit", wires=4)

# circuit used in the paper + encoding
rotation_circuit_shapes = {"weights": (4,)}
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def rotation_circuit(inputs, weights):
    for i in range(4):
        qml.RY(inputs[i] * np.pi, wires=i)
        
    for i in range(4):
        qml.RY(weights[i], wires=i)
        
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


strongly_entangled_circuit_shapes = {"weights": (8,)}
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def strongly_entangled_circuit(inputs, weights):
    for i in range(4):
        qml.RY(inputs[i] * np.pi, wires=i)
        
    for i in range(4):
        qml.RZ(weights[i], wires=i)
    for j in range(4):
        qml.RX(weights[j], wires=j)
        
    for k in range(3):
        qml.CNOT(wires=[k, k+1]) 
     
        
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    