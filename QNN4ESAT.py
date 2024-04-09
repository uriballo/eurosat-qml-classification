import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pennylane.qnn as qnn

from circuits import rotation_circuit, rotation_circuit_shapes
from circuits import strongly_entangled_circuit, strongly_entangled_circuit_shapes
from circuits import quanv_circuit, quanv_circuit_shapes
from circuits import rot_entangle_loop_circuit, rot_entangle_loop_circuit_shapes

class QNN4ESAT(nn.Module):
    def __init__(self, device = torch.device("mps")):
        super(QNN4ESAT, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 4)
        
        self.qlayer = qnn.TorchLayer(rot_entangle_loop_circuit, rot_entangle_loop_circuit_shapes)
        self.fc4 = nn.Linear(4, 10)
        
        self.device = device
        
    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)

        x = self.conv2(x) 
        x = self.bn2(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)

        x = self.conv3(x) 
        x = self.bn3(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)

        x = self.conv4(x) 
        x = self.bn4(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)     

        x = x.view(-1, 64 * 2 * 2)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.fc2(x))
        if self.device.type == "mps":
            x = F.tanh(self.fc3(x)).to("cpu")
        else:
            x = F.tanh(self.fc3(x))
        x = torch.stack([self.qlayer(i) for i in x]).to(self.device)
        return F.softmax(self.fc4(x), dim=1)
        
        
    