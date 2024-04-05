import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pennylane.qnn as qnn

from circuits import rotation_circuit, rotation_circuit_shapes

class QNN4ESAT(nn.Module):
    def __init__(self):
        super(QNN4ESAT, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.dropout = nn.Dropout(p=.3)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 4)
        
        self.qlayer = qnn.TorchLayer(rotation_circuit, rotation_circuit_shapes)
        self.fc3 = nn.Linear(4, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        x = torch.stack([self.qlayer(i) for i in x])
        return self.fc3(x)
        
        
    