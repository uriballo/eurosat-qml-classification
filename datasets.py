import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import collections

class EuroSATLoader:
    def __init__(self, root, image_size=64, batch_size=256, test_size=0.2, random_state=42):
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            #transforms.RandomHorizontalFlip(p=0.5),      # Flips for more viewpoints
            #transforms.RandomRotation(degrees=30),       # Wider range of angles
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),   # Simulate lighting variation
            #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight shifts
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])


    def get_loaders(self):  
        dataset = torchvision.datasets.ImageFolder(root=self.root, transform=self.transform)
        
        train_size = 1 - self.test_size
        if train_size + self.test_size != 1.0:
            raise ValueError("train_size + test_size must equal 1.0")

        train_set, val_set = train_test_split(dataset, test_size=self.test_size, train_size=train_size, random_state=self.random_state, stratify=dataset.targets)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

        class_counts = collections.Counter(target for _, target in train_set) 
        for class_label, count in sorted(class_counts.items()):
            print(f"Class {class_label}: {count} members")

        return train_loader, val_loader