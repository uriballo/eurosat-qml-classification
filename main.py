from datasets import EuroSATLoader
from QNN4ESAT import QNN4ESAT
from VGG16Quanv import VGG16
import torch
import torch.nn.functional as F
import schedulefree as sf
import time
import sys

import time
import torch.nn.functional as F

LOG_FILE = "QNN4ESAT.log"

def train(model, device, train_loader, optimizer, epoch, verbose=False):
    model.train()
    optimizer.train()
    total_time = 0
    correct = 0
    total = 0
    epoch_loss = 0.0  # Variable to keep track of the total loss in an epoch

    # Open log file if 'verbose' is True
    if verbose:
        log_file = open(LOG_FILE, "a")  # Append mode

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()

        batch_time = time.time() - batch_start_time
        total_time += batch_time

        # Accuracy Calculation
        _, predicted = output.max(1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Loss Accumulation
        epoch_loss += loss.item()

        # Updated Output
        progress_bar = "#" * (batch_idx * 10 // len(train_loader)) + " " * (10 - batch_idx * 10 // len(train_loader))
        batch_accuracy = 100 * correct / total
        if batch_idx % 1 == 0:
            print(f"[i] Epoch: {epoch} | Batch: {batch_idx + 1}/{len(train_loader)} [{progress_bar}] | "
                  f"Loss: {loss.item():.4f} | Batch Time: {batch_time:.2f}s (lr={lr}) | Accuracy: {batch_accuracy:.2f}%",
                  end='\r')
            sys.stdout.flush()


    epoch_time = total_time
    epoch_accuracy = 100 * correct / total  

    # Print and log epoch summary
    print(f"\n\t[t] Epoch {epoch} completed in {epoch_time:.2f} seconds |  Loss: {epoch_loss / len(train_loader):.4f} | Accuracy: {epoch_accuracy:.2f}%") 
    if verbose:
        log_file.write(f"\nEpoch {epoch} Summary: Loss: {epoch_loss / len(train_loader):.4f} | Accuracy: {epoch_accuracy:.2f}%\n")
        log_file.close()

def test(model, device, test_loader, optimizer, verbose):
    model.eval()
    optimizer.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    print(f"\t[>] Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

if __name__ == '__main__':
    loader = EuroSATLoader(root='./EuroSAT_RGB', image_size=64, batch_size=512, test_size=0.1, random_state=42)
    train_loader, val_loader = loader.get_loaders()

    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Create the MPS device object
        print('MPS available')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print('MPS not available, falling back to CUDA')
    else:
        device = torch.device("cpu")  # Fallback to CPU if necessary
        print('MPS not available, falling back to CPU')
    
    print()
    model = QNN4ESAT(device=device)
    model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    
    opt = sf.AdamWScheduleFree(model.parameters(), lr = 0.025,  weight_decay=1e-3) 
    opt = sf.SGDScheduleFree(model.parameters(), lr= 10, weight_decay=0.0001, warmup_steps=4)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_list = []
    accuracy_list = []  # Track accuracies
    for epoch in range(20):
        gen = iter(train_loader)
        train(model, device, gen, opt, epoch, verbose=True)

        
        # Testing
        model.eval()  # Set model to evaluation mode
        opt.eval()  # Set optimizer to evaluation mode
        test_correct = 0
        test_samples = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_samples += target.size(0)
                test_correct += (predicted == target).sum().item()

        test_accuracy = 100 * test_correct / test_samples
        print('\t[>] Test Accuracy: {:.2f}%'.format(test_accuracy))
    
    print('Training complete')
    print('Saving model...')
    torch.save(model.state_dict(), 'model.pth')