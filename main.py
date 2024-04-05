from datasets import EuroSATLoader
from QNN4ESAT import QNN4ESAT
import torch

if __name__ == '__main__':
    loader = EuroSATLoader(root='./EuroSAT_RGB', image_size=64, batch_size=256, test_size=0.2, random_state=42)
    train_loader, val_loader = loader.get_loaders()

    model = QNN4ESAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    loss_list = []
    accuracy_list = []  # Track accuracies
    for epoch in range(20):
        gen = iter(train_loader)
        total_loss = []
        total_correct = 0  # For training accuracy
        total_samples = 0  # For training accuracy

        for i in range(len(train_loader)):   
            (data, target) = next(gen)

            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            # Accuracy calculation
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()

            total_loss.append(loss.item())
            print('\rBatch '+str(i+1)+' of ' + str(len(train_loader)) + ' Loss: %.5f' % loss.item(), end='')

        loss_list.append(sum(total_loss) / len(total_loss))
        accuracy_list.append(100 * total_correct / total_samples)  # Store accuracy
        print(' -> Training [{:.0f}%]\tEpoch Loss: {:.4f} Accuracy: {:.2f}%'.format(
            100. * (epoch + 1) / 20, loss_list[-1], accuracy_list[-1]))

    # Testing
    model.eval()  # Set model to evaluation mode
    test_correct = 0
    test_samples = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_samples += target.size(0)
            test_correct += (predicted == target).sum().item()

    test_accuracy = 100 * test_correct / test_samples
    print('\nTest Accuracy: {:.2f}%'.format(test_accuracy))