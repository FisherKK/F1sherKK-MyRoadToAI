import torch

def metric_logloss_accuracy(model, loader, criterion, use_gpu):
    with torch.no_grad():
        device = "cuda" if use_gpu else "cpu"

        loss = 0
        accuracy = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)  

            output = model.forward(images)
            loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return loss, accuracy
