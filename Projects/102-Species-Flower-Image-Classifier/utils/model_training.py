import torch
from torch import nn, optim


def train_and_test_model(model, train_loader, val_loader, test_loader, epochs, 
                         learning_rate, use_gpu, validation_function=None):
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print(" - Using criterion: {}".format(criterion))
    print(" - Using optimizer:\n{}".format(optimizer))
    
    device = "cuda" if use_gpu else "cpu"
    model.to(device)
    print(" - Using device: {}".format(device))
    print(" - Training epochs set to: {}".format(epochs))
    print(" - Batch size set to: {}".format(train_loader.batch_size))
    print()
    
    loss = 0
    iteration = 0
    print_every = 20       
    for e in range(epochs):
        model.train()
          
        for images, labels in train_loader:
            iteration += 1
                  
            images, labels = images.to(device), labels.to(device)                                     
            optimizer.zero_grad()
                       
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            loss += loss.item()
                       
            if iteration % print_every == 0:
                model.eval()
                
                val_loss, accuracy = validation_function(model, val_loader, criterion, use_gpu)
                       
                print("    * Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(loss / print_every),
                      "Val Loss: {:.3f}.. ".format(val_loss / len(val_loader)),
                      "Val Accuracy: {:.3f}".format(accuracy / len(val_loader)))

                loss = 0
                model.train()
                
    print()
    
    test_loss, test_accuracy = validation_function(model, test_loader, criterion, use_gpu)
    print("    * Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
          "Test Accuracy: {:.3f}".format(test_accuracy / len(test_loader)))
    
    print()
    
    return model, e, optimizer
    