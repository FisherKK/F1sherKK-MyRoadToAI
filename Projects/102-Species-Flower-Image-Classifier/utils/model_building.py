from collections import OrderedDict

from torchvision import models
from torch import nn


def _build_classifier(model, hidden_units, class_num):
    in_features = None
    for block in model.classifier:
        if hasattr(block, "in_features"):
            in_features = block.in_features
            break
    
    structure = [
        ("fc1", nn.Linear(in_features, hidden_units)),
        ("relu1", nn.ReLU()),
        ("fc2", nn.Linear(hidden_units, hidden_units)),
        ("relu2", nn.ReLU()),
        ("fc3", nn.Linear(hidden_units, class_num)),
        ("output", nn.LogSoftmax(dim=1))
    ]

    return nn.Sequential(OrderedDict(structure))
    
def build_model(architecture_name, hidden_units, classes_num):
    model_class = getattr(models, architecture_name)
         
    model = model_class(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = _build_classifier(model, hidden_units, classes_num)
         
    return model

def reconstruct_model(checkpoint):
    architecture_name = checkpoint["arch"]
    hidden_units = checkpoint["hidden_units"]
    classes_num = checkpoint["classes_num"]
    
    model = build_model(architecture_name, hidden_units, classes_num)
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
                        
    return model