import numpy as np

import torch
from torch.autograd import Variable

def predict(numpy_image, model, top_k, use_gpu):
    device = "cuda" if use_gpu else "cpu"
    
    tensor_img = torch.FloatTensor(numpy_image)
    tensor_img = Variable(tensor_img, requires_grad=True)
    tensor_img = tensor_img.unsqueeze(0) 
    tensor_img = tensor_img.to(device)
    
    model.to(device)
    model.eval()
    
    logits, class_ids = model.forward(tensor_img).topk(top_k)
    probs = torch.exp(logits)
    
    ids_map = {v: k for k, v in model.class_to_idx.items()}
    classes = [ids_map[x] for x in class_ids.cpu().numpy().ravel()]
   
    return probs.detach().cpu().numpy().ravel(), np.array(classes).astype(np.int64)

