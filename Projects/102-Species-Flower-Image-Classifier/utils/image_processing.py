import numpy as np 

from PIL import Image

def process_image(image, transform):
    img = Image.open(image)
    img = transform(img).float()
    
    np_img = np.array(img)
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    np_img = (np.transpose(np_img, (1, 2, 0)) - means) / stds
    np_img = np.transpose(np_img, (2, 0, 1))
    
    return np_img