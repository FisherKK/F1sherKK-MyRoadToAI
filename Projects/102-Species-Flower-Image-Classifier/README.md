### General
#### Description
Creating classifier that is capable of identifying 102 species of flowers.
#### Dataset
Dataset consists of 102 different classes. Each class consists of between 40 and 258 images. Total amount of images is 8189. Images are saved in .jpg format. Images have different shape and each of them is of 3-channel RGB type.

- source: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

#### Framework and Model Architecture
For training of the classifier - transfer learning on `VGG16`, `VGG19`, `AlexNet` models is performed with usage of [PyTorch](https://pytorch.org/) framework.

### Prerequisites
In order to run the models make sure to use `requirements.txt` file to setup the environment.
```
numpy==1.15.4
Pillow==5.3.0
six==1.12.0
torch==1.0.0
torchvision==0.2.1
```

Apart from that you will need data split into ImageFolders that will satisfy PyTorch's DataLoader:
```
dataset/
├── train/
│   ├──1/
│   │  ├──image_xxxx.jpg
│   │  ├──image_xxxx.jpg
│   │  └──...
│   ├──2/
│   │  ├──image_xxxx.jpg
│   │  ├──image_xxxx.jpg
│   │  └──...
│   ├──.../
│   └──102/
│      ├──image_xxxx.jpg
│      ├──image_xxxx.jpg
│      └──...
├── valid/
│   ├──1/
│   │  ├──image_xxxx.jpg
│   │  ├──image_xxxx.jpg
│   │  └──...
│   ├──2/
│   │  ├──image_xxxx.jpg
│   │  ├──image_xxxx.jpg
│   │  └──...
│   ├──.../
│   └──102/
│      ├──image_xxxx.jpg
│      ├──image_xxxx.jpg
│      └──...
└── test/
    ├──1/
    │  ├──image_xxxx.jpg
    │  ├──image_xxxx.jpg
    │  └──...
    ├──2/
    │  ├──image_xxxx.jpg
    │  ├──image_xxxx.jpg
    │  └──...
    ├──.../
    └──102/
       ├──image_xxxx.jpg
       ├──image_xxxx.jpg
       └──...
```

### Training
Use `train.py` script allowing to generate checkpoint with trained model.

Arguments:
- **--data_dir**: path to directory where dataset is located
- **--save_dir**: path to which trained model checkpoint will be saved
- **--save_filename**: name of saved model checkpoint
- **--arch**: available options are `vgg16`, `vgg19`, `alexnet`
- **--epochs**: epochs for how long network should be trained, (default is 2)
- **--hidden_units**: number of neurons in added hidden layer at the end of network (default is 1024)
- **--learning_rate**: learning rate (default is 0.001)
- **--gpu**: adding it will trigger GPU usage mode

#### Example use:
```
python train.py --data_dir flower_data/ --arch vgg19 --save_dir checkpoint --save_filename vgg19_cpkt
```

#### Example output
```
Model architecture:

VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace)
    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace)
    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace)
    (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (fc1): Linear(in_features=25088, out_features=1024, bias=True)
    (relu1): ReLU()
    (fc2): Linear(in_features=1024, out_features=1024, bias=True)
    (relu2): ReLU()
    (fc3): Linear(in_features=1024, out_features=102, bias=True)
    (output): LogSoftmax()
  )
)
Training and testing model.
 - Using criterion: NLLLoss()
 - Using optimizer:
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.001
    weight_decay: 0
)
 - Using device: cpu
 - Training epochs set to: 2
 - Batch size set to: 32

    * Epoch: 1/2..  Training Loss: 0.397..  Val Loss: 3.952..  Val Accuracy: 0.146
    * Epoch: 1/2..  Training Loss: 0.360..  Val Loss: 2.989..  Val Accuracy: 0.292
    * Epoch: 1/2..  Training Loss: 0.259..  Val Loss: 2.291..  Val Accuracy: 0.408
    * Epoch: 1/2..  Training Loss: 0.199..  Val Loss: 1.845..  Val Accuracy: 0.494
    * Epoch: 1/2..  Training Loss: 0.241..  Val Loss: 1.545..  Val Accuracy: 0.600
    * Epoch: 1/2..  Training Loss: 0.179..  Val Loss: 1.458..  Val Accuracy: 0.586
    * Epoch: 1/2..  Training Loss: 0.151..  Val Loss: 1.474..  Val Accuracy: 0.615
    * Epoch: 1/2..  Training Loss: 0.151..  Val Loss: 1.218..  Val Accuracy: 0.643
    * Epoch: 1/2..  Training Loss: 0.168..  Val Loss: 1.267..  Val Accuracy: 0.653
    * Epoch: 1/2..  Training Loss: 0.159..  Val Loss: 1.119..  Val Accuracy: 0.702
    * Epoch: 2/2..  Training Loss: 0.132..  Val Loss: 1.008..  Val Accuracy: 0.712
    * Epoch: 2/2..  Training Loss: 0.107..  Val Loss: 1.063..  Val Accuracy: 0.707
    * Epoch: 2/2..  Training Loss: 0.125..  Val Loss: 1.184..  Val Accuracy: 0.673
    * Epoch: 2/2..  Training Loss: 0.110..  Val Loss: 1.041..  Val Accuracy: 0.715
    * Epoch: 2/2..  Training Loss: 0.148..  Val Loss: 0.994..  Val Accuracy: 0.731
    * Epoch: 2/2..  Training Loss: 0.152..  Val Loss: 1.047..  Val Accuracy: 0.720
    * Epoch: 2/2..  Training Loss: 0.139..  Val Loss: 0.968..  Val Accuracy: 0.720
    * Epoch: 2/2..  Training Loss: 0.132..  Val Loss: 0.876..  Val Accuracy: 0.764
    * Epoch: 2/2..  Training Loss: 0.065..  Val Loss: 0.899..  Val Accuracy: 0.744
    * Epoch: 2/2..  Training Loss: 0.121..  Val Loss: 0.862..  Val Accuracy: 0.754

    * Test Loss: 0.879..  Test Accuracy: 0.752

Model saved to dir: checkpoint/vgg19_cpkt.pth
```

Training results for `vgg19` option with using defaults is `75.2%`.

### Inference
Use **predict.py** script allowing to use generated checkpoint for inference.

Arguments:
- **--input_image_dir**: path to input image
- **--checkpoint_filepath**: path to checkpoint file
- **--top_k**: number of top probabilities that should be displayed
- **--gpu**: adding it will trigger GPU usage mode
- **--category_names**: optional path to .json file with names for classes

#### Example usage:
```
python predict.py  --input_image_dir=flowers_data/test/43/image_02329.jpg --checkpoint_filepath=checkpoint/vgg19_cpkt.pth
```

#### Example output:
```
Results for image 'flower_data/test/43/image_02329.jpg':
  0. 91.67 % - (Class id: 43)
  1. 2.58 % - (Class id: 96)
  2. 2.09 % - (Class id: 93)
  3. 0.55 % - (Class id: 76)
  4. 0.31 % - (Class id: 75)
```

### Inference Results
Results can me more nicely visualized if needed (this is not included in script):
- image_02329.jpg

<img src="https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/Projects/102-Species-Flower-Image-Classifier/img/example1.png" width="350">

- image_02550.jpg

<img src="https://github.com/FisherKK/F1sherKK-MyRoadToAI/blob/master/Projects/102-Species-Flower-Image-Classifier/img/example2.png" width="350">
