import os

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

TRAIN_FOLDER = "train"
VAL_FOLDER = "valid"
TEST_FOLDER = "test"


def _assert_dir_exists(filepath):
    if not os.path.exists(filepath):
        raise Exception("Directory '{}' does not exists!".format(filepath))


def _assert_transforms_not_empty(transforms, train_key, val_key, test_key):
    if transforms is None:
        raise Exception("Dictionary 'transforms' cannot be None!")

    if len(transforms.keys()) == 0:
        raise Exception("Dictionary 'transforms' cannot be empty!")

    for transform_key in [train_key, val_key, test_key]:
        if transform_key is None:
            raise Exception("Transform key '{}' cannot be None!".format(transform_key))
        
        if transform_key not in transforms:
            raise Exception("Transform for key '{}' not found!".format(transform_key))


def load_imagefolders(filepath, transforms, 
                      train_key=None, val_key=None, test_key=None):
                            
    train_filepath = os.path.join(filepath, TRAIN_FOLDER)
    val_filepath = os.path.join(filepath, VAL_FOLDER)
    test_filepath = os.path.join(filepath, TEST_FOLDER)

    _assert_dir_exists(train_filepath)
    _assert_dir_exists(val_filepath)
    _assert_dir_exists(test_filepath)
    _assert_transforms_not_empty(transforms, train_key, val_key, test_key)

    train_imagefolder = ImageFolder(train_filepath, transform=transforms[train_key])
    val_imagefolder = ImageFolder(val_filepath, transform=transforms[val_key])
    test_imagefolder = ImageFolder(test_filepath, transform=transforms[test_key])

    return train_imagefolder, val_imagefolder, test_imagefolder


def create_dataloaders(train_imagefolder, val_imagefolder, test_imagefolder, 
                       train_batch=32, val_batch=32, test_batch=32,
                       train_shuffle=True, val_shuffle=True, test_shuffle=False):
    
    train_loader = DataLoader(train_imagefolder, batch_size=train_batch, shuffle=train_shuffle)
    val_loader = DataLoader(val_imagefolder, batch_size=val_batch, shuffle=val_shuffle)
    test_loader = DataLoader(test_imagefolder, batch_size=test_batch, shuffle=test_shuffle)
    
    return train_loader, val_loader, test_loader
                         