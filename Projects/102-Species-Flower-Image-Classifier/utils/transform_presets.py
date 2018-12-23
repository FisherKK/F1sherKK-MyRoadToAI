from torchvision import transforms

KEY_TRANSFORM_TRAIN = "key_train"
KEY_TRANSFORM_VAL = "key_val"
KEY_TRANSFORM_TEST = "key_test"

T_AGUMENTATION_RESIZE_CROP_NORM = transforms.Compose(
    [transforms.RandomRotation(30),
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])]
)

T_RESIZE_CROP_NORM = transforms.Compose(
     [transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])]
 )

T_RESIZE_CROP = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])