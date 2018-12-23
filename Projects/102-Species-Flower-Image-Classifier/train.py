import argparse

from utils.transform_presets import (
    KEY_TRANSFORM_TRAIN,
    KEY_TRANSFORM_VAL,
    KEY_TRANSFORM_TEST,
    T_AGUMENTATION_RESIZE_CROP_NORM,
    T_RESIZE_CROP_NORM
)

from utils.dataset_loading import load_imagefolders, create_dataloaders
from utils.model_building import build_model
from utils.model_training import train_and_test_model
from utils.model_validation import metric_logloss_accuracy
from utils.model_storage import save_model


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Path to directory where dataset is located", required=True)
parser.add_argument("--save_dir", type=str, help="Path to which trained model will be saved")
parser.add_argument("--save_filename", type=str, help="Name of saved model")
parser.add_argument("--arch", type=str, help="Network architecture names", 
                    choices=["vgg16", "vgg19", "alexnet"], default="vgg16")
parser.add_argument("--epochs", type=int, help="Epochs for how long network should be trained", default=2)
parser.add_argument("--hidden_units", type=int, help="Number of neuron in classifier hidden layer", default=1024)
parser.add_argument("--learning_rate", type=float, help="Determine size of network parameters update step", default=0.001)
parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

args = parser.parse_args()

if __name__ == "__main__":
    
    # Loading data
    TRANSFORMS = {
        KEY_TRANSFORM_TRAIN: T_AGUMENTATION_RESIZE_CROP_NORM,
        KEY_TRANSFORM_VAL: T_RESIZE_CROP_NORM,
        KEY_TRANSFORM_TEST: T_RESIZE_CROP_NORM
    }

    train_imagefolder, val_imagefolder, test_imagefolder = load_imagefolders(
        args.data_dir, TRANSFORMS, KEY_TRANSFORM_TRAIN, KEY_TRANSFORM_VAL, KEY_TRANSFORM_TEST)
    print("Data successfully loaded.")

    train_datalodaer, val_dataloader, test_dataloader = create_dataloaders(
        train_imagefolder, val_imagefolder, test_imagefolder)
    print("Created data loaders.")

    # Building model
    classes_num = len(train_datalodaer.dataset.classes)
    model = build_model(args.arch, args.hidden_units, classes_num)
    print("Model successfully built.")
    print("Model architecture:\n\n{}".format(model))
          
    # Training & Testing
    print("Training and testing model.")
    model, epoch, optimizer = train_and_test_model(
        model, train_datalodaer, val_dataloader, test_dataloader, args.epochs, 
        args.learning_rate, args.gpu, validation_function=metric_logloss_accuracy
    )
          
    # Saving model
    result_dir = save_model(model, args.arch, args.hidden_units, classes_num, epoch, optimizer, 
                            train_imagefolder, checkpoint_dir=args.save_dir, checkpoint_name=args.save_filename)
    print("Model saved to dir: {}".format(result_dir))
