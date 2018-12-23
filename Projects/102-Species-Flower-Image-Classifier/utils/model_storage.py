import os
import time
import torch

def _get_save_filepath(checkpoint_dir, checkpoint_filename):
    file_dir = os.getcwd()
    if checkpoint_dir is not None:
       if not os.path.exists(checkpoint_dir):
            raise Exception("Directory '{}' does not exist!".format(checkpoint_dir))
            
       file_dir = checkpoint_dir
    
    filename = "checkpoint_{}.pth".format(int(time.time()))
    if checkpoint_filename and checkpoint_filename is not None:
        filename = checkpoint_filename
        if ".pth" not in filename:
            filename += ".pth"
            
    return os.path.join(file_dir, filename)

def save_model(model, arch, hidden_units, classes_num, epoch, optimizer, 
               train_imagefolder, checkpoint_dir=None, checkpoint_name=None):
    
    model.class_to_idx = train_imagefolder.class_to_idx

    checkpoint = {
      "arch": arch,
      "hidden_units": hidden_units,
      "classes_num": classes_num,
      "state_dict": model.state_dict(),
      "class_to_idx": model.class_to_idx,
      "epochs": epoch,
      "optimizer": optimizer.state_dict(),
    }
  
    filepath = _get_save_filepath(checkpoint_dir, checkpoint_name)
    torch.save(checkpoint, filepath)
    
    return filepath

def load_checkpoint(checkpoint_filepath):
    if not os.path.exists(checkpoint_filepath):
        raise Exception("Filepath '{}' does not exist".format(checkpoint_filepath))
                        
    return torch.load(checkpoint_filepath)

  
    
