import torch 
import os
import shutil
import pdb

class Checkpoint(object):
    def __init__(self, option):
        save_dir = os.path.join(option.save_path, option.experiment_id)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir
        self.option = option
    
    def save_checkpoint(self, state, is_best):
        checkpoint_path = os.path.join(self.save_dir, "checkpoint.pth.tar")
        best_path = os.path.join(self.save_dir, "best_checkpoint.pth.tar")
        torch.save(state, checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, best_path)
        print("Saved checkpoint")

    def load_checkpoint(self, checkpoint_path=None):
        if os.path.isfile(self.option.resume):
            checkpoint = torch.load(self.option.resume)
            print("Loaded {}".format(self.option.resume))
            return checkpoint
        else:
            print("{} does not exist.".format(self.option.resume))
            return None