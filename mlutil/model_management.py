import os
import re
import pathlib
import datetime

import tensorflow as tf



def get_curr_timestamp():
  return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  
  
def generate_model_family_name(dpconfig, friendly_name):
  return get_curr_timestamp() + "__" + friendly_name + "__" + dpconfig.serialize_to_string()


def get_model_model_checkpoint_dir(model_family_name):
  return os.path.join(os.path.abspath(""), "model_chkpts", model_family_name) 

  
def load_model_epoch(model, model_family_name, epoch_num):
  model_checkpoint_dir = get_model_model_checkpoint_dir(model_family_name) 
    
  for folder, _, files in os.walk(model_checkpoint_dir):
    for file in files:
        
      basename, fileext = os.path.splitext(file)
      
      if fileext.lower() != ".hdf5":
        continue
        
      regex_match = re.match(r"^epoch(\d+)_.+", basename)
      curr_epoch_num = int(regex_match.group(1))
            
      if curr_epoch_num == epoch_num:
        curr_filename_full = os.path.join(os.path.abspath(folder), file)      
      
        model.load_weights(curr_filename_full)
        return
     
  raise RuntimeError("Could not find model weights!")

    
def get_model_checkpointer(model_family_name):
  model_checkpoint_dir = get_model_model_checkpoint_dir(model_family_name) 
  pathlib.Path(model_checkpoint_dir).mkdir(parents = True, exist_ok = False)
  checkpoint_filepath = os.path.join(model_checkpoint_dir, "epoch{epoch:03d}_vloss{val_loss:.2f}_vacc{val_accuracy:.2f}.hdf5")
  return tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath, save_weights_only = True, save_best_only = False, verbose = 0)
