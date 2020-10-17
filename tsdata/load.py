#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import numpy as np
import cv2
import random

import skimage.exposure



class DataPipelineConfig:
  
  def __init__(self, target_width, target_height, is_color_mode, augmentation):
    self.target_width  = target_width
    self.target_height = target_height
    self.is_color_mode = is_color_mode
    
    if not augmentation in ["none", "fliplr", "turnimprove"]:
      raise RuntimeError("Invalid augmentaiton mode!")
      
    self.augmentation = augmentation
        
  def serialize_to_string(self):
    config_serialized = "color" if self.is_color_mode else "gray"
    config_serialized += "_w" + str(self.target_width) + "_h" + str(self.target_height)
    config_serialized += "_aug-" + self.augmentation
    return config_serialized
  
  def get_keras_input_shape(self):
    if self.is_color_mode:
      return (self.target_height, self.target_width, 3)
    else:
      return (self.target_height, self.target_width)
 

class CacheFileNotFound(Exception):
  pass

  
def get_cachefile(dpconfig, set):
  return os.path.join(os.path.dirname(__file__), 
    f"cache__{dpconfig.serialize_to_string()}__{set}.npz")
   
   
def load_data(dpconfig, set):
  try:
    return load_data_cached(dpconfig, set)
  except CacheFileNotFound:
    return load_data_fresh(dpconfig, set)


def load_data_cached(dpconfig, set):
  cachefile = get_cachefile(dpconfig, set)
  
  if not os.path.isfile(cachefile):
    raise CacheFileNotFound("Cache file not found!")
    
  print(f"Loading cached {set} data...")

  npzfile = np.load(cachefile)
  return npzfile["images"], npzfile["labels"]


def save_to_cache(dpconfig, set, images, labels):
  cachefile = get_cachefile(dpconfig, set)
  print(f"Saving to cache file {cachefile}...")
  np.savez(cachefile, images = images, labels = labels)
 
  
def count_files_in_dir(dir, fileext_expected):
  num_files = 0
  
  for folder, _, files in os.walk(dir):
    for file in files:
      fileext_actual = os.path.splitext(file)[1]
      
      if fileext_actual.lower() != fileext_expected.lower():
        continue
            
      num_files += 1
      
  return num_files
  
  
class ImageContainer:
  def __init__(self, num_prealloc, sample_shape, dtype = np.float):
    prealloc_shape = [num_prealloc] + list(sample_shape)

    self._images = np.empty(shape = prealloc_shape, dtype = dtype)

    self._numImages = 0
  
    self._sample_shape = sample_shape

    
  def append(self, new_sample):
  
    if self._sample_shape != new_sample.shape:
      raise RuntimeError("Invalid sample shape!")
      
    if new_sample.dtype != self._images.dtype:
      raise RuntimeError("Invalid data type!")

    if self._numImages == self._images.shape[0]:
      raise RuntimeError("Capacity exceeded!")
      
    self._images[self._numImages, Ellipsis] = new_sample
      
    self._numImages += 1
      
        
  def get_images(self):
    return self._images[0:self._numImages, Ellipsis]
  
  
def load_data_fresh(dpconfig, set):

  is_left_right_flippable = get_is_left_right_flippable()

  label_min = 0
  label_max = 20

  if set == "train":
    dataroot = os.path.join(os.path.dirname(__file__), "train")
  elif set == "test":
    dataroot = os.path.join(os.path.dirname(__file__), "test")
  else:
    raise RuntimeError("Invalid set specified!")
 
  print(f"Reading images from '{dataroot}' directory...\n")
    
  # Have a rough estimation about how much memory we will need based on the
  # number of files
  num_files = count_files_in_dir(dataroot, ".ppm")    

  imageContainer = ImageContainer(num_files * 2, dpconfig.get_keras_input_shape())
  labels = []
    
  for folder, _, files in os.walk(dataroot):
    for file in files:
                       
      filename_full = os.path.join(os.path.abspath(folder), file)      
        
      fileext = os.path.splitext(filename_full)[1]
      
      if fileext.lower() != ".ppm":
        continue
   
      curr_label = os.path.basename(os.path.normpath(folder))
      
      try:
        curr_label = int(curr_label)
      except ValueError:
        raise RuntimeError(f"Unexpected directory structure for file: {filename_full}")
      
      if (curr_label < label_min) or (curr_label > label_max):
        raise RuntimeError(f"Invalid label for file: {filename_full}")
      
      curr_img = load_image(dpconfig, filename_full)

      # Plain image
      imageContainer.append(curr_img)
      labels.append(curr_label)

      # Augmentation: Horizontal mirroring
      if dpconfig.augmentation in ["fliplr", "turnimprove"]:
        if is_left_right_flippable[curr_label]:
          imageContainer.append(np.fliplr(curr_img))
          labels.append(curr_label)
          
      # Additional augmentation for left and right turns
      if dpconfig.augmentation == "turnimprove":
        if curr_label == 12:      # left hand curve
          imageContainer.append(np.fliplr(curr_img))
          labels.append(13)
        elif curr_label == 13:    # right hand curve
          imageContainer.append(np.fliplr(curr_img))
          labels.append(12)
      
  if len(labels) == 0:
    raise RuntimeError(f"Images of the {set} dataset are unavailable! Please check the instructions in the README file regarding dataset extraction!")
      
  images = imageContainer.get_images()
  labels = np.asarray(labels)
  
  # Shuffle
  shuffled_indices = np.random.permutation(len(labels))
  images = images[shuffled_indices, Ellipsis]  
  labels = labels[shuffled_indices]

  save_to_cache(dpconfig, set, images, labels)
      
  return images, labels
  
  
def load_image(dpconfig, image_filename):

  img = cv2.imread(image_filename, cv2.IMREAD_COLOR)
  
  if not dpconfig.is_color_mode:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  else:
  
    # Do CLAHE contrast improvement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
    lab[...,0] = clahe.apply(lab[...,0])    
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)    
  
    ##  # Work with RGB, not OpenCV's default BGR
    ##  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
  # Scale data to range 0...1
  img = img / 255.0
    
  img = cv2.resize(img, (dpconfig.target_width, dpconfig.target_height), interpolation = cv2.INTER_LINEAR)
  
  return img

  
def get_class_names():
  return [
    "speed limit 20",
    "speed limit 50",
    "speed limit 70",
    "no overtaking",
    "roundabout",
    "priority road",
    "give way",
    "stop",
    "road closed",
    "no heavy goods vehicles",
    "no entry",
    "obstacles",
    "left hand curve",
    "right hand curve",
    "keep straight ahead",
    "slippery road",
    "keep straight or turn right",
    "construction ahead",
    "rough road",
    "traffic lights",
    "school ahead" ]

    
def get_is_left_right_flippable():
  # For augmentation, images can be mirrored.
  # For some signs this makes sense, for others it doesn't.  

  return [
    False,  # speed limit 20
    False,  # speed limit 50
    False,  # speed limit 70
    False,  # no overtaking
    False,  # roundabout
    True,   # priority road
    True,   # give way
    False,  # stop
    True,   # road closed
    False,  # no heavy goods vehicles
    True,   # no entry
    True,   # obstacles
    False,  # left hand curve
    False,  # right hand curve
    True,   # keep straight ahead
    False,  # slippery road
    False,  # keep straight or turn right
    False,  # construction ahead
    True,   # rough road
    True,   # traffic lights
    False ] # school ahead

    
    
    
def main():

  dpconfig = DataPipelineConfig(
    target_width  = 128,
    target_height = 128,
    is_color_mode = False,
    augmentation  = "fliplr"
  ) 
  
  image_filename = "../train/traffic_signs_train/germany/6/00003_00029.ppm"
  
  img = load_image(dpconfig, image_filename)
  
  cv2.imshow("img", img)
  
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  
if __name__ == "__main__":
  main()
