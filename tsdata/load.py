#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import numpy as np
import cv2


def load_image(image_filename, target_width, target_height):

  is_color_mode = False

  img = cv2.imread(image_filename, cv2.IMREAD_COLOR)
  
  if not is_color_mode:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  img = cv2.resize(img, (target_width, target_height), interpolation = cv2.INTER_LINEAR)

  # Make image stackable
  if is_color_mode:
    img = img.reshape(1, target_height, target_width, 3)
  else:
    img = img.reshape(1, target_height, target_width)
    
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
   
   
def load_data(set):

  target_width = 28
  target_height = 20
  
  label_min = 0
  label_max = 20

  if set == "train":
    dataroot = os.path.join(os.path.dirname(__file__), "train")
  elif set == "test":
    dataroot = os.path.join(os.path.dirname(__file__), "test")
  else:
    raise RuntimeError("Invalid set specified!")
 
  print(f"Reading images from '{dataroot}' directory...\n")

    
  images = []
  labels = []
    
  for folder, _, files in os.walk(dataroot):
    for file in files:
                       
      filename_full = os.path.join(os.path.abspath(folder), file)      
        
      fileext = os.path.splitext(filename_full)[1]
      
      if fileext.lower() != ".ppm":
        continue
   
      print(f'Processing file "{filename_full}"...')

      curr_label = os.path.basename(os.path.normpath(folder))
      
      try:
        curr_label = int(curr_label)
      except ValueError:
        raise RuntimeError(f"Unexpected directory structure for file: {filename_full}")
      
      if (curr_label < label_min) or (curr_label > label_max):
        raise RuntimeError(f"Invalid label for file: {filename_full}")
      
      curr_img = load_image(filename_full, target_width, target_height)
            
      images.append(curr_img)
      labels.append(curr_label)
      
  images = np.concatenate(images, axis=0)
      
  return images, labels
  