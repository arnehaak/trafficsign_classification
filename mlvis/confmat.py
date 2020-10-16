import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib



def load_traffic_sign_icons(num_traffic_signs):
  icon_list = []
  for idx in range(num_traffic_signs):
    curr_file = "icons/{:02d}.png".format(idx)
    
    if not os.path.isfile(curr_file):
      raise RuntimeError(f"File does not exist: {curr_file}")
   
    curr_img = plt.imread(curr_file)
 
    icon_list.append(curr_img)

  return icon_list
 
 
# Coordinate systems:
#   figure points   : points from the lower left corner of the figure
#   figure pixels   : pixels from the lower left corner of the figure
#   figure fraction : 0,0 is lower left of figure and 1,1 is upper, right
#   axes points     : points from lower left corner of axes
#   axes pixels     : pixels from lower left corner of axes
#   axes fraction   : 0,0 is lower left of axes and 1,1 is upper right
#   offset points   : Specify an offset (in points) from the xy value
#   offset pixels   : Specify an offset (in pixels) from the xy value
#   data            : use the axes data coordinate system
 
 
def add_icon_xaxis(ax, pos, icon_img, icon_zoom):
  imagebox = matplotlib.offsetbox.OffsetImage(icon_img, icon_zoom)
  imagebox.image.axes = ax

  ab = matplotlib.offsetbox.AnnotationBbox(
                      offsetbox     = imagebox,
                      xy            = (pos, 0),                   # Where the (invisible) arrow tip is pointing to
                      xybox         = (0, -14),                   # Location of the icon
                      xycoords      = ("data", "axes fraction"),  # Coordinate systems of xy ("data" coordinate for pos, "axes fraction" for 0)
                      boxcoords     = "offset points",            # Coordinate system of xybox
                      box_alignment = (.5, .5),                   # Anchor point of the icon, with (0,0) = TL and (1,1) = BR
                      bboxprops     = {"edgecolor" : "none"})

  ax.add_artist(ab)
  
 
def add_icon_yaxis(ax, pos, icon_img, icon_zoom):
  imagebox = matplotlib.offsetbox.OffsetImage(icon_img, icon_zoom)
  imagebox.image.axes = ax

  ab = matplotlib.offsetbox.AnnotationBbox(
                      offsetbox     = imagebox,
                      xy            = (0, pos),                   # Where the (invisible) arrow tip is pointing to
                      xybox         = (-14, 0),                   # Location of the icon relative to xy
                      xycoords      = ("axes fraction", "data"),  # Coordinate systems of xy ("axes fraction" for 0, "data" coordinate for pos)
                      boxcoords     = "offset points",            # Coordinate system of xybox
                      box_alignment = (.5, .5),                   # Anchor point of the icon, with (0,0) = TL and (1,1) = BR
                      bboxprops     = {"edgecolor" : "none"})

  ax.add_artist(ab)
  
  
  

  
def plot_traffic_sign_confmat(confmat, xlabel = 'True label', ylabel = 'Predicted label'):

  num_traffic_signs = 21
  icon_zoom = 0.4

  if (len(confmat.shape) != 2) or (confmat.shape[0] != num_traffic_signs) or (confmat.shape[1] != num_traffic_signs):
    raise RuntimeError("Unexpected confusion matrix dimensions!")
    
  
  traffic_sign_icons = load_traffic_sign_icons(num_traffic_signs)
 
  title='Confusion matrix'
  cmap=plt.cm.Blues
  
  
  # Make the norm
  norm = matplotlib.colors.Normalize(vmin = np.min(confmat), vmax = np.max(confmat), clip = False)

  fig, ax = plt.subplots()
  ax.imshow(confmat, interpolation='nearest', cmap=cmap)

  ax.set_title(title)
  
  fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

  tick_marks = np.arange(num_traffic_signs)
  target_names = [" " for _ in range(num_traffic_signs)]
  
  ax.set_xticks(tick_marks)
  ax.set_xticklabels(target_names)
  
  ax.set_yticks(tick_marks)
  ax.set_yticklabels(target_names)

  ax.set_ylabel(ylabel, labelpad = 20)
  ax.set_xlabel(xlabel, labelpad = 10)
  
  fig.tight_layout()

  # Place icons
  for idx, icon_img in enumerate(traffic_sign_icons):
    add_icon_xaxis(ax, idx, icon_img, icon_zoom)
    add_icon_yaxis(ax, idx, icon_img, icon_zoom)

    
    
def main():

  confmat = np.random.rand(21, 21)
    
  plot_traffic_sign_confmat(confmat, xlabel = 'True label', ylabel = 'Predicted label')
  
  plt.show(block=True)

  
if __name__ == "__main__":
  main()
