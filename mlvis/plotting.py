
import numpy as np
import matplotlib.pyplot as plt


def plot_image(class_names, predictions_array, true_label, img):
  
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  if (len(img.shape) == 2) or ((len(img.shape) == 3) and (img.shape[2] == 1)):
    colormap = 'gray'
  else:
    colormap = None
  
  plt.imshow(img, cmap=colormap)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

                                
def plot_value_array(class_names, predictions_array, true_label):
  num_classes = len(class_names)

  plt.grid(False)
  plt.xticks(range(num_classes))
  plt.yticks([])
  thisplot = plt.bar(range(num_classes), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  