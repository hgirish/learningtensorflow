import matplotlib.image as mpimg
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"
image = mpimg.imread(filename)
print(image.shape)