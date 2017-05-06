import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"
image = mpimg.imread(filename)
#im = np.fliplr(image)
print("Original Shape: ", image.shape)
#print("fliplr shape",im.shape)
height, width, depth = image.shape

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    #x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
    x = tf.image.flip_left_right(image)
    session.run(model)
    result = session.run(x)

print("result shape: ",result.shape)
plt.imshow(result)
plt.show()