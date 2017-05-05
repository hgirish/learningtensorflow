import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

path_of_current_file = os.path.realpath(__file__)
dir_path = os.path.dirname(path_of_current_file)
filename = dir_path + "/MarshOrchid.jpg"
image = mpimg.imread(filename)

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1,0,2])
    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()