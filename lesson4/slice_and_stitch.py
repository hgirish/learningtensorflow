import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"
raw_image_data  = mpimg.imread(filename)

height, width, depth = raw_image_data.shape
x_height = int(height / 2) 
x_width =  int(width /2)
print("height: {0}, width: {1}, x_height: {2}, x_widht: {3}".format(height, width,x_height, x_width))
image = tf.placeholder("uint8")
#slicedimage = tf.slice(image,[0,0,0],[x_height, x_width,-1])
#slicedimage = tf.slice(image,[x_height,0,0],[-1, x_width,-1])
#slicedimage = tf.slice(image,[0,x_width,0],[x_height, -1,-1])
#slicedimage = tf.slice(image,[x_height,x_width,0],[-1, -1,-1])
arr = [
    tf.slice(image,[0,0,0],[x_height, x_width,-1]),   
    tf.slice(image,[0,x_width,0],[x_height, -1,-1]),
    tf.slice(image,[x_height,0,0],[ -1,x_width, -1]),
    tf.slice(image,[x_height,x_width,0],[-1, -1,-1])
]

idx = 0
fig = plt.figure()
with tf.Session() as session:
    for slicedimage in arr:
        idx = idx + 1
        #fig = plt.figure()
        a = fig.add_subplot(2,2,idx)
        result = session.run(slicedimage, feed_dict={image: raw_image_data})
        print(result.shape)
        imgplot = plt.imshow(result)
       # plt.show()


plt.show()