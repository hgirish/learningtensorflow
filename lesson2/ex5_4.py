'''Use TensorBoard to visualise the graph for some of these examples. using example 4'''
import tensorflow as tf
import numpy as np

y = tf.Variable(np.random.randint(1000), name='y')
avg = tf.Variable(0., name='avg')
total = tf.Variable(0., name='total')
model = tf.global_variables_initializer()

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tempfiles',session.graph)
    for idx in range(1000):
        session.run(model)
        y = np.random.randint(1000)
        total += y
        avg = (total)/(idx+1.0)
        print(idx,session.run(avg))




   
