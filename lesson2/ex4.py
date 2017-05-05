'''Using the code from (2) and (3) above, create a program 
that computers the “rolling” average of the following line of code: 
np.random.randint(1000). In other words, keep looping, and in each loop, 
call np.random.randint(1000) once in that loop, and store the current average 
in a Variable that keeps updating each loop.'''
import tensorflow as tf
import numpy as np

y = tf.Variable(np.random.randint(1000), name='y')
avg = tf.Variable(0., name='avg')
total = tf.Variable(0., name='total')
model = tf.global_variables_initializer()

with tf.Session() as session:
    for idx in range(1000):
        session.run(model)
        y = np.random.randint(1000)
        total += y
        avg = (total)/(idx+1.0)
        print(idx,session.run(avg))
