'''You can also update variables in loops, which we will use later for machine learning.'''
import tensorflow as tf

x = tf.Variable(0, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    for _ in range(5):
        session.run(model)
        x = x + 1
        print(session.run(x))
        