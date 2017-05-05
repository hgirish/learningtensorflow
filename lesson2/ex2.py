'''Generate a NumPy array of 10,000 random numbers (called x) 
and create a Variable storing the equation y=5x2−3x+15'''
import tensorflow as tf
import numpy as np

data = np.random.randint(1000, size=10000)

x = tf.constant(data, name='x')
y = tf.Variable((5 * x * x) -( 3 * x) + 15, name='y')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    runresult = session.run(y)
    print("Length of result: %d"%(len(runresult)))
    print(runresult)
