import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

#Set parameters
#learning_rate defines how fast to update weights
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

#TF graph input
#A placeholder is a variable where data will be assigned for future use
x = tf.placeholder("float", [None, 784]) #mnist data image of shape 28*28 = 784
y = tf.placeholder("float", [None], 10) #0-9 digits recognition=> 10 classes


