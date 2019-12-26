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

#The flatten digit image, is the process of turning a 3D array to a 2D
#It is more efficient formatting
x = tf.placeholder("float", [None, 784]) #mnist data image of shape 28*28 = 784

#The output value will be a 10 dimentional vector of which digit class the correspodning mnist image belongs to
y = tf.placeholder("float", [None], 10) #0-9 digits recognition=> 10 classes

#Set model weight
#The weights are the probability that affects how data flows in the graph
#The are updated during training for that the results get closer to the right solution
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Scopes help us organize nodes in the graphic visualizer (Tensorboard)
wth tf.name_scope("Wx_b") as scope:
    #Constrct a linear model
    #First Scope: Logistic Regression on MNIST
        #We matrix multiply the input images x by the weigth matrix W and then adding the bias b
    model = tf.nn.softmax(tf.matmul(x, W) + b) #Softmax
    
#Add summary ops to collect data
#Summary operations help us visualize the distribution of our weights and biases
w_h = tf.histogram_summary("weights", W)
b_h = tf_histogram_summary("biases", b)


