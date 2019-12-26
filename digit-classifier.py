import input_data
mnist = input_data.read_data_sets("/Users/davidaraya/Documents/GitHub/Digit-Images-Classifier", one_hot=True)

# import tensorflow as tf

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
with tf.name_scope("Wx_b") as scope:
    #Constrct a linear model
    #First Scope: Logistic Regression on MNIST
        #We matrix multiply the input images x by the weigth matrix W and then adding the bias b
    model = tf.nn.softmax(tf.matmul(x, W) + b) #Softmax
    
#Add summary ops to collect data
#Summary operations help us visualize the distribution of our weights and biases
w_h = tf.histogram_summary("weights", W)
b_h = tf_histogram_summary("biases", b)

#Second Scope:
#More name scopes will up graph representation
#Cost function helps us minimize errors during training
with tf.name_scope("cost_function") as scope:
    #Minimize error using entropy
    #Cost Entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    
    #Create a summary to monitor the cost function to visualize it later
    tf.scalar_summary("cost_function", cost_function)
    
#Create optimization function to make model improve during training
#We use Gradient Descent that takes our learning rate as a parameter for pacing and our cost function as a parameter to help minimize the error.
with tf.name_scope("train") as scope:
    #Gradient descent
    optimizer = tf.traing.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    
#Initializing the variables
#After building our graph we initialize all our variables
init = tf.initialize_all_variables()

#Merge all summaries into a single operator
merged_summary_op = tf.merge_all_summaries()

#Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    #Set the logs writer to the folder /tmp/tensorflow_logs
    summary_writer = tf.train.SummaryWriter('/home/sergo/work/logs', graph_def=sess.graph_def)
    
    #Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        #loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            #Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            
            #Computer the average loss
            avg_cost += sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
            
        #Display logs per iteration step
        if iteration % display_step == 0:
            print ("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))
    
    print ("Tuning completed!")
    
    #Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    
    #Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.text.images, y: mnist.test.labels}))
    
            
