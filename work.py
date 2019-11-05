import sys
from PyQt5.QtWidgets import QMainWindow,QApplication, QWidget, QPushButton, QLineEdit,QMessageBox,QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
def checklabel(list):
 print(list)
 for i in range(10):
  if(list[i]==1):
    return i
# Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 128
display_step = 100
optimizer="adamoptimizer"
fig_loss = np.zeros([num_steps])
fig_acc = np.zeros([num_steps])
model_path = "/tmp/model.ckpt"
# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
# place
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
 
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)    #load minst 
def button1_clicked():  
 image_array=mnist.train.images[0:10,:]
 image=image_array.reshape(280,28)

 plt.matshow(image, cmap = plt.get_cmap('gray'))

 k=[0,1,2,3,4,5,6,7,8,9]   #show tne images
 for j in range(10):
  k[j]=checklabel(mnist.train.labels[j,:])
 plt.yticks([0,14,42,70,98,126,154,182,210,238,266,280],[0,k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7],k[8],k[9],280])
 plt.show()

def button2_clicked():
 print("hyperparameters:")
 print("batch_size:",batch_size)
 print("learning rate:",learning_rate)
 print("optimizer:",optimizer)

def button3_clicked():
 def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
# Construct model
 logits = neural_net(X)
 prediction = tf.nn.softmax(logits)

# Define loss and optimizer
 loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
 optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
 train_op = optimizer.minimize(loss_op)

# Evaluate model
 correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
 accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


 init = tf.global_variables_initializer()
 with tf.Session() as sess:
 
    # Run the initializer
    sess.run(init)
    for epoch in range(1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        for step in range(1000):
         sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
         loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
         fig_loss[step]=loss;
         fig_acc[step]=acc;
        """if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss))
            
    print("Optimization Finished!")"""
    # Calculate accuracy for MNIST test images
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(num_steps), fig_loss, label="Loss")
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('training loss')
    plt.show()
def button4_clicked():
 def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
# Construct model
 logits = neural_net(X)
 prediction = tf.nn.softmax(logits)

# Define loss and optimizer
 loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
 optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
 train_op = optimizer.minimize(loss_op)

# Evaluate model
 correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
 accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


 init = tf.global_variables_initializer()

 saver = tf.train.Saver()
 with tf.Session() as sess:
 
    # Run the initializer
    sess.run(init)
    for epoch in range(1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        for step in range(1000):
         sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
         loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
         fig_loss[step]=loss;
         fig_acc[step]=acc;
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path) 
    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(num_steps), fig_loss, label="Loss")
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('training loss')
    fig2, ax1 = plt.subplots()
    ax1.plot(np.arange(num_steps), fig_acc, label="Loss")
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('accuracy')
     
    plt.show()
class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'test'
        self.left = 10
        self.top = 10
        self.width = 420
        self.height = 500
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

 
        self.button1 = QPushButton('5.1show train images', self)
        self.button1.move(100,50)
        self.button1.resize(200,32);
        self.button1.clicked.connect(button1_clicked)
        
        button2 = QPushButton('5.2show hyperparameters', self)
        button2.move(100,100)
        button2.resize(200,32);
        button2.clicked.connect(button2_clicked)
        
        button3 = QPushButton('5.3train 1 epoch', self)
        button3.move(100,150)
        button3.resize(200,32);
        button3.clicked.connect(button3_clicked)
        
        button4 = QPushButton('5.4show training result', self)
        button4.move(100,200)
        button4.resize(200,32);
        button4.clicked.connect(button4_clicked)
        self.show()
    @pyqtSlot()
    def on_click(self):
      text =self.textbox.text()
      print(text)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

