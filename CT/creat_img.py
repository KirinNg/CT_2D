import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

ct = np.loadtxt("data/data_5.txt")
output = tf.Variable(tf.zeros([512,512]))
sess = tf.Session()

loss = 0

for i in range(180):
    loss += np.sum(cv2.rotate(sess.run(output.eval()),2*i),0)-ct[:,i]

train = tf.train.RMSPropOptimizer(0.001).minimize(loss)
sess.run(train)

out = sess.run(output.eval())
np.save("out.npy",out)
plt.imshow(out)
plt.show()