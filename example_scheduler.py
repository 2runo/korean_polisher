import tensorflow as tf
import matplotlib.pyplot as plt
from korean_polisher.train.scheduler import learning_rate


plt.plot(learning_rate(tf.range(100000, dtype=tf.float32)))
plt.show()
