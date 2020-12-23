import os
import sys

import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from korean_polisher.train.scheduler import learning_rate


plt.plot(learning_rate(tf.range(100000, dtype=tf.float32)))
plt.show()
