"""
lr scheduler
"""
import tensorflow as tf
from . import options as opt


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, opt.float_dtype)

        self.warmup_steps = warmup_steps
        self.batch = opt.BATCH_SIZE / 32  # 실제 배치 사이즈 / 32
        self.batch = tf.cast(self.batch, dtype=tf.float32)


    def __call__(self, step):
        arg1 = tf.math.rsqrt(step * self.batch)
        arg2 = (step * self.batch) * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        lr *= 1
        tf.print(lr, end='\r')
        return lr
