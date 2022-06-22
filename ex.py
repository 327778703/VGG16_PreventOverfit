import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def decay1( epoch):
    step_size = 1600
    iterations = epoch * 41
    base_lr = 1e-02
    max_lr = 3.65e-03
    cycle = np.floor(1 + iterations / (2 * step_size))
    x = np.abs(iterations / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * x
    return lr


def decay2(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * tf.math.pow(drop, tf.math.floor((1 + epoch) / epochs_drop))
    return lrate

plt.figure()
epoch = range(0, 100)
lrlist = []
lrlist2 = []
for i in epoch:
    lr = decay2(i)
    lr2 = decay1(i)
    lrlist.append(lr)
    lrlist2.append(lr2)
plt.plot(epoch, lrlist)
plt.show()
plt.plot(epoch, lrlist2)
plt.show()
