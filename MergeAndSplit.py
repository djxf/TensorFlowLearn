#合并和分割， Merge and Split
# *tf.concat: 合并
# *tf.split: 分割
# *tf.stack: 堆叠(栈), j
#       create new dim
# *tf.unstack:

import tensorflow as tf
a = tf.ones([4, 35, 8])
b = tf.ones([2, 35, 8])
c = tf.concat([a, b], axis=0)
print(c.shape) # (6, 35, 8)

a = tf.ones([4, 32, 8])
b = tf.ones([4, 3, 8])
c = tf.concat([a, b], axis=1)
print(c.shape) # (4, 35, 8)

a = tf.ones([2, 3])
b = tf.ones([2, 3])
print(tf.stack([a, b], axis=0).shape) # (2, 2, 3)

#concat, stack区别？
#stack： Shapes of all input must match.
#concat: Dimensions of inputs should match.


#unstack: stack的逆操作。
a1 = tf.ones([4, 3, 2, 5])
print(len(tf.unstack(a1, axis=1)))
print(tf.unstack(a1, axis=1))


#split: 可以指定分割的数量
a2 = tf.ones([3, 4, 2])
res = tf.split(a2, axis=0, num_or_size_splits=3)
print(res[0])
#res:
    # tf.Tensor(
    # [[[1. 1.]
    #   [1. 1.]
    #   [1. 1.]
    #   [1. 1.]]], shape=(1, 4, 2), dtype=float32)

