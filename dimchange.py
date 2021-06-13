###维度变换
import tensorflow as tf

print("Hello tensorflow")



#View:视图的概念
#[b, 28, 28]
#[b, 28*28]
#----> [b, 2, 14*28]
#----> [b, 28, 28, 1]
### tf.random.normal()函数
###
###Args:
    # shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    # mean: A Tensor or Python value of type `dtype`, broadcastable with `stddev`.
    #   The mean of the normal distribution.
    # stddev: A Tensor or Python value of type `dtype`, broadcastable with `mean`.
    #   The standard deviation of the normal distribution.
    # dtype: The type of the output.
    # seed: A Python integer. Used to create a random seed for the distribution.
    #   See
    #   `tf.random.set_seed`
    #   for behavior.
    # name: A name for the operation (optional).
# a = tf.random.normal([4,28,28,3], mean=100)
# print(a.shape)
# print(a.ndim)
# a2 = tf.reshape(a, [4, 28 * 28, 3])
# print(a2)

b = tf.random.normal([2,2], mean=1)
print(b)
print(tf.transpose(b,perm=[1,0]))
b2 = tf.expand_dims(b, axis=1)
print(b2)


#Broadcasting
#How to understand?     When it has no axis？
#
# why is broadcasting?
# 1 for real demanding
# 2 memory consumption