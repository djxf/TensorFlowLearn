#数学运算
import tensorflow as tf

b = tf.fill([2,2], 2.)
a = tf.ones([2,2])

print(a)
print(b)

print(a + b)
print(a - b)
print(a * b)
print(a // b)

print(b % a)


# tf.math ,tf.exp

# a = tf.fill([2,2], )
# #换底公式
# a3 = tf.exp(a)
# print(a3)
print(tf.matmul(a,b))