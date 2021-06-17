#fill and copy
#pad, tile, broadcast_to

import tensorflow as tf


a = tf.reshape(tf.range(9), [3, 3])
print(a)
a = tf.pad(a, [[1, 1], [1, 1]])
print(a)

#image padding


#tile:
    # * repeat data along dim n times
    # * [a, b, c], 2
    # * ---> [a, b, c, a, b, c]
# broadcast_to只是一个隐式的复制，并不是真实的复制。而tile是真实的。

print(tf.tile(a, [2, 1]))

a2 = tf.constant([1,2])
print(a2.shape)
print(tf.broadcast_to(a2, [4]).shape)