#
import tensorflow as tf

#argmax/argmin
#argmax: 类似每一列的最大值。
a = tf.random.normal([3, 3], mean=10, stddev=3)
print(a)

print(tf.argmax(a))


b = tf.constant([[2, 20, 30, 3, 6],
                  [3, 11, 16, 1, 8],
                  [14, 45, 23, 5, 27]])
print("argmax(b, 0): ")
print(tf.argmax(b, 0))
print("argmax(b, 1): ")
print(tf.argmax(b, 1)) #axis


#tf.equals
a1 = tf.constant([0, 1, 3, 4, 4])
b1 = tf.range(5)

print(tf.equal(a1, b1))

res = tf.equal(a1, b1)
print(tf.reduce_sum(tf.cast(res, dtype=tf.int32)))

#tf.unique:去除重复元素
a3 = tf.range(100)
print(tf.unique(a3))

b3 = tf.constant([4,2,2,4,3])
print(tf.unique(b3))
# print(tf.gather())


#范数： tf.norm,一范数，二范数，无穷范数。向量的范数，矩阵的范数。
#一范数：绝对值的和
#二范数：平方和开根号
#无穷范数：最大值。

a4 = tf.ones([2,3])
print(tf.norm(a4))


