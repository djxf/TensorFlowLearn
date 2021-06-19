#clip_by_value
import tensorflow as tf

a = tf.range(9)
print(a)

print(tf.maximum(a, 2))
print(tf.minimum(a, 8))

print(tf.maximum(tf.minimum(a, 8), 2))

#clip_by_value
print(tf.clip_by_value(a, 2, 8))

#relu func
tf.nn.relu(a)
tf.maximum(a, 0)

a1 = tf.random.normal([2, 2], mean=10)
print(a1)
print(tf.norm(a1))
a2 = tf.clip_by_norm(a1, 15)
print(a2)

tf.norm(a2)

#Gradient clipping
    # *gradient exploding or vanishing(梯度爆炸 梯度消失)
    # *set lr = 1
    # *new_gradient, total_norm = tf.clip_by_global_norm()