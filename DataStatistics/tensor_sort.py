# tensor sort

import tensorflow as tf
a = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)
print(a)

print(tf.sort(a))

print(tf.argsort(a))

#top k accuracy:
prob = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])
target = tf.constant([2, 0])
k_b = tf.math.top_k(prob, 3).indices

print(k_b)

k_b = tf.transpose(k_b, [1, 0])
print(k_b)

target = tf.broadcast(target, [3, 2])

#@param output: [b, N]
#@param target: [b]
def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, prem=[1,0])
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k / batch_size)
        res.append(acc)
    return res
