#1 梯度爆炸是如何产生的？
#2 如何解决梯度爆炸？


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

# x: [60K, 28, 28]
# y: [60k]
(x, y), _ = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)
#print(x.shape, y.shape, x.dtype, y.dtype)

print(tf.reduce_min(x), tf.reduce_min(y))
print(tf.reduce_max(x), tf.reduce_max(y))

train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
#sample[0]: [128, 28, 28],sample[1] : [128]
print('batch: ', sample[0].shape, sample[1].shape)

# [b, 784] => [b, 256] => [b, 10]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.01))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.01))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.01))
b3 = tf.Variable(tf.zeros([10]))


#print(y_onehot)

#学习率
lr = 1e-3
for epoch in range(10):
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2 @ w3 + b3  # out: [b, 10]

            # 对分类样本进行one_hot编码
            y_onehot = tf.one_hot(y, depth=10)  # y: [60k, 10]


            # loss function 损失函数 mse = mean(y - out)^2
            loss = tf.square(y_onehot - out)  # [60k, 10] - [60K, 10]
            loss = tf.reduce_mean(loss)  # loss?

            # compute gradients
            grads = tape.gradient(loss, [w1,b1,w2,b2,w3,b3]) # 这时候loss 的梯度？？？

            #根据梯度更新参数的值
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])
            w2.assign_sub(lr * grads[2])
            b2.assign_sub(lr * grads[3])
            w3.assign_sub(lr * grads[4])
            b3.assign_sub(lr * grads[5])

            if step % 100 == 0:
                print(step, 'loss: ', float(loss))


print(w3)
print(b3)






