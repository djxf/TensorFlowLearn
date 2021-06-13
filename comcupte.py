import numpy as np
import tensorflow as tf

def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 计算损失/误差
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        w_gradient += (2 / N) * x * ((w_current * x + b_current) - y)
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(points, start_b, start_w, learn_rate, num_iterations):
    b = start_b
    w = start_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, points, learn_rate)
    return [b, w]

print("Hello tensorflow")
a = tf.ones([2, 5, 5, 3])
#print(a)


##下标还是从0开始
print(a[2,1,1])

##索引与切片
