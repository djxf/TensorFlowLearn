##索引与切片
import tensorflow as tf
import numpy as np

a = tf.range(10)
print(a)
print(a[-1:])
print(a[:])
b = tf.ones([4,3,3])
#print(b)
#print(tf.gather(b,axis=0,indices=[1,2]))
#print(tf.gather_nd(a, [0]))
print(tf.boolean_mask(b, mask=[[True,True,False],[False,False,True],[False,False,True],[False,False,True]]))
