import math
from keras.layers import *
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
import keras.backend as K
import tensorflow as tf
def eca_layer(inputs_tensor=None,num=None,gamma=2,b=1,**kwargs):
    """
    ECA-NET
    :param inputs_tensor: input_tensor.shape=[batchsize,h,w,channels]
    :param num:
    :param gamma:
    :param b:
    :return:
    """
    channels = K.int_shape(inputs_tensor)[-1]
    t = int(abs((math.log(channels,2)+b)/gamma))
    k = t if t%2 else t+1
    x_global_avg_pool = GlobalAveragePooling2D()(inputs_tensor)
    x = Reshape((channels,1))(x_global_avg_pool)
    x = Conv1D(1,kernel_size=k,padding="same",name="eca_conv1_" + str(num))(x)
    x = Activation('sigmoid', name='eca_conv1_relu_' + str(num))(x)  #shape=[batch,chnnels,1]
    x = Reshape((1, 1, channels))(x)
    output = multiply([inputs_tensor,x])
    return output
