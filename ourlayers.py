import math
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras


class MGATLayer(keras.layers.Layer):
    def __init__(
        self,
        att_embedding_size=8, 
        head_num=8, 
        activation=tf.nn.relu,
        reduction='concat', 
        use_bias=True, 
        **kwargs):
        """
            Input:
                  att_embedding_size: hidden shape
                  head_num: the number of heads
                  activation: activation function
                  reduction: concat or mean operation
                  use_bias: True/False a flag for deciding if the weight bias is used
        """
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')

        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.activation = activation
        self.act = activation
        self.reduction = reduction
        self.use_bias = use_bias

        super(MGATLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        X, A = input_shape
        embedding_size = int(X[-1])
        self.weight = self.add_weight(
            name='weight', 
            shape=[embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )

        self.att_self_weight = self.add_weight(
            name='att_self_weight',
            shape=[1, self.head_num, self.att_embedding_size],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )
        self.att_neighs_weight = self.add_weight(
            name='att_neighs_weight',
            shape=[1, self.head_num, self.att_embedding_size],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )

        if self.use_bias:
            self.bias_weight = self.add_weight(
                name='bias', shape=[1, self.head_num, self.att_embedding_size],
                dtype=tf.float32,
                initializer=keras.initializers.Zeros()
            )
     
        super(MGATLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        X, A = inputs
        if keras.backend.ndim(X) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(X)))

        features = tf.matmul(X, self.weight,)  # None F'*head_num
        features = tf.reshape(
            features, [-1, self.head_num, self.att_embedding_size])  # None head_num F'

        attn_for_self = tf.reduce_sum(
            features * self.att_self_weight, axis=-1, keepdims=True)  # None head_num 1
        attn_for_neighs = tf.reduce_sum(
            features * self.att_neighs_weight, axis=-1, keepdims=True)

        dense = tf.transpose(
            attn_for_self, [1, 0, 2]) + tf.transpose(attn_for_neighs, [1, 2, 0])
        dense = tf.nn.leaky_relu(dense, alpha=0.2)

        embs = []
        for i in range(len(A)):
            mask = -10e9 * (1.0 - tf.sign(A[i]))
            dense_cur = dense + tf.expand_dims(mask, axis=0)  

            normalized_att_scores = tf.nn.softmax(
                dense_cur, axis=-1, )  # head_num None(F) None(F)
          
            result = tf.matmul(normalized_att_scores, tf.transpose(features, [1, 0, 2]))  # head_num None F D   [8,2708,8] [8,2708,3]
            result = tf.transpose(result, [1, 0, 2])  # None head_num attsize

            if self.use_bias:
                result += self.bias_weight

            # head_num Node embeding_size
            if self.reduction == "concat":
                result = tf.concat(
                    tf.split(result, self.head_num, axis=1), axis=-1)
                result = tf.squeeze(result, axis=1)
            else:
                result = tf.reduce_mean(result, axis=1)

            if self.act:
                result = self.activation(result)
            embs.append(tf.expand_dims(result,axis=1))

        embs = tf.concat(embs, axis=1)

        return  embs 


class HXGATlayer(keras.layers.Layer):
    def __init__(
        self, 
        att_embedding_size=8, 
        hete_att_size=8, 
        activation=tf.nn.relu,
        hete_num=2,
        reduction='concat', 
        divide=False,
        use_bias=False, 
        **kwargs
    ):
        """
            Input:
                  att_embedding_size: hidden shape
                  head_num: the number of heads
                  activation: activation function
                  reduction: concat or mean operation for a single GAT layer
                  use_bias: True/False a flag for deciding if the weight bias is used
        """
        if  hete_num <= 0:
            raise ValueError('head_num must be a int > 0')
        
        self.activation = activation
        self.act = activation
        self.reduction = reduction
        self.use_bias = use_bias
        self.divide = divide

        if self.divide:
            self.gats = []
            for i in range(hete_num):
                g = GATLayer(
                    att_embedding_size=att_embedding_size, 
                    head_num=1, 
                    activation=self.activation, 
                    use_bias=False,
                    reduction='mean'
                )
                self.gats.append(g)
        else:
            self.gats = MGATLayer(
                        att_embedding_size=att_embedding_size,
                        head_num=1, 
                        activation=self.activation, 
                        use_bias=False,
                        reduction='mean'
                    )

        self.mixer = Mixer(hidden_size=hete_att_size, activation=activation, use_bias=True)

        super(HXGATlayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):

        x, adjs = inputs
        if self.divide:
            embs = []
            for i in range(len(self.gats)):
                g = self.gats[i]([x, adjs[i]])
                embs.append(tf.expand_dims(g, axis=1))

            embs = tf.concat(embs, axis=1)

        else:
            embs = self.gats([x, adjs])

        result = self.mixer(embs)

        return result


class Mixer(keras.layers.Layer):
    def __init__(self, hidden_size, activation=tf.nn.relu, use_bias=False):
        super(Mixer, self).__init__()

        self.hidden_size = hidden_size
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", shape=[int(input_shape[-1]), self.hidden_size])

        self.att_weight = self.add_weight(
            name='att_weight', 
            shape=[self.hidden_size],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )

        if self.use_bias:
            self.bias_weight = self.add_weight(
                name='bias', 
                shape=[self.hidden_size],
                dtype=tf.float32,
                initializer=keras.initializers.Zeros()
            )

    def call(self, inputs):
        if inputs.shape[0] < inputs.shape[1]:
            inputs = tf.transpose(inputs, [1, 0, 2]) #(num_he,N,D) -> (N,num_he,D)

        features = tf.tensordot(inputs, self.kernel, axes=1) + self.bias_weight  # (N,num_he,D)*(D,D')->(N,num_he,D')

        att_w = tf.tensordot(tf.nn.leaky_relu(features, alpha=0.2),self.att_weight,axes=1) # (N,num_he,D')*(D',) -> (N,num_he)
        nor_att_w = tf.nn.softmax(att_w, axis=-1) # (N,num_he)

        emb = tf.reduce_sum(inputs * tf.expand_dims(nor_att_w, -1), 1) #(N,D)

        return  emb


class RepLayer(keras.layers.Layer):
    def __init__(self, num_outputs,activation=tf.nn.relu):
        """ representation layer
        input:
              num_outputs:  hidden shape
              activation: the activation function of representation layer
        Output:
              The representation of current layer
        """
        super(RepLayer,self).__init__()
        self.num_outputs = num_outputs
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[int(input_shape[-1]), self.num_outputs],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )

        self.bias = self.add_weight("bias", shape=[self.num_outputs], initializer=keras.initializers.Zeros())
       
    def call(self,input):
        output = tf.matmul(input, self.kernel) + self.bias
        output = self.activation(output)        
        return output


class GATLayer(keras.layers.Layer):
    def __init__(
        self, 
        att_embedding_size=8, 
        head_num=8, 
        activation=tf.nn.relu,
        reduction='concat', 
        use_bias=True, 
        **kwargs
    ):
        """
            Input:
                  att_embedding_size: hidden shape
                  head_num: the number of heads
                  activation: activation function
                  reduction: concat or mean operation
                  use_bias: True/False a flag for deciding if the weight bias is used
        """
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')

        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.activation = activation
        self.act = activation
        self.reduction = reduction
        self.use_bias = use_bias
        super(GATLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        X, A = input_shape
        embedding_size = int(X[-1])

        self.weight = self.add_weight(
            name='weight', 
            shape=[embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )

        self.att_self_weight = self.add_weight(
            name='att_self_weight',
            shape=[1, self.head_num, self.att_embedding_size],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )
        self.att_neighs_weight = self.add_weight(
            name='att_neighs_weight',
            shape=[1, self.head_num, self.att_embedding_size],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )

        if self.use_bias:
            self.bias_weight = self.add_weight(
                name='bias', 
                shape=[1, self.head_num, self.att_embedding_size],
                dtype=tf.float32,
                initializer=keras.initializers.Zeros()
            )

        super(GATLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        X, A = inputs
        if keras.backend.ndim(X) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(X)))

        features = tf.matmul(X, self.weight,)  # None F'*head_num
        features = tf.reshape(
            features, [-1, self.head_num, self.att_embedding_size])  # None head_num F'

        attn_for_self = tf.reduce_sum(
            features * self.att_self_weight, axis=-1, keepdims=True)  # None head_num 1
        attn_for_neighs = tf.reduce_sum(
            features * self.att_neighs_weight, axis=-1, keepdims=True)

        dense = tf.transpose(
            attn_for_self, [1, 0, 2]) + tf.transpose(attn_for_neighs, [1, 2, 0])
        dense = tf.nn.leaky_relu(dense, alpha=0.2)
        mask = -10e9 * (1.0 - tf.sign(A))
        dense += tf.expand_dims(mask, axis=0)  
      
        self.normalized_att_scores = tf.nn.softmax(
            dense, axis=-1, )  

        result = tf.matmul(self.normalized_att_scores, tf.transpose(features, [1, 0, 2]))  
        result = tf.transpose(result, [1, 0, 2])  

        if self.use_bias:
            result += self.bias_weight

        # head_num Node embeding_size
        if self.reduction == "concat":
            result = tf.concat(
                tf.split(result, self.head_num, axis=1), axis=-1)
            result = tf.squeeze(result, axis=1)

        else:
            result = tf.reduce_mean(result, axis=1)

        if self.act:
            result = self.activation(result)

        return result
