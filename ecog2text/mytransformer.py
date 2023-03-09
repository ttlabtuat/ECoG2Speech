#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import math



class FeedForwardNetwork(tf.keras.models.Model):
    '''
    Position-wise Feedforward Neural Network for Transformer
    '''
    def __init__(self, hidden_dim: int, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.filter_dense_layer = tf.keras.layers.Dense(hidden_dim * 4, use_bias=True,
                                                        activation=tf.nn.relu, name='filter_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=True, name='output_layer')
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input: tf.Tensor, training: bool) -> tf.Tensor:
        '''
        Apply FeedForwardNetwork
        :param input: shape = [batch_size, length, hidden_dim]
        :return: shape = [batch_size, length, hidden_dim]
        '''
        tensor = self.filter_dense_layer(input)
        tensor = self.dropout_layer(tensor, training=training)
        return self.output_dense_layer(tensor)


class ResidualNormalizationWrapper(tf.keras.models.Model):
    '''
    - Layer Normalization
    - Dropout
    - Residual Connection
    '''
    def __init__(self, layer: tf.keras.layers.Layer, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.layer_normalization = LayerNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input: tf.Tensor, training: bool, *args, **kwargs) -> tf.Tensor:
        tensor = self.layer_normalization(input)
        tensor = self.layer(tensor, training=training, *args, **kwargs)
        tensor = self.dropout_layer(tensor, training=training)
        return input + tensor

class ResidualNormalizationWrapper2(tf.keras.models.Model):
    '''
    - Layer Normalization
    - Dropout
    - Residual Connection
    '''
    def __init__(self, layer: tf.keras.layers.Layer, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.layer_normalization = LayerNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input: tf.Tensor, training: bool, *args, **kwargs) -> tf.Tensor:
        tensor = self.layer_normalization(input)
        tensor, weights = self.layer(tensor, training=training, *args, **kwargs)
        tensor = self.dropout_layer(tensor, training=training)
        return input + tensor, weights




class LayerNormalization(tf.keras.layers.Layer):
    '''
   	LayerNormalization　
    '''
    def build(self, input_shape: tf.TensorShape) -> None:
        hidden_dim = input_shape[-1]
        self.scale = self.add_weight('layer_norm_scale', shape=[hidden_dim],
                                     initializer=tf.ones_initializer())
        self.bias = self.add_weight('layer_norm_bias', [hidden_dim],
                                    initializer=tf.zeros_initializer())
        super().build(input_shape)

    def call(self, x: tf.Tensor, epsilon: float = 1e-6) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.compat.v1.rsqrt(variance + epsilon)

        return norm_x * self.scale + self.bias




class AddPositionalEncoding(tf.keras.layers.Layer):

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        fl_type = inputs.dtype
        batch_size, max_length, depth = tf.unstack(tf.shape(inputs))

        depth_counter = tf.range(depth)
        depth_matrix = tf.tile(tf.expand_dims(depth_counter, 0), [max_length, 1])
        depth_matrix = tf.pow(10000.0, tf.cast(depth_matrix / depth, fl_type))

        phase = tf.cast(tf.range(depth) % 2, fl_type) * math.pi / 2
        phase_matrix = tf.tile(tf.expand_dims(phase, 0), [max_length, 1])

        pos_counter = tf.range(max_length)
        pos_matrix = tf.cast(tf.tile(tf.expand_dims(pos_counter, 1), [1, depth]), fl_type)
        
        positional_encoding = tf.sin(pos_matrix / depth_matrix + phase_matrix)
        # [batch_size, max_length, depth]

        positional_encoding = tf.tile(tf.expand_dims(positional_encoding, 0), [batch_size, 1, 1])

        return inputs + positional_encoding



class MultiheadAttention(tf.keras.models.Model):
    
    def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='output_dense_layer')
        self.attention_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    

    def call(
        self,
        input: tf.Tensor,
        memory: tf.Tensor,
        attention_mask: tf.Tensor,
        training: bool,        

    ) -> tf.Tensor:

        q = self.q_dense_layer(input) # [batch_size, q_length, hidden_dim]
        k = self.k_dense_layer(memory) # [batch_size, m_length, hidden_dim]
        v = self.v_dense_layer(memory)

        q = self._split_head(q) # [batch_size, head_num, q_length, hidden_dim/head_num]
        k = self._split_head(k) # [batch_size, head_num, m_length, hidden_dim/head_num]
        v = self._split_head(v) # [batch_size, head_num, m_length, hidden_dim/head_num]

        depth = self.hidden_dim // self.head_num
        q *= depth ** -0.5 # for scaled dot production

        logit = tf.matmul(q, k, transpose_b=True) # [batch_size, head_num, q_length, k_length]
        logit += tf.compat.v1.to_float(attention_mask) * input.dtype.min 

        attention_weight = tf.nn.softmax(logit, name='attention_weight')
        attention_weight = self.attention_dropout_layer(attention_weight, training=training)

        attention_output = tf.matmul(attention_weight, v) # [batch_size, head_num, q_length, hidden_dim/head_num]
        attention_output = self._combine_head(attention_output) # [batch_size, q_length, hidden_dim]

        return self.output_dense_layer(attention_output), attention_weight


    def _split_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('split_head'):
            batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [batch_size, length, self.head_num, self.hidden_dim // self.head_num])
            return tf.transpose(x, [0, 2, 1, 3])
    
        
    def _combine_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.hidden_dim])






class SelfAttention(MultiheadAttention):

    def call(
        self,
        input: tf.Tensor,
        attention_mask: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        return super().call(
            input=input,
            memory=input,
            attention_mask=attention_mask,
            training=training,
        )





class Encoder(tf.keras.models.Model):
    '''
    Transformer Encoder
    '''
    def __init__(
        self,
        hopping_num: int,
        head_num: int,
        hidden_dim: int,
        dropout_rate: float,
        max_length: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.add_positional_encoding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            #attention_layer_ = SelfAttention_(hidden_dim, head_num, dropout_rate, name='self_attention_')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper2(attention_layer, dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
            ])
        self.output_normalization = LayerNormalization()


    def call(
        self,
        x: tf.Tensor,
        self_attention_mask: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
       
        x = self.add_positional_encoding(x)
        query = self.input_dropout_layer(x, training=training)

        weights_list = []
        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query, weights = attention_layer(query, attention_mask=self_attention_mask, training=training)
                query = ffn_layer(query, training=training)
                #print(weights)
                weights_list.append(weights)
                #print(len(weights_list))
        return self.output_normalization(query), weights_list




class Decoder(tf.keras.models.Model):
    '''
		Transformer Decoder   	
    '''
    def __init__(
            self,
            vocab_size: int,
            hopping_num: int,
            head_num: int,
            hidden_dim: int,
            dropout_rate: float,
            max_length: int,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        #self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            self_attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            enc_dec_attention_layer = MultiheadAttention(hidden_dim, head_num, dropout_rate, name='enc_dec_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper2(self_attention_layer, dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper2(enc_dec_attention_layer, dropout_rate, name='enc_dec_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
            ])
        self.output_normalization = LayerNormalization()
        self.output_dense_layer = tf.keras.layers.Dense(vocab_size, use_bias=False)

    def call(
            self,
            input: tf.Tensor,
            encoder_output: tf.Tensor,
            self_attention_mask: tf.Tensor,
            enc_dec_attention_mask: tf.Tensor,
            training: bool,
    ) -> tf.Tensor:
        '''
        :param input: shape = [batch_size, length]
        :param training: True when training phase
        :return: shape = [batch_size, length, hidden_dim]
        '''
        # [batch_size, length, hidden_dim]
        #print('hogehogehoge', input)
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)

        for i, layers in enumerate(self.attention_block_list):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query, _ = self_attention_layer(query, attention_mask=self_attention_mask, training=training)
                query, _ = enc_dec_attention_layer(query, memory=encoder_output, attention_mask=enc_dec_attention_mask, training=training)
                query = ffn_layer(query, training=training)

        query = self.output_normalization(query)  # [batch_size, length, hidden_dim]
        return self.output_dense_layer(query)  # [batch_size, length, vocab_size]

        #return input



class Encoder2(tf.keras.models.Model):
    '''
    Encoder for Electrode attention 
    '''
    def __init__(
            self,
            hopping_num: int,
            head_num: int,
            hidden_dim: int,
            dropout_rate: float,
            max_length: int,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.in_dense =  tf.keras.layers.Dense(hidden_dim, use_bias=False)
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        
        for _ in range(hopping_num):
            attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper2(attention_layer, dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
            ])
        self.output_normalization = LayerNormalization()

    def call(
            self,
            input: tf.Tensor,
            self_attention_mask: tf.Tensor,
            training: bool,
    ) -> tf.Tensor:
        '''
        モデルを実行します

        :param input: shape = [batch_size, length]
        :param training: 学習時は True
        :return: shape = [batch_size, length, hidden_dim]
        '''
        # [batch_size, length, hidden_dim]
        embedded_input = self.in_dense(input)
        #embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)
        weights_list = []
        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query, weights = attention_layer(query, attention_mask=self_attention_mask, training=training)
                query = ffn_layer(query, training=training)
                #print(weights)
                weights_list.append(weights)
                #print(len(weights))
        # [batch_size, length, hidden_dim]
        return self.output_normalization(query), weights_list




