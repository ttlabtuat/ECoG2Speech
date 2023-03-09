#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Conv1D,Conv2D,Conv3D,Concatenate,SeparableConv2D,Reshape,DepthwiseConv2D,Permute,Dropout

from ecog2text.mytransformer import Encoder, Decoder, Encoder2


class ECoG2TextInputConv(tf.keras.Model):
    def __init__(self, inshape, conf):
        super(ECoG2TextInputConv, self).__init__()
        self.inshape = inshape
        self.kernel_size = int(conf['kernel_size'])
        self.strides = int(conf['strides'])
        self.en_emb = int(conf['en_emb'])
        self.padding = conf['padding'] 
        self.dropout_rate = float(conf['dropout']) 

        if len(self.inshape) == 2:
            self.ch = self.inshape[0]
            self.band = self.inshape[1] 
            self.conv = Conv3D(self.en_emb,
                kernel_size=(self.ch, self.band, self.kernel_size),
                strides=(self.ch, self.band, self.strides),
                padding=self.padding
            )
        elif len(self.inshape) == 1:
            self.ch = self.inshape[0]
            self.conv = Conv2D(self.en_emb,
                kernel_size=(self.ch, self.kernel_size),
                strides=(self.ch, self.strides),
                padding=self.padding)

        
        self.dropout = Dropout(self.dropout_rate)

    def call(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = Reshape((x.shape[len(self.inshape) + 1], self.en_emb))(x)               
        return x, ''


# BLSTM
class ECoG2TextEncoder(tf.keras.Model):
    def __init__(self, conf, outdim):
        super(ECoG2TextEncoder, self).__init__()
    
        self.nLSTM = int(conf['nLSTM'])
        self.en_nunits = int(conf['en_nunits'])
        self.en_dense_nunits = int(conf['en_dense_nunits'])
        self.bidir = conf.getboolean('bidirectional')
        self.en_dropout = float(conf['en_dropout'])
        self.ff_dropout = float(conf['ff_dropout'])  

        self.outdim = outdim

        self.lstms = []
        for num in range(self.nLSTM + 1):
            if self.bidir:
                lstm = Bidirectional(LSTM(
                    self.en_nunits,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.en_dropout
                ),
                merge_mode='mul',
                weights=None
                )
            else:
                lstm = LSTM(
                    self.enc_nunits,
                    return_sequences=True,
                    return_states=True,
                    dropout=self.en_dropout
                )

            self.lstms.append(lstm)

        self.dense = Dense(self.en_dense_nunits, activation='relu')
        self.dropout = Dropout(self.ff_dropout) 
        self.outdense = Dense(self.outdim)

    def call(self, x, _):
        for num in range(self.nLSTM):
            if self.bidir:
                x, _, _, _, _ = self.lstms[num](x)
            else:
                x, _, _, = self.lstms[num](x)


        # to decoder
        if self.bidir:
            xd, fh, fc, bh, bc  = self.lstms[self.nLSTM](x)
            h = Concatenate()([fh, bh])
            c = Concatenate()([fc, bc])
            states = [h, c]
        else:
            xd, h, c = self.lstmes[self.nLSTM](x)
            states = [h, c]


        # to mfcc
        xm = self.dense(x)
        xm = self.dropout(xm)
        xm = self.outdense(xm)

        return xm, xd, states




# Encoder w/ partial Transformer
class ECoG2TextEncoder2(tf.keras.Model):
    def __init__(self, conf, outdim):
        super(ECoG2TextEncoder2, self).__init__()
    
        self.nLSTM = int(conf['nLSTM'])
        self.en_nunits = int(conf['en_nunits'])
        self.en_dense_nunits = int(conf['en_dense_nunits'])
        self.bidir = conf.getboolean('bidirectional')
        self.en_dropout = float(conf['en_dropout'])
        self.ff_dropout = float(conf['ff_dropout'])  
        self.max_len = int(conf['max_len'])
        self.outdim = outdim
        self.enc = Encoder(hopping_num = self.nLSTM, head_num = 10, hidden_dim = 100, dropout_rate = self.en_dropout, max_length = self.max_len)

        if self.bidir:
            lstm = Bidirectional(LSTM(
                self.en_nunits,
                return_sequences=True,
                return_state=True,
                dropout=self.en_dropout
            ),
            merge_mode='mul',
            weights=None
            )
        else:
            lstm = LSTM(
                self.enc_nunits,
                return_sequences=True,
                return_states=True,
                dropout=self.en_dropout
            )
        self.lstm = lstm            

        self.dense = Dense(self.en_dense_nunits, activation='relu')
        self.dropout = Dropout(self.ff_dropout) 
        self.outdense = Dense(self.outdim)



    def call(self, x, training):
        # making mask
        batch_size, length, depth = tf.unstack(tf.shape(x))
        mask = tf.zeros([batch_size, length])
        mask = tf.reshape(mask, [batch_size, 1, 1, length])

        x, _ = self.enc(x, mask, training=training)

        # to decoder
        if self.bidir:
            xd, fh, fc, bh, bc  = self.lstm(x)
            h = Concatenate()([fh, bh])
            c = Concatenate()([fc, bc])
            states = [h, c]
        else:
            xd, h, c = self.lstm(x)
            states = [h, c]


        # to mfcc
        xm = self.dense(x)
        xm = self.dropout(xm)
        xm = self.outdense(xm)

        return xm, xd, states
 

# LSTM
class ECoG2TextDecoder(tf.keras.Model):
    def __init__(self, vocab_size, conf):
        super(ECoG2TextDecoder, self).__init__()
        self.dense_dim = int(conf['dec_emb'])
        self.vocab_size = vocab_size
        self.dec_units = int(conf['de_nunits'])
        self.dec_dropout = float(conf['de_dropout'])
        self.attn_flag = conf.getboolean('attn_flag')

        self.dense1 = Dense(self.dense_dim, activation='relu')
        self.dense2 = Dense(self.vocab_size, activation='softmax')
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, dropout=self.dec_dropout)

        self.token_embedding = tf.keras.layers.Embedding(vocab_size, self.dense_dim)
    
        #self.attention = BahdanauAttention(self.dec_units)
 

    def call(self, x, hidden, enc_output):
        #print('decoder exe')
        #print(hidden)
        x = self.token_embedding(x)
        #print(x)
        x = self.dense1(x)
        #print(x)
        x, h, c = self.lstm(x, initial_state=hidden)
        #print(c)

        attention_weights = tf.zeros([1], tf.int32)
        #if self.attn_flag:
        #    context_vec, attention_weights = self.attention(Concatenate()(hidden), enc_output)
        #    x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)

        x = self.dense2(x) 
        states = [h, c]
                

        return x, states, attention_weights






class ECoG2TextEncoderTrf(tf.keras.Model):
    def __init__(self, conf, outdim):
        super(ECoG2TextEncoderTrf, self).__init__()
    
        self.nLSTM = int(conf['nLSTM'])
        self.nLSTM2 = int(conf['nLSTM2'])
        self.en_nunits = int(conf['en_nunits'])
        self.en_dense_nunits = int(conf['en_dense_nunits'])
        self.bidir = conf.getboolean('bidirectional')
        self.en_dropout = float(conf['en_dropout'])
        self.ff_dropout = float(conf['ff_dropout'])  
        self.max_len = int(conf['max_len'])
        self.outdim = outdim
        self.enc = Encoder(hopping_num = self.nLSTM, head_num = 10, hidden_dim = 100, dropout_rate = self.en_dropout, max_length = self.max_len)
            
        self.dense = Dense(self.en_dense_nunits, activation='relu')
        self.dropout = Dropout(self.ff_dropout) 
        self.outdense = Dense(self.outdim)

        self.enc2 = Encoder(hopping_num = self.nLSTM2, head_num = 10, hidden_dim = 100, dropout_rate = self.en_dropout, max_length = self.max_len)

    def call(self, x, training):
        # making mask
        batch_size, length, depth = tf.unstack(tf.shape(x))
        mask = tf.zeros([batch_size, length])
        mask = tf.reshape(mask, [batch_size, 1, 1, length])


        # to decoder
        x, weights_list = self.enc(x, mask, training=training)
        if self.nLSTM2 > 0:
            y, _ = self.enc2(x, mask, training=training)
        else:
            y = x
        # to mfcc
        xm = self.dense(x)
        xm = self.dropout(xm)
        xm = self.outdense(xm)

        return xm, y, weights_list
 

class ECoG2TextDecoderTrf(tf.keras.Model):
  def __init__(self, vocab_size, conf):
    super(ECoG2TextDecoderTrf, self).__init__()
    self.hopping_num = int(conf['trf_de_hopping_num'])
    self.head_num = int(conf['trf_de_head_num'])
    self.hidden_dim = int(conf['trf_de_hidden_dim'])
    self.dropout_rate = float(conf['de_dropout'])
    print(vocab_size)	
    self.decoder = Decoder(vocab_size=vocab_size, hopping_num=self.hopping_num, head_num=self.head_num, 
          hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate, max_length=int(conf['trf_de_max_length']))


  def _create_enc_self_attention_mask(self, encoder_input: tf.Tensor):
    with tf.name_scope('enc_attention_amsk'):
      batch_size = tf.shape(encoder_input)[0]
      length = tf.shape(encoder_input)[1]
      pad_array = tf.zeros([batch_size, length], dtype=tf.bool)
      return tf.reshape(pad_array, [batch_size, 1, 1, length])

  def _create_dec_self_attention_mask(self, decoder_input: tf.Tensor):
    with tf.name_scope('dec_self_attention_mask'):
     #batch_size, length = tf.unstack(tf.shape(decoder_input))
      batch_size = tf.shape(decoder_input)[0]
      length = tf.shape(decoder_input)[1]

      pad_array = tf.equal(decoder_input, 0)  # [batch_size, m_length]
      pad_array = tf.reshape(pad_array, [batch_size, 1, 1, length])

      autoregression_array = tf.logical_not(
      	tf.compat.v1.matrix_band_part(tf.ones([length, length], dtype=tf.bool), -1, 0))  # 下三角が False
      autoregression_array = tf.reshape(autoregression_array, [1, 1, length, length])

      return tf.logical_or(pad_array, autoregression_array)


  def call(self, encoder_output: tf.Tensor, decoder_input: tf.Tensor, training: bool) -> tf.Tensor:
    enc_self_attention_mask = self._create_enc_self_attention_mask(encoder_output)
    dec_self_attention_mask = self._create_dec_self_attention_mask(decoder_input)
		
    decoder_output = self.decoder(
      decoder_input,
      encoder_output,
      self_attention_mask=dec_self_attention_mask,
      enc_dec_attention_mask=enc_self_attention_mask,
      training=training,
    )

    return decoder_output



class ElecAttention(tf.keras.Model):
  def __init__(self, conf, training):
    super(ElecAttention, self).__init__()	
    self.strides = int(conf['strides'])
    self.outdim = int(conf['en_emb'])

    self.maxelec = int(conf['maxelec'])
    self.hopping_num = int(conf['hopping_num'])
    self.head_num = int(conf['head_num'])
    self.hidden_dim = int(conf['hidden_dim'])
    self.dropout_rate = float(conf['dropout_rate'])	

    self.encoder = Encoder2(self.hopping_num, self.head_num, self.hidden_dim, self.dropout_rate, self.maxelec + 1)
    self.dense = tf.keras.layers.Dense(self.outdim, use_bias=False)
    self.training = training
    self.dropout = Dropout(self.dropout_rate)

  def call(self, x):

    weights_list_by_time = []
    for i in range(int(x.shape[2]/self.strides)):
      y_ = x[:,:,i*self.strides:(i+1)*self.strides,0]
      batch_size, length ,depth = tf.unstack(tf.shape(y_))
      
      mask = tf.ones([batch_size, self.maxelec - length])
      z = tf.zeros([batch_size, length + 1])
      mask = tf.keras.layers.Concatenate(axis=1)([z, mask])
      mask = tf.reshape(mask, [batch_size, 1, 1, self.maxelec+1])

      z = tf.zeros([batch_size, self.maxelec - length, self.strides])
      cls = tf.ones([batch_size, 1, self.strides])
      y_ = tf.keras.layers.Concatenate(axis=1)([cls, y_])
      y_ = tf.keras.layers.Concatenate(axis=1)([y_, z])      

      encoder_out, weights_list = self.encoder(y_, mask, self.training)
      weights_list_by_time.append(weights_list)
      #print(len(weights_list_by_time))
      cls_vec = encoder_out[:, 0, :]

      y_ = self.dense(cls_vec)
      y_ = tf.expand_dims(y_, axis=1)
      if not i == 0:
        y = tf.keras.layers.Concatenate(axis=1)([y, y_])
      else:
        y = y_

      if i == int(x.shape[2]/self.strides) - 1:
        p = tf.zeros_like(y_)
        y = tf.keras.layers.Concatenate(axis=1)([y, p])

    return y, weights_list_by_time


