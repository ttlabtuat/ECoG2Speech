#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import codecs
import configparser
import argparse
import shutil

import numpy as np
from ecog2text.model import ECoG2TextInputConv, ECoG2TextEncoder, ECoG2TextEncoder2, ECoG2TextDecoder, ECoG2TextEncoderTrf, ECoG2TextDecoderTrf, ElecAttention
from ecog2text.preprocess import preprocessingX

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

def parse_args():
    parser = argparse.ArgumentParser(description='Train and Eval')
    parser.add_argument('data_csv')
    parser.add_argument('conf')
    parser.add_argument('--chs', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--tl_type', default=None)
    parser.add_argument('--epochs', default=-1)

    return parser.parse_args()



def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        if l >= ndx + n:
            yield iterable[ndx:ndx + n]

def shuffle_samples(*args):
    zipped = list(zip(*args))
    np.random.shuffle(zipped)
    shuffled = list(zip(*zipped))
    result = []
    for ar in shuffled:
        result.append(np.asarray(ar))
    return result


def train(in_layer, encoder, decoder, ecog, mfcc, text, conf, epochs=-1):
    batch_size = int(conf['batch_size'])
    batch_shuffle = conf.getboolean('batch_shuffle')

    optimizer = tf.keras.optimizers.Adam(lr=float(conf['learning_rate']))


    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)

        return tf.reduce_mean(loss_)

    def loss_mfcc_function(real, pred):
        loss = tf.keras.losses.mean_squared_error(real, pred)
        return tf.reduce_mean(loss)



    train_step_signature = [
      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
      tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    ]

    @tf.function(input_signature=train_step_signature)
    def dec(tar_inp, tar_real):
        return tar_inp, tar_real
    @tf.function
    def train_step(ecog, mfcc, tar_inp, tar_real, input_layer, encoder, decoder):
        loss = 0
        input_seq = tf.convert_to_tensor(ecog)
        tar_inp, tar_real = dec(tar_inp, tar_real)


        with tf.GradientTape() as tape:
            input_seq, _ = input_layer(input_seq)
            mfcc_out, enc_out, enc_hidden  = encoder(input_seq, True)

            dec_hidden = enc_hidden

            mfcc_loss = loss_mfcc_function(mfcc, mfcc_out)
            loss += float(conf['mfcc_penalty']) * mfcc_loss

            for t in range(0, tar_inp.shape[1]):
                #print('loop ', t, tar_inp[:, t])
                dec_input = tf.expand_dims(tar_inp[:, t], axis=-1)
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
                #print(tf.expand_dims(tar_real[:, t], axis=-1), predictions)
                #exit()
                loss_ = loss_function(tf.expand_dims(tar_real[:, t], axis=-1), predictions)
                loss += loss_


            batch_loss = loss

            variables = input_layer.trainable_variables + encoder.trainable_variables + decoder.trainable_variables

            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss



    train_func = tf.function(train_step)
    if epochs == -1:
        epochs = int(conf['epochs'])

    batch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        num = 0

        if batch_shuffle or len(ecog)%batch_size != 0:
            ecog, mfcc, text = shuffle_samples(ecog, mfcc, text)

        for e, m, t in zip(batch(ecog, batch_size), batch(mfcc, batch_size), batch(text, batch_size)):
            tar_inp = t[:, :-1]
            tar_real = t[:, 1:]
            batch_loss = train_func(e, m, tar_inp, tar_real, in_layer, encoder, decoder)
            total_loss += batch_loss


        print('Epoch {} Loss {:.5f}'.format(epoch + 1, total_loss / batch_size))
        batch_losses.append(total_loss / batch_size)

    return batch_losses







def trainTRF(in_layer, encoder, decoder, ecog, mfcc, text, conf, epochs=-1):
    batch_size = int(conf['batch_size'])
    batch_shuffle = conf.getboolean('batch_shuffle')

    optimizer = tf.keras.optimizers.Adam(lr=float(conf['learning_rate']))


    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)

        return tf.reduce_mean(loss_)

    def loss_mfcc_function(real, pred):
        loss = tf.keras.losses.mean_squared_error(real, pred)
        return tf.reduce_mean(loss)



    train_step_signature = [
      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
      tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    ]

    @tf.function(input_signature=train_step_signature)
    def dec(tar_inp, tar_real):
        return tar_inp, tar_real
    @tf.function
    def train_step(ecog, mfcc, tar_inp, tar_real, input_layer, encoder, decoder):
        loss = 0
        input_seq = tf.convert_to_tensor(ecog)
        tar_inp, tar_real = dec(tar_inp, tar_real)
        with tf.GradientTape() as tape:
            input_seq, _ = input_layer(input_seq)
            mfcc_out, enc_out, _  = encoder(input_seq, True)

            predictions = decoder(enc_out, tar_inp, True)
            #print(tar_real, predictions)
            #exit()
            loss += loss_function(tar_real, predictions)

            mfcc_loss = loss_mfcc_function(mfcc, mfcc_out)
            loss += float(conf['mfcc_penalty']) * mfcc_loss


            batch_loss = loss

            variables = input_layer.trainable_variables + encoder.trainable_variables + decoder.trainable_variables

            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss



    train_func = tf.function(train_step)
    if epochs == -1:
        epochs = int(conf['epochs'])

    batch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        num = 0

        if batch_shuffle or len(ecog)%batch_size != 0:
            ecog, mfcc, text = shuffle_samples(ecog, mfcc, text)

        for e, m, t in zip(batch(ecog, batch_size), batch(mfcc, batch_size), batch(text, batch_size)):
            tar_inp = t[:, :-1]
            tar_real = t[:, 1:]
            batch_loss = train_func(e, m, tar_inp, tar_real, in_layer, encoder, decoder)
            total_loss += batch_loss


        print('Epoch {} Loss {:.5f}'.format(epoch + 1, total_loss / batch_size))
        batch_losses.append(total_loss / batch_size)

    return batch_losses





if __name__ == '__main__':
    print('=====passed main=====')
    args = parse_args()
    # preprocess

    xid = os.path.basename(args.conf).replace('config_', '').replace('.ini','')
    subj_task = os.path.basename(args.data_csv).replace('list_', '').replace('.csv', '')
    outdir = 'exp/' + xid + '_' + subj_task
    if not os.path.exists(outdir):
        os.makedirs(outdir)



    conf = configparser.ConfigParser()
    conf.read(args.conf, encoding='utf-8')

    ch_sum = False
    part_sum = False
    band_sum = True
    car = True

    input_attn = True
    if 'INLAYER' in conf:
        if 'input_attn' in conf['INLAYER']:
           input_attn = conf['INLAYER'].getboolean('input_attn')


    if 'FEAT' in conf:
        if 'ch_sum' in conf['FEAT']:
            ch_sum = conf['FEAT'].getboolean('ch_sum')
        if 'band_sum' in conf['FEAT']:
            band_sum = conf['FEAT'].getboolean('band_sum')
        if 'part_sum' in conf['FEAT']:
            part_sum = conf['FEAT'].getboolean('part_sum')


        if 'car' in conf['FEAT']:
            car = conf['FEAT'].getboolean('car')


    TL_surfix = ''

    if args.model == None:
        outmodeldir = outdir + '/model' + TL_surfix
    else:
        outmodeldir = outdir + '/' + args.model + TL_surfix

    if os.path.exists(outmodeldir):
        print('Already exists')
        exit()

    ecog_data, mfcc_data, tokens, tokenizer, _ = preprocessingX(args.data_csv, conf, car, ch_sum, part_sum, band_sum)

    tokens = np.array(tokens)
    print(type(ecog_data), type(mfcc_data), type(tokens))
    print(tokens)


    ConvType = ''

    if band_sum:
        if ch_sum:
            ConvType = 'Error'
        else:
            ConvType = 'Conv2D'
    else:
        if ch_sum and part_sum:
            ConvType = 'Error'
        elif ch_sum:
            ConvType = 'Conv2D'
        else:
            ConvType = 'Conv3D'

    seq_len = ecog_data.shape[1]
    if ConvType == 'Conv3D':
        ch = ecog_data.shape[2]
        band = ecog_data.shape[3]
        encoder_inputs = Input(shape=(ch, band, seq_len, 1))
        inshape = [ch, band]
    elif ConvType == 'Conv2D':
        ch = ecog_data.shape[2]
        encoder_inputs = Input(shape=(ch, seq_len, 1))
        inshape = [ch]
    else:
        print('Error: The convination of ch_sum, part_sum, band_sum is incorrect; ', ch_sum, part_sum, band_sum )
        exit()

    mfcc_dim = mfcc_data.shape[2]

    #prepare input

    if ConvType == 'Conv3D':
        ecog_data_in = ecog_data.transpose(0,2,3,1).reshape(-1, ch, band, seq_len, 1)
    else:
        ecog_data_in = ecog_data.transpose(0,2,1).reshape(-1,ch,seq_len,1)

    decoder_inputs = Input(shape=(None, len(tokens[0])-1))


    # prepare model
    if input_attn:
        in_layer = ElecAttention(conf['INPUTATTN'], True)
    else:
        in_layer = ECoG2TextInputConv(inshape, conf['INPUTCONV'])


    trf_flag_enc = conf['ENCODER'].getboolean('trf_flag')
    trf_flag_dec = conf['DECODER'].getboolean('trf_flag')

    if trf_flag_enc:
        if trf_flag_dec:
            encoder = ECoG2TextEncoderTrf(conf['ENCODER'], mfcc_dim)
            decoder = ECoG2TextDecoderTrf(len(tokenizer) + 2, conf['DECODER'])
        else:
            encoder = ECoG2TextEncoder2(conf['ENCODER'], mfcc_dim)
            decoder = ECoG2TextDecoder(len(tokenizer) + 2, conf['DECODER'])

    else:
        encoder = ECoG2TextEncoder(conf['ENCODER'], mfcc_dim)
        decoder = ECoG2TextDecoder(len(tokenizer) + 2, conf['DECODER'])


    TL_surfix = ''

    if args.model == None:
        outmodeldir = outdir + '/model' + TL_surfix
    else:
        outmodeldir = outdir + '/' + args.model + TL_surfix


    if not args.resume == None:
    # transfer training
        if args.tl_type == None:
            TL_surfix = '_TL00'
            losses = []
            encoder.load_weights(args.resume + '/encoder/weigths')
            decoder.load_weights(args.resume + '/decoder/weights')
            encoder.trainable = False
            decoder.trainable = False
            losses_ = train(in_layer, encoder, decoder, ecog_data_in, mfcc_data, dec_target_data, conf['TRAIN'], char2token, 60)
            losses.extend(losses_)

            encoder.trainable = True
            decoder.trainable = True
            losses_ = train(in_layer, encoder, decoder, ecog_data_in, mfcc_data, dec_target_data, conf['TRAIN'], char2token, 540)
            losses.extend(losses_)


    elif not os.path.exists(outmodeldir):
    # scracth train
        if trf_flag_dec:
            losses = trainTRF(in_layer, encoder, decoder, ecog_data_in, mfcc_data, tokens, conf['TRAIN'])
        else:
            losses = train(in_layer, encoder, decoder, ecog_data_in, mfcc_data, tokens, conf['TRAIN'])


    # save
    if not os.path.exists(outmodeldir):
        os.makedirs(outmodeldir + '/in_layer')
        os.makedirs(outmodeldir + '/encoder')
        os.makedirs(outmodeldir + '/decoder')

        in_layer.save_weights(outmodeldir + '/in_layer/weights')
        encoder.save_weights(outmodeldir + '/encoder/weigths')
        decoder.save_weights(outmodeldir + '/decoder/weights')
        shutil.copy2(args.conf, outmodeldir + '/')

        ofp = codecs.open(outmodeldir + '/loss.txt', 'w', 'utf-8')
        #ofp.write(args.epochs + '\n')
        for i, loss in enumerate(losses):
            ofp.write(str(i) + ',' + str(loss) + '\n')
        ofp.close()


