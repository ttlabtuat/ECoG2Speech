#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import codecs
import subprocess
import configparser
import argparse

import numpy as np
from ecog2text.model import ECoG2TextInputConv, ECoG2TextEncoder, ECoG2TextEncoder2, ECoG2TextDecoder, ECoG2TextEncoderTrf, ECoG2TextDecoderTrf, ElecAttention
from ecog2text.preprocess import preprocessingX

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



def parse_args():
    parser = argparse.ArgumentParser(description='Train and Eval')
    parser.add_argument('data_csv')
    parser.add_argument('model')
    parser.add_argument('--chs', default=None)
    parser.add_argument('--car', action='store_true')   

    return parser.parse_args()     




def evaluate(input_seq, input_layer, encoder, decoder, tokenizer):
    input_seq = tf.convert_to_tensor(input_seq)
    input_seq, weights_out1 = input_layer(input_seq)
    mfcc_out, enc_out, enc_hidden = encoder(input_seq, False) 
    
    dec_hidden = enc_hidden
    
    decoder_input = tf.expand_dims([len(tokenizer)], 0)
    output = decoder_input
   

    max_length = 17
    for i in range(max_length):
#      print(decoder_input)
      predictions, dec_hidden, _ = decoder(decoder_input, dec_hidden, enc_out)
      predictions = predictions[:, -1:, :]

      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
#      print('1.predicted_id: ', i, predicted_id)
      decoder_input = tf.expand_dims(predicted_id[0], 0)
      if predicted_id == len(tokenizer) + 1:
        return tf.squeeze(output, axis=0), mfcc_out, weights_out1

     
      output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), mfcc_out, weights_out1
 


def smapLSTM(input_seq__, in_layer, encoder, decoder, tokenizer):
    input_seq = tf.Variable(input_seq__, dtype=float)
    i0 = Input(shape=input_seq[0].shape)
    #i0 = Input(shape=input_seq.shape)
    x1 = in_layer(i0)
    
    grad_model = Model(i0, x1)
    class_output = 0
    with tf.GradientTape() as tape:
        tape.watch(input_seq)
        input_seq_, _ = grad_model(input_seq)
        mfcc_out, enc_out, enc_hidden = encoder(input_seq_, False)

        dec_hidden = enc_hidden
    
        decoder_input = tf.expand_dims([len(tokenizer)], 0)
        output = decoder_input        

        max_length = 17
        for i in range(max_length):        
            #print(i, decoder_input)
            predictions, dec_hidden, _ = decoder(decoder_input, dec_hidden, enc_out)
            predictions = predictions[:, -1, :]
            
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            decoder_input = tf.expand_dims(predicted_id,0)
            predicted_score = predictions[:,predicted_id]
            if predicted_id == len(tokenizer) + 1:
                break

            class_output += predicted_score
            #print('2. predictions: ', i, predictions)
            output = tf.concat([output, [predicted_id]], axis=-1)

    grads = tape.gradient(class_output, input_seq)
    grads = tf.reduce_max(grads, axis=-1)
    
    min_val, max_val = np.min(grads), np.max(grads)
    smap = (grads - min_val) / (max_val - min_val)

    return smap





def evaluateTRF(input_seq, input_layer, encoder, decoder, tokenizer):
    input_seq = tf.convert_to_tensor(input_seq)
    input_seq, weights_out1 = input_layer(input_seq)
    mfcc_out, enc_out, weights_out2 = encoder(input_seq, False) 

    decoder_input = [len(tokenizer)]
    output = tf.expand_dims(decoder_input, 0)


    max_length = 17
    for i in range(max_length):
      predictions = decoder(enc_out, output)
      predictions = predictions[:, -1:, :]

      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

      if predicted_id == len(tokenizer) + 1:
        return tf.squeeze(output, axis=0), mfcc_out, weights_out1, weights_out2

      #print('1.predicted_id: ', i, predicted_id)
      output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), mfcc_out, weights_out1, weights_out2
 



def smapTRF(input_seq__, in_layer, encoder, decoder, tokenizer):
    input_seq = tf.Variable(input_seq__, dtype=float)
    i0 = Input(shape=input_seq[0].shape)
    #i0 = Input(shape=input_seq.shape)
    x1 = in_layer(i0)
    
    grad_model = Model(i0, x1)
    class_output = 0
    with tf.GradientTape() as tape:
        tape.watch(input_seq)
        input_seq_, _ = grad_model(input_seq)
        mfcc_out, enc_out, _ = encoder(input_seq_, False)
        decoder_input = [len(tokenizer)]
        output = tf.expand_dims(decoder_input, 0)
        
        max_length = 17
        for i in range(max_length):        
            predictions = decoder(enc_out, output)
            predictions = predictions[:, -1, :]
            
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            predicted_score = predictions[:,predicted_id]
            if predicted_id == len(tokenizer) + 2:
                break

            class_output += predicted_score
            #print('2. predictions: ', i, predictions)
            output = tf.concat([output, [predicted_id]], axis=-1)

    grads = tape.gradient(class_output, input_seq)
    grads = tf.reduce_max(grads, axis=-1)
    
    min_val, max_val = np.min(grads), np.max(grads)
    smap = (grads - min_val) / (max_val - min_val)

    return smap



def calc_cer(ref, hyp):
    if not len(ref) == len(hyp):
        print('Error: different length')
        return -9999,-9999,-9999,-9999

    random_ = random.randint(10000,99999)
    ref_text_name = "ref"+str(random_)+".txt"
    hyp_text_name = "hyp"+str(random_)+".txt"
    log_cer_name = "log_cer"+str(random_)+".txt"
    ofp1 = codecs.open(ref_text_name, 'w', 'utf-8')
    ofp2 = codecs.open(hyp_text_name, 'w', 'utf-8')

    cnt = 0
    for r, h in zip(ref, hyp):
        ref_ = ''
        hyp_ = ''
        for r_ in r:
            ref_ += ' ' + r_
        for h_ in h:
            hyp_ += ' ' + h_

        ref_ = ref_.strip()
        hyp_ = hyp_.strip()

        ofp1.write(ref_ + ' (A' + str(cnt).zfill(4) + '-log)\n')
        ofp2.write(hyp_ + ' (A' + str(cnt).zfill(4) + '-log)\n')
        cnt += 1
    ofp1.close()
    ofp2.close()

    cmd = './bin/sclite -r '+ref_text_name+' -h '+hyp_text_name+' trn -i rm -o all stdout > '+log_cer_name
    subprocess.call(cmd, shell=True)
    subprocess.call('rm '+ref_text_name, shell=True)
    subprocess.call('rm '+hyp_text_name, shell=True)

    tot = 0
    sub = 0
    del_ = 0
    ins = 0

    with codecs.open(log_cer_name, 'r', 'utf-8') as f:
        for line in f.readlines():
            if ' Sum ' in line:
                #print(line)
                col1 = line.split('|')[2]
                col2 = line.split('|')[3]
                tot = int(col1.split()[-1])
                sub = int(col2.split()[1])
                del_ =  int (col2.split()[2])
                ins = int (col2.split()[3])
    subprocess.call('rm '+log_cer_name, shell=True)


    return tot, sub, del_, ins



def save_fig(filename, data):
    plt.figure()
    sns.heatmap(data, square=True)
    plt.savefig(filename)
    plt.close('all')


def save_evallogs(result_out, num, mfcc, smap):
    mfcc_out = result_out + '/mfcc_' + str(num).zfill(2) + '.npy'
    np.save(mfcc_out, mfcc[0])
    mfcc_out = result_out + '/mfcc_' + str(num).zfill(2) + '.png'
    save_fig(mfcc_out, mfcc[0].transpose())
   
    smap_out = result_out + '/smap_' + str(num).zfill(2) + '.npy'
    np.save(smap_out, smap[0])
    smap_out = result_out + '/smap_' + str(num).zfill(2) + '.png'
    save_fig(smap_out, smap[0].transpose())
   
    smap_out = result_out + '/smap_std' + str(num).zfill(2) + '.txt' 
    smap_std = np.std(smap[0], axis=-1)
    np.savetxt(smap_out, smap_std)

    return smap_std 


def save_evallogs2(result_out, num, weights1, weights2):
 
    weights_out1 = result_out + '/attn_weights_inenc'
    if not os.path.exists(weights_out1):
        os.makedirs(weights_out1)

    for i in range(len(weights1[0])):
        weight = np.zeros((len(weights1), weights1[0][i].shape[-2]))
        for t in range(len(weights1)):
            tmp = np.sum(weights1[t][i][0,:, 0, :], axis=0)
            #weight[t] = tmp / tmp.max()
            weight[t] = tmp
            #weight += np.sum(weights1[t][i][0,:, :, :], axis=0)
            #png_out_t = weights_out1 + '/n' + str(num).zfill(2) + '_' + str(i).zfill(2) + '_' + str(t).zfill(4) + '.png' 
            #save_fig(png_out_t, np.sum(weights1[t][i][0, :, :, :], axis=0)

        print('weight_shape: ', weight.shape)
        npy_out = weights_out1 + '/n' + str(num).zfill(2) + '_' + str(i).zfill(2) + '.npy' 
        png_out = weights_out1 + '/n' + str(num).zfill(2) + '_' + str(i).zfill(2) + '.png' 
        txt_out = weights_out1 + '/n' + str(num).zfill(2) + '_' + str(i).zfill(2) + '.txt' 
        np.save(npy_out, weight)
        save_fig(png_out, weight)
        np.savetxt(txt_out, weight)

        weight_std = np.std(weight, axis=0)[1:]
        txt_out = weights_out1 + '/n' + str(num).zfill(2) + '_std_' + str(i).zfill(2) + '.txt'
        np.savetxt(txt_out, weight_std)



    weights_out2 = result_out + '/attn_weights_seqenc'
    if not os.path.exists(weights_out2):
        os.makedirs(weights_out2)

    for i in range(len(weights2)):
        npy_out = weights_out2 + '/n' + str(num).zfill(2) + '_' + str(i).zfill(2) + '.npy' 
        png_out = weights_out2 + '/n' + str(num).zfill(2) + '_' + str(i).zfill(2) + '.png' 
        txt_out = weights_out2 + '/n' + str(num).zfill(2) + '_' + str(i).zfill(2) + '.txt' 
        np.save(npy_out,  np.sum(weights2[i][0, :, :, :], axis=0))
        save_fig(png_out, np.sum(weights2[i][0, :, :, :], axis=0))
        np.savetxt(txt_out, np.sum(weights2[i][0, :, :, :], axis=0))




if __name__ == '__main__':
    args = parse_args()
   
    modelpath = args.model
    modelpath_ = modelpath.replace('exp/', '')
    xid_subj_task = modelpath_.split('/')[0]
    xid = xid_subj_task.split('_')[0]
    modelname = modelpath_.split('/')[1]

    configfile = modelpath + '/config_' + xid + '.ini'

    if not os.path.exists(configfile):
        print('Error: does not exist config file: ' + configfile) 
        exit()

    listname = os.path.basename(args.data_csv).replace('list_', '').replace('.csv', '')
    
    result_base = modelpath + '/result_' + listname

    # preprocess
    conf = configparser.ConfigParser()
    conf.read(configfile, encoding='utf-8')

    ch_sum = False
    band_sum = True
    part_sum = True
    zscore_hayashi = False
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

        if 'zscore_hayashi' in conf['FEAT']:
            zscore_hayashi = conf['FEAT'].getboolean('zscore_hayashi')

        if 'car' in conf['FEAT']:
            car = conf['FEAT'].getboolean('car')





#    print(ch_sum)
#    exit()

    ecog_data, mfcc_data, tokens, tokenizer = preprocessingX(args.data_csv, conf, car, ch_sum, part_sum, band_sum, zscore_hayashi) 
   
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
        
    #decoder_inputs = Input(shape=(None, dec_input_data.shape[2]))


    # prepare model
    if input_attn:
        in_layer = ElecAttention(conf['INPUTATTN'], False)    
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


    in_layer.load_weights(modelpath + '/in_layer/weights')
    encoder.load_weights(modelpath + '/encoder/weigths')
    decoder.load_weights(modelpath + '/decoder/weights')




    # eval
    new_CER_dec_text = []
    new_CER_ref_text = []
 

    result_out = result_base + '.txt'
    result_out2 = result_base + '_evallog'
    ofp = codecs.open(result_out, 'w', 'utf-8')
    if not os.path.exists(result_out2):
        os.makedirs(result_out2)




    for num, input_seq in enumerate(ecog_data_in):   
        #print(input_seq.shape)

        if trf_flag_dec:
            decoded_sentence, mfcc_out, weights_out1, weights_out2 = evaluateTRF(np.expand_dims(input_seq,0), in_layer, encoder, decoder, tokenizer)
            smap = smapTRF(np.expand_dims(input_seq,0), in_layer, encoder, decoder, tokenizer)
        else:
            decoded_sentence, mfcc_out, weights_out1 = evaluate(np.expand_dims(input_seq,0), in_layer, encoder, decoder, tokenizer)
            smap = smapLSTM(np.expand_dims(input_seq,0), in_layer, encoder, decoder, tokenizer)
            

        save_evallogs(result_out2, num, mfcc_out, smap)
        if input_attn and trf_flag_dec:
            save_evallogs2(result_out2, num, weights_out1, weights_out2)
        print(decoded_sentence)
        res = tokenizer[tokens[num][1]] + ' ' +  tokenizer[tokens[num][2]] + ' ' +  tokenizer[tokens[num][3]] 
        hyp = tokenizer[decoded_sentence[1]] + ' ' + tokenizer[decoded_sentence[2]] + ' ' + tokenizer[decoded_sentence[3]]

        print(num, res, ', ' , hyp)
        ofp.write(str(num) + ',' + res + ',' + hyp + '\n')

        new_CER_ref_text.append(res)
        new_CER_dec_text.append(hyp)

    tot, sub, del_, ins = calc_cer(new_CER_ref_text, new_CER_dec_text)
    cer = (sub + del_ + ins) /tot * 100
    print("WER:",cer)
    ofp.write('PER: ' + str(cer) + ' '+ str(sub+ del_ + ins) + ' / ' + str(tot))
    #ofp.write('CER:' + str(cer) + '\n')
    ofp.close()

