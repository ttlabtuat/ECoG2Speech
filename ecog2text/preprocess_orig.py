#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import csv
import pandas as pd
import numpy as np
import argparse
import pickle
import configparser
import tensorflow_datasets as tfds

import scipy.io.wavfile as wav
from scipy import signal
from scipy import stats

import matplotlib.pyplot as plt

#sys.path.append('/shared/home/komeshu/AWS/python')
from .python_speech_features import mfcc
from .my_module import my_function_kome as mf


def make_mfcc(wav_file):
    win_len = 0.02 #[s]
    win_step = 0.005 #[s]
        
    (rate,sig) = wav.read(wav_file)
    mfcc_data = mfcc(sig,rate,win_len,win_step)

    return mfcc_data



def read_data(file_csv):
    if not os.path.exists(file_csv):
        print('Error: no data_csv: ' + file_csv)
        exit()

    df = pd.read_csv(file_csv)
   
    file_id = []
    ecogs = []
    ecogs_fs = []
    mfccs = []
    transcripts = []    
    electrodes = []

    for index, row in df.iterrows():
    
        ecog_file = row['ecog_filename']
        ecog_fs = row['ecog_fs']
        wav_file = row['wav_filename']
        transcript = row['transcripts']    
        electrode = row['electrodes']        
        
        ecogs.append(np.load(ecog_file).transpose())
        ecogs_fs.append(int(ecog_fs))
        mfccs.append(make_mfcc(wav_file))
        transcripts.append(transcript)
        electrodes.append(electrode)

        file_id.append(ecog_file)
    return file_id, ecogs, ecogs_fs, mfccs, transcripts, electrodes 



def read_channels(file_chs):
    chs = []
    refs = {}
    if not file_chs == None:
        with open(file_chs, 'r') as f:
            parts = [ part for part in f.readlines()[0].split(':')]
            for part in parts:
                ch = []
                for c in part.split(','):
                    cols = c.split('-')
                    m = int(cols[0]) - 1
                    ch.append(m)
                    if len(cols) > 1:
                        r = int(cols[1]) - 1
                        refs[m] = r
                chs.append(ch)
       
    return chs, refs


def plot_ecog(ecogs, num, file_name):
    dirname = os.path.dirname(file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i in range(num):
        outname = file_name + '_' + str(i) + '.png'
        plt.figure()
        plt.plot(ecogs[i,0,:200])
        plt.savefig(outname)
        plt.close('all')
            




def ecog_preprocess(ecog, fs, chs, refs, car, ch_sum, part_sum, band_sum, zscore_hayashi=True):
    num_ch = ecog.shape[1]    
    ecog_part = ecog
   
    num = 0 
    if not len(chs) == 0:
        used_chs = []
        used_ref = []
        #print(chs, refs, refs[0])
        for part in chs:
            used_chs.extend(part)    

        ch2num = {}
        for i, used_ch in enumerate(sorted(used_chs)):
            ch2num[used_ch] = i



        num_ch = len(used_chs)
        #print(refs)    
        for m in refs:
            #print(m)
            ecog[m, :] = ecog[m, :] - ecog[refs[m], :] 

        ecog_part = ecog[used_chs,:]
        #np.savetxt('dbg_bipola.txt', ecog_part)    

    ecog_lowpass = mf.lowpassFilter(ecog_part, fs, 200, axis=1, order=7)
    #print(ecog_lowpass.shape)
    ecog_400 = signal.decimate(ecog_lowpass, int(fs/400))
    #print(ecog_400.shape)    

    ecog_400_notch = mf.notch_iir(ecog_400, fs=400, fn=50, Q=30, axis=1, show=False)
    ecog_400_notch = mf.notch_iir(ecog_400_notch, fs=400, fn=100, Q=30, axis=1, show=False)

    ecog_ref = ecog_400_notch
    
    if car:
        ref = np.mean(ecog_400_notch, axis=0)
        ecog_ref = ecog_400_notch - ref

    cut_list = [73.0-4.68,73.0+4.68,
            79.5-4.92,79.5+4.92,
            87.8-5.17,87.8+5.17,
            96.9-5.43,96.9+5.43,
            107.0-5.70,107.0+5.70,
            118.1-5.99,118.1+5.99,
            130.4-6.30,130.4+6.30,
            144.0-6.62,144.0+6.62]

    
    ecog_filterd = mf.band_pass_fir_8(data=ecog_ref,fs=400,cut_list=cut_list,numtaps=255)
    ecog_hilberted = np.abs(signal.hilbert(ecog_filterd))
    #print(ecog_hilberted.shape)
    #print(ch_sum, band_sum, zscore_hayashi)


    if ch_sum:
        ecog_ave = np.mean(ecog_hilberted, axis=0)
    else:
        if part_sum and not len(ch2num) == 0:
            ecog_ave = np.zeros((ecog_hilberted.shape[0], len(chs), ecog_hilberted.shape[2]))
            for i, part in enumerate(chs):
                ecog_ave[:,i,:] = np.mean(ecog_hilberted[:,[ch2num[ch] for ch in part] ,:], axis=1)      
        else:
            ecog_ave = ecog_hilberted

    if band_sum:
        ecog_ave = np.mean(ecog_ave, axis=0)
      


    #print(ecog_ave.shape)
    ecog_200 = signal.decimate(ecog_ave, int(400/200))
    print(ecog_200.shape)
    
    if zscore_hayashi:
        ecog_z = mf.z_scored(ecog_200,0.3,200)
    else:     
        ecog_z = stats.zscore(ecog_200, axis=-1)
        # 時系列次元について，zscore が計算できていることを確認済み
        #np.savetxt('ecog_200.csv', ecog_200[0][0],delimiter=',')
        #np.savetxt('ecog_z.csv', ecog_z[0][0],delimiter=',')

    #print(ecog_z.shape)
    #print(ecog_z.transpose().shape)
    #exit()

    return ecog_z
    


def mk_ndarray_padding_with_zero(nd_list):
    max_len = max([nd.shape[0] for nd in nd_list])
    print(max_len)

    nd_array = np.zeros((len(nd_list),  max_len) +  nd_list[0][0].shape)
    for i, nd in enumerate(nd_list):
        nd_array[i,:nd.shape[0],:] = nd

    return nd_array



def text_proc(data_text):
    texts = []
    chars = set()    

    for line in data_text:
        text = '\t' + line + '\n'
        texts.append(text)
        for char in text:
            if not char in chars:
                chars.add(char)

    chars = sorted(list(chars))
    
    num_tokens = len(chars)
    max_len = max([len(text) for text in texts])
    print('Info: # of tokens: ', num_tokens)
    print('Info: Max length of output text: ', max_len)

    char2token = dict([(char, i) for i, char in enumerate(chars)])
    token2char = dict((i, char) for char, i in char2token.items())

    dec_input_data = np.zeros((len(data_text), max_len, num_tokens), dtype='int8')
    dec_target_data = np.zeros((len(data_text), max_len, num_tokens), dtype='int8')

    for i, text in enumerate(texts):
        for t, char in enumerate(text):
            dec_input_data[i, t, char2token[char]] = 1.0
            if t > 0:
                dec_target_data[i, t - 1, char2token[char]] = 1.0

    return dec_input_data, dec_target_data, char2token, token2char
     

def text_procX(data_text):
  seqs = []
  tokenizer = ['w', 'k', 'g', 's', 'i', 'm']

  for line in data_text:
    seq = []
    if u'わがい' in line:
      seq = [0, 2, 4]
    elif u'わがむ' in line:
      seq = [0, 2, 5]
    elif u'わしい' in line:
      seq = [0, 3, 4]
    elif u'わしむ' in line:
      seq = [0, 3, 5]
    elif u'きがい' in line:
      seq = [1, 2, 4]
    elif u'きがむ' in line:
      seq = [1, 2, 5]
    elif u'きしい' in line:
      seq = [1, 3, 4]
    elif u'きしむ' in line:
      seq = [1, 3, 5]
    seqs.append(seq)

	#tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
  #  (txt for txt in seqs), target_vocab_size=258)
	
  tokens = []
  for seq in seqs:
    token = [len(tokenizer)] + seq + [len(tokenizer)+1]
    tokens.append(token)

  return tokens, tokenizer






def prep_data(ecogs_preprocess, mfccs, transcripts, strides, shuffle=False):
    ecog_data = mk_ndarray_padding_with_zero(ecogs_preprocess)
    mfcc_data = mk_ndarray_padding_with_zero(mfccs)

    #ecog_data = np.load('ecog_data.npy')
    #mfcc_data = np.load('mfcc_data.npy')

    if shuffle:
        ecog_data = shuffle_along_axis(ecog_data, 1)

    print('prep_data: ', ecog_data.shape)
    print('prep_data: ', mfcc_data.shape) 

    mfcc_data = mfcc_data[:,strides-1::strides,:]
    ecog_out_len = (ecog_data.shape[1] - 1) // strides + 1

    dec_input_data, dec_target_data, char2token, token2char = text_proc(transcripts)

    print(dec_input_data.shape, dec_target_data.shape)


    if ecog_out_len > mfcc_data.shape[1]:
        mfcc_data = np.pad(mfcc_data, [(0,0), (0, ecog_out_len - mfcc_data.shape[1]), (0, 0)], 'constant')            
    else:
        ecog_data = np.pad(ecog_data, [(0,0), (0, mfcc_data.shape[1] * strides - ecog_data.shape[1]), (0, 0)], 'constant')

    print(ecog_data.shape, mfcc_data.shape)
    
    return ecog_data, mfcc_data, dec_input_data, dec_target_data, transcripts, char2token, token2char 
   
def prep_dataX(ecogs_preprocess, mfccs, transcripts, strides, shuffle=False):
    ecog_data = mk_ndarray_padding_with_zero(ecogs_preprocess)
    mfcc_data = mk_ndarray_padding_with_zero(mfccs)

    #ecog_data = np.load('ecog_data.npy')
    #mfcc_data = np.load('mfcc_data.npy')

    if shuffle:
        ecog_data = shuffle_along_axis(ecog_data, 1)

    print('prep_data: ', ecog_data.shape)
    print('prep_data: ', mfcc_data.shape) 

    mfcc_data = mfcc_data[:,strides-1::strides,:]
    ecog_out_len = (ecog_data.shape[1] - 1) // strides + 1

    tokens, tokenizer = text_procX(transcripts)


    if ecog_out_len > mfcc_data.shape[1]:
        mfcc_data = np.pad(mfcc_data, [(0,0), (0, ecog_out_len - mfcc_data.shape[1]), (0, 0)], 'constant')            
    else:
        ecog_data = np.pad(ecog_data, [(0,0), (0, mfcc_data.shape[1] * strides - ecog_data.shape[1]), (0, 0)], 'constant')

    print(ecog_data.shape, mfcc_data.shape)
    
    return ecog_data, mfcc_data, tokens, tokenizer 
   


def preprocessing(data_csv, conf, car, ch_sum, part_sum, band_sum, zscore_hayashi):

    file_id, ecogs, ecogs_fs, mfccs, transcripts_, electrodes = read_data(data_csv)
		
		

    ecogs_preprocess = []
    for ecog, fs, electrode in zip (ecogs, ecogs_fs, electrodes):
        elecs, refs = read_channels(electrode)

        ecog_prop = ecog_preprocess(ecog, fs, elecs, refs, car, ch_sum, part_sum, band_sum, zscore_hayashi=zscore_hayashi)
        ecogs_preprocess.append(ecog_prop.transpose())
      

    return prep_data(ecogs_preprocess, mfccs, transcripts_, int(conf['INPUTCONV']['strides']))

def preprocessingX(data_csv, conf, car, ch_sum, part_sum, band_sum, zscore_hayashi):

    file_id, ecogs, ecogs_fs, mfccs, transcripts_, electrodes = read_data(data_csv)
		
		

    ecogs_preprocess = []
    for ecog, fs, electrode in zip (ecogs, ecogs_fs, electrodes):
        elecs, refs = read_channels(electrode)

        ecog_prop = ecog_preprocess(ecog, fs, elecs, refs, car, ch_sum, part_sum, band_sum, zscore_hayashi=zscore_hayashi)
        ecogs_preprocess.append(ecog_prop.transpose())
      

    return prep_dataX(ecogs_preprocess, mfccs, transcripts_, int(conf['INPUTCONV']['strides']))


