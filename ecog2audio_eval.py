#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import random
import codecs
import subprocess
import configparser
import argparse
import logging
import h5py
import math

import numpy as np
from ecog2text.model import ECoG2TextInputConv, ECoG2TextEncoder, ECoG2TextEncoder2, ECoG2TextEncoderTrf, ElecAttention
from ecog2text.preprocess import preprocessingX

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import librosa
import librosa.display

def parse_args():
    parser = argparse.ArgumentParser(description='Train and Eval')
    parser.add_argument('data_csv')
    parser.add_argument('model')
    parser.add_argument('--chs', default=None)
    parser.add_argument('--car', action='store_true')

    return parser.parse_args()




def evaluate(input_seq, input_layer, encoder, tokenizer):
    input_seq = tf.convert_to_tensor(input_seq)
    input_seq, weights_out1 = input_layer(input_seq)
    audio_feat_out, enc_out, enc_hidden = encoder(input_seq, False)

    # dec_hidden = enc_hidden

    # decoder_input = tf.expand_dims([len(tokenizer)], 0)
    # output = decoder_input


#     max_length = 17
#     for i in range(max_length):
# #      print(decoder_input)
#     #   predictions, dec_hidden, _ = decoder(decoder_input, dec_hidden, enc_out)
#       predictions = predictions[:, -1:, :]

#       predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
# #      print('1.predicted_id: ', i, predicted_id)
#     #   decoder_input = tf.expand_dims(predicted_id[0], 0)
#       if predicted_id == len(tokenizer) + 1:
#         return tf.squeeze(output, axis=0), audio_feat_out, weights_out1


#       output = tf.concat([output, predicted_id], axis=-1)

    # return tf.squeeze(output, axis=0), audio_feat_out, weights_out1
    return audio_feat_out, weights_out1



def smapLSTM(input_seq__, in_layer, encoder, tokenizer):
    input_seq = tf.Variable(input_seq__, dtype=float)
    i0 = Input(shape=input_seq[0].shape)
    #i0 = Input(shape=input_seq.shape)
    x1 = in_layer(i0)

    grad_model = Model(i0, x1)
    class_output = 0
    with tf.GradientTape() as tape:
        tape.watch(input_seq)
        input_seq_, _ = grad_model(input_seq)
        audio_feat_out, enc_out, enc_hidden = encoder(input_seq_, False)

        dec_hidden = enc_hidden

        # decoder_input = tf.expand_dims([len(tokenizer)], 0)
        # output = decoder_input

        # max_length = 17
        # for i in range(max_length):
        #     #print(i, decoder_input)
        #     # predictions, dec_hidden, _ = decoder(decoder_input, dec_hidden, enc_out)
        #     predictions = predictions[:, -1, :]

        #     predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        #     decoder_input = tf.expand_dims(predicted_id,0)
        #     predicted_score = predictions[:,predicted_id]
        #     if predicted_id == len(tokenizer) + 1:
        #         break

        #     class_output += predicted_score
        #     #print('2. predictions: ', i, predictions)
        #     output = tf.concat([output, [predicted_id]], axis=-1)

    grads = tape.gradient(class_output, input_seq)
    grads = tf.reduce_max(grads, axis=-1)

    min_val, max_val = np.min(grads), np.max(grads)
    smap = (grads - min_val) / (max_val - min_val)

    return smap





def evaluateTRF(input_seq, input_layer, encoder, tokenizer):
    input_seq = tf.convert_to_tensor(input_seq)
    input_seq, weights_out1 = input_layer(input_seq)
    audio_feat_out, enc_out, weights_out2 = encoder(input_seq, False)

    # decoder_input = [len(tokenizer)]
    # output = tf.expand_dims(decoder_input, 0)


    # max_length = 17
    # for i in range(max_length):
    # #   predictions = decoder(enc_out, output)
    #   predictions = predictions[:, -1:, :]

    #   predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    #   if predicted_id == len(tokenizer) + 1:
    #     return tf.squeeze(output, axis=0), audio_feat_out, weights_out1, weights_out2

    #   #print('1.predicted_id: ', i, predicted_id)
    #   output = tf.concat([output, predicted_id], axis=-1)

    # return tf.squeeze(output, axis=0), audio_feat_out, weights_out1, weights_out2
    return audio_feat_out, weights_out1, weights_out2




def smapTRF(input_seq__, in_layer, encoder, tokenizer):
    input_seq = tf.Variable(input_seq__, dtype=float)
    i0 = Input(shape=input_seq[0].shape)
    #i0 = Input(shape=input_seq.shape)
    x1 = in_layer(i0)

    grad_model = Model(i0, x1)
    class_output = 0
    with tf.GradientTape() as tape:
        tape.watch(input_seq)
        input_seq_, _ = grad_model(input_seq)
        audio_feat_out, enc_out, _ = encoder(input_seq_, False)
        # decoder_input = [len(tokenizer)]
        # output = tf.expand_dims(decoder_input, 0)

        max_length = 17
        # for i in range(max_length):
            # predictions = decoder(enc_out, output)
            # predictions = predictions[:, -1, :]

            # predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # predicted_score = predictions[:,predicted_id]
            # if predicted_id == len(tokenizer) + 2:
                # break

            # class_output += predicted_score
            #print('2. predictions: ', i, predictions)
            # output = tf.concat([output, [predicted_id]], axis=-1)

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
    plt.clf()
    plt.close()


def save_melspec(filename, mel, conf):
    fig, ax = plt.subplots()

    conf_melspec = conf['MELSPEC']
    sampling_rate = int(conf_melspec.get('re_sample_rate'))
    win_length = int(conf_melspec.get('win_length'))
    hop_size = int(conf_melspec.get('hop_size'))
    fft_size = int(conf_melspec.get('fft_size'))
    fmin = int(conf_melspec.get('fmin'))
    fmax = int(conf_melspec.get('fmax'))

    # tensor -> numpy
    mel = mel.numpy()

    img = librosa.display.specshow(
        20*mel.T,
        sr=sampling_rate,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        hop_length=hop_size,
        n_fft=fft_size,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        cmap='magma'
    )
    # exit()
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # title = os.path.basename(audio_path) + ' n_mfcc' + str(n_mel)
    # ax.set(title=title)
    plt.savefig(filename)
    plt.clf()
    plt.close()



def save_melspec_forpubli(filename, mel, conf):
    fig, ax = plt.subplots()

    conf_melspec = conf['MELSPEC']
    sampling_rate = int(conf_melspec.get('re_sample_rate'))
    win_length = int(conf_melspec.get('win_length'))
    hop_size = int(conf_melspec.get('hop_size'))
    fft_size = int(conf_melspec.get('fft_size'))
    fmin = int(conf_melspec.get('fmin'))
    # fmax = int(conf_melspec.get('fmax'))
    fmax = 4800

    # tensor -> numpy
    mel = mel.numpy()

    img = librosa.display.specshow(
        20*mel.T,
        sr=sampling_rate,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        hop_length=hop_size,
        n_fft=fft_size,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        cmap='magma'
    )
    # exit()
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # title = os.path.basename(audio_path) + ' n_mfcc' + str(n_mel)
    # ax.set(title=title)
    plt.savefig(filename)
    plt.clf()
    plt.close()


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning(
                    "Dataset in hdf5 file already exists. recreate dataset in hdf5."
                )
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()



def save_evallogs(result_out, num, audio_feat, smap, conf):
    audio_feat_type = conf['FEAT']['audio_feat']
    audio_feat_out_base = result_out + '/audio_feat_' + str(num).zfill(2)

    audio_feat_out = audio_feat_out_base + '.npy'
    np.save(audio_feat_out, audio_feat[0])

    if audio_feat_type == 'melspec':
        audio_feat_out = audio_feat_out_base + '.h5'
        write_hdf5(
            audio_feat_out,
            "feats",
            audio_feat[0].astype(np.float32),
            # is_overwrite=False
        )

    audio_feat_out = audio_feat_out_base + '.png'
    print('audio_feat shape:', audio_feat[0].shape)
    if audio_feat_type == 'mfcc':
        save_fig(audio_feat_out, audio_feat[0].transpose())
    elif audio_feat_type == 'melspec':
        save_melspec(audio_feat_out, audio_feat[0], conf)
    else:
        print('Error: incorrect audio_feat')
        exit()

    smap_out = result_out + '/smap_' + str(num).zfill(2) + '.npy'
    np.save(smap_out, smap[0])
    smap_out = result_out + '/smap_' + str(num).zfill(2) + '.png'
    save_fig(smap_out, smap[0].transpose())

    smap_out = result_out + '/smap_std' + str(num).zfill(2) + '.txt'
    smap_std = np.std(smap[0], axis=-1)
    np.savetxt(smap_out, smap_std)

    return smap_std



def save_evallogs_test(result_out, num, audio_feat_pred_raw, audio_feat_pred_zerocut, audio_feat_pred_proc, conf):
    audio_feat_type = conf['FEAT']['audio_feat']
    audio_feat_out_base = result_out + '/audio_feat_' + str(num).zfill(2)

    # モデルから推定されたmelをそのまま保存(.npy .png)
    audio_feat_pred_raw_out = audio_feat_out_base + '_pred.npy'
    np.save(audio_feat_pred_raw_out, audio_feat_pred_raw[0])
    audio_feat_pred_raw_out = audio_feat_out_base + '_pred.png'
    print('audio_feat_pred shape:', audio_feat_pred_raw[0].shape)
    if audio_feat_type == 'mfcc':
        save_fig(audio_feat_pred_raw_out, audio_feat_pred_raw[0].transpose())
    elif audio_feat_type == 'melspec':
        save_melspec(audio_feat_pred_raw_out, audio_feat_pred_raw[0], conf)
    else:
        print('Error: incorrect audio_feat')
        exit()

    # モデルから推定されたmelのゼロ埋めを削除して高周波帯域を小さいデシベル値で埋めたものを保存(.h5 .png)
    # この.h5ファイルをボコーダに渡して音声再合成する
    if audio_feat_type == 'melspec':
        audio_feat_pred_zerocut_out = audio_feat_out_base + '.h5'
        write_hdf5(
            audio_feat_pred_zerocut_out,
            "feats",
            audio_feat_pred_zerocut[0].astype(np.float32),
            # is_overwrite=False
        )
    audio_feat_pred_zerocut_out = audio_feat_out_base + '_zerocut.png'
    print('audio_feat shape:', audio_feat_pred_zerocut[0].shape)
    if audio_feat_type == 'mfcc':
        save_fig(audio_feat_pred_zerocut_out, audio_feat_pred_zerocut[0].transpose())
    elif audio_feat_type == 'melspec':
        save_melspec(audio_feat_pred_zerocut_out, audio_feat_pred_zerocut[0], conf)
    else:
        print('Error: incorrect audio_feat')
        exit()

    # モデルから推定されたmelのゼロ埋めと高周波帯域を削除したものを保存(.png)
    # 余計な部分が入っていない図を示す時用
    audio_feat_pred_proc_out = audio_feat_out_base + '_proc.png'
    print('audio_feat_pred shape:', audio_feat_pred_proc[0].shape)
    if audio_feat_type == 'mfcc':
        save_fig(audio_feat_pred_proc_out, audio_feat_pred_proc[0].transpose())
    elif audio_feat_type == 'melspec':
        save_melspec_forpubli(audio_feat_pred_proc_out, audio_feat_pred_proc[0], conf)
    else:
        print('Error: incorrect audio_feat')
        exit()

    # smap_out = result_out + '/smap_' + str(num).zfill(2) + '.npy'
    # np.save(smap_out, smap[0])
    # smap_out = result_out + '/smap_' + str(num).zfill(2) + '.png'
    # save_fig(smap_out, smap[0].transpose())

    # smap_out = result_out + '/smap_std' + str(num).zfill(2) + '.txt'
    # smap_std = np.std(smap[0], axis=-1)
    # np.savetxt(smap_out, smap_std)

    # return smap_std
    return None


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

def loss_audio_feat_function(real, pred, conf):
    l_2 = tf.keras.losses.mean_squared_error(real, pred)
    loss = l_2
    if conf['TRAIN'].get('loss_l1'):
        l_1 = tf.keras.losses.mean_absolute_error(real, pred)
        loss = loss + l_1
    return tf.reduce_mean(loss)



def melspec_padding(mel):
    mel = mel.numpy() # <tensorflow.python.framework.ops.EagerTensor> to <numpy.ndarray>

    mel_tmp = mel[:,:,:65]
    max = np.amax(mel_tmp)
    min = np.amin(mel_tmp)
    print('mel max, mel min =', max, ',', min)
    # mel[:,:,65:] = -5.0 # 64次元目が音声データのある最大次元数(決め打ち),65以上はdB値で小さい値を入れる(ここでのmelは対数melで,20をかけたものがdB値になる.-5.0も決め打ち)
    mel[:,:,65:] = math.floor(min) - 0.5 # 64次元目が音声データのある最大次元数(決め打ち),65以上はdB値で小さい値を入れる(ここでのmelは対数melで,20をかけたものがdB値になる.-0.5も決め打ち)
    # exit()
    mel = tf.convert_to_tensor(mel) #  <numpy.ndarray> to <tensorflow.python.framework.ops.EagerTensor>
    return mel


if __name__ == '__main__':
    args = parse_args()

    modelpath = args.model
    modelpath_ = modelpath.replace('exp/', '')
    xid_subj_task = modelpath_.split('/')[0]
    xid = xid_subj_task.split('_')[0]
    subj_task = xid_subj_task.split('_',1)[1]
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


    if '01' in subj_task:
        meldump = True
    else:
        meldump = False

#    print(ch_sum)
#    exit()

    ecog_data, audio_feat_data, tokens, tokenizer, audio_feat_padding = preprocessingX(args.data_csv, conf, car, ch_sum, part_sum, band_sum, args.model, meldump)

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

    audio_feat_dim = audio_feat_data.shape[2]

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
    # trf_flag_dec = conf['DECODER'].getboolean('trf_flag')
    if trf_flag_enc:
        encoder = ECoG2TextEncoderTrf(conf['ENCODER'], audio_feat_dim)
        # if trf_flag_dec:
        #     encoder = ECoG2TextEncoderTrf(conf['ENCODER'], audio_feat_dim)
        #     decoder = ECoG2TextDecoderTrf(len(tokenizer) + 2, conf['DECODER'])
        # else:
        #     encoder = ECoG2TextEncoder2(conf['ENCODER'], audio_feat_dim)
        #     decoder = ECoG2TextDecoder(len(tokenizer) + 2, conf['DECODER'])

    else:
        encoder = ECoG2TextEncoder(conf['ENCODER'], audio_feat_dim)
        # decoder = ECoG2TextDecoder(len(tokenizer) + 2, conf['DECODER'])


    in_layer.load_weights(modelpath + '/in_layer/weights')
    encoder.load_weights(modelpath + '/encoder/weigths')
    # decoder.load_weights(modelpath + '/decoder/weights')




    # eval
    new_CER_dec_text = []
    new_CER_ref_text = []


    result_out = result_base + '.txt'
    result_out2 = result_base + '_evallog'
    result_mse_out = result_base + '_mse.txt'
    # ofp = codecs.open(result_out, 'w', 'utf-8')
    ofp2 = codecs.open(result_mse_out, 'w', 'utf-8')
    if not os.path.exists(result_out2):
        os.makedirs(result_out2)




    for num, input_seq in enumerate(ecog_data_in):
        #print(input_seq.shape)

        if trf_flag_enc:
            audio_feat_out, weights_out1, weights_out2 = evaluateTRF(np.expand_dims(input_seq,0), in_layer, encoder, tokenizer)
            # smap = smapTRF(np.expand_dims(input_seq,0), in_layer, encoder, tokenizer)
        else:
            audio_feat_out, weights_out1 = evaluate(np.expand_dims(input_seq,0), in_layer, encoder, tokenizer)
            # smap = smapLSTM(np.expand_dims(input_seq,0), in_layer, encoder, tokenizer)

        audio_feat_out_pred_raw = audio_feat_out
        audio_feat_data_real_raw = audio_feat_data[num]

        # 時間方向のゼロ埋めと高周波地域の不要部分を削る
        if audio_feat_padding != None:
            audio_feat_cutpad = audio_feat_out[0].shape[0] - audio_feat_padding

            audio_feat_out_pred_zerocut = audio_feat_out[:,0:audio_feat_cutpad,:] # 後で高周波帯域を別の値で埋めるため，余分に変数を用意

            audio_feat_out_pred = audio_feat_out[:,0:audio_feat_cutpad,0:65]
            audio_feat_data_real = audio_feat_data_real_raw[0:audio_feat_cutpad,0:65]

        # 不要部分を削ったmelでMSE lossを計算
        loss = loss_audio_feat_function(audio_feat_data_real, audio_feat_out_pred[0], conf)
        print('MSE loss: ', loss)
        ofp2.write(str(loss.numpy()) + '\n')

        # 高周波帯域の不要部分を別の値で埋める
        audio_feat_out_pred_high_processed = melspec_padding(audio_feat_out_pred_zerocut)

        # save_evallogs(result_out2, num, audio_feat_out, smap, conf)
        save_evallogs_test(result_out2, num, audio_feat_out_pred_raw, audio_feat_out_pred_high_processed, audio_feat_out_pred, conf)
        # if input_attn and trf_flag_dec:
        #     save_evallogs2(result_out2, num, weights_out1, weights_out2)
        # print(decoded_sentence)
        # res = tokenizer[tokens[num][1]] + ' ' +  tokenizer[tokens[num][2]] + ' ' +  tokenizer[tokens[num][3]]
        # # hyp = tokenizer[decoded_sentence[1]] + ' ' + tokenizer[decoded_sentence[2]] + ' ' + tokenizer[decoded_sentence[3]]

        # print(num, res, ', ' , hyp)
        # ofp.write(str(num) + ',' + res + ',' + hyp + '\n')

        # new_CER_ref_text.append(res)
        # new_CER_dec_text.append(hyp)

    # tot, sub, del_, ins = calc_cer(new_CER_ref_text, new_CER_dec_text)
    # cer = (sub + del_ + ins) /tot * 100
    # print("WER:",cer)
    # ofp.write('PER: ' + str(cer) + ' '+ str(sub+ del_ + ins) + ' / ' + str(tot))
    # #ofp.write('CER:' + str(cer) + '\n')
    # ofp.close()
    ofp2.close()

