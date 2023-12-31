#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging

import seaborn as sns
import scipy.io.wavfile as wav
from scipy import signal
from scipy import stats
import soundfile as sf

import matplotlib.pyplot as plt

#sys.path.append('/shared/home/komeshu/AWS/python')
from .python_speech_features import mfcc
from .my_module import my_function_kome as mf

import librosa
import librosa.display


def save_fig(filename, data):
    plt.figure()
    sns.heatmap(data, square=True)
    plt.savefig(filename)
    plt.clf()
    plt.close()

# almost same as def save_fig()
def save_fig_(filename, data):
    plt.figure()
    ax = sns.heatmap(data, square=True)
    ax.invert_yaxis()
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

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # title = os.path.basename(audio_path) + ' n_mfcc' + str(n_mel)
    # ax.set(title=title)
    plt.savefig(filename)
    plt.clf()
    plt.close()



def make_mfcc(wav_file, conf):
    # win_len = 0.02 #[s]
    # win_step = 0.005 #[s]

    conf_mfcc = conf['MFCC']
    win_len = float(conf_mfcc.get('win_len'))
    win_step = float(conf_mfcc.get('win_step'))
    num_mel = int(conf_mfcc.get('num_mel'))
    num_cep = int(conf_mfcc.get('num_cep'))
    n_fft = int(conf_mfcc.get('n_fft'))
    win = conf_mfcc.get('win')

    (rate,sig) = wav.read(wav_file)
    mfcc_data = mfcc(sig,rate,win_len,win_step) # shape -> (time) * (num_cep)

    return mfcc_data



def audio_resample(orig_audio, orig_sr, target_sr):
    return librosa.resample(orig_audio, orig_sr=orig_sr, target_sr=target_sr)


# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)
def make_melspec(
    audio_path, # change audio -> audio_path
    conf, # add conf file path
    outmodeldir, # for dump
    loop_count, # for dump
    meldump,
    sampling_rate=24000,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
    ):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
        log_base (float): Log base. If set to None, use np.log.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """

    conf_melspec = conf['MELSPEC']
    exp_sample_rate = int(conf_melspec.get('exp_sample_rate'))
    sampling_rate = int(conf_melspec.get('re_sample_rate'))
    win_length = int(conf_melspec.get('win_length'))
    hop_size = int(conf_melspec.get('hop_size'))
    num_mels = int(conf_melspec.get('num_mels'))
    fft_size = int(conf_melspec.get('fft_size'))
    fmin = int(conf_melspec.get('fmin'))
    fmax = int(conf_melspec.get('fmax'))
    window = conf_melspec.get('window')
    log_base = float(conf_melspec.get('log_base'))
    global_gain_scale = float(conf_melspec.get('global_gain_scale'))




    # load wav (sampling rate at experiment (1200 Hz or 9600 Hz))
    audio, _ = librosa.load(audio_path, sr=exp_sample_rate)


    # resample the audio (to match sampling rate with pretrained-neural-vocoder's sampling rate)
    audio = audio_resample(audio, orig_sr=exp_sample_rate, target_sr=sampling_rate)


    # 音声を一旦書き出して変数の型を変換
    dump_wav_dir = os.path.dirname(outmodeldir) + '/dump-' + os.path.basename(outmodeldir)
    if not os.path.exists(dump_wav_dir):
        os.makedirs(dump_wav_dir)
    dump_wav = dump_wav_dir + '/' + os.path.basename(audio_path)
    sf.write(file=dump_wav, data=audio, samplerate=24000)
    audio, _ = librosa.load(dump_wav, sr=sampling_rate)



    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))

    if log_base is None:
        mel = np.log(mel)
    elif log_base == 10.0:
        mel = np.log10(mel)
    elif log_base == 2.0:
        mel = np.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")


    # make sure the audio length and feature length are matched
    audio = np.pad(audio, (0, fft_size), mode="reflect")
    audio = audio[: len(mel) * hop_size]
    assert len(mel) * hop_size == len(audio)

    # apply global gain
    if global_gain_scale > 0.0:
        audio *= global_gain_scale
    if np.abs(audio).max() >= 1.0:
        logging.warn(
            f"{audio_path} causes clipping. "
            "it is better to re-consider global gain scale."
        )

    if meldump:
        dump_mel = dump_wav_dir + '/' + os.path.basename(audio_path).split('.')[0] + '.png'
        save_melspec(dump_mel,mel,conf)

    # # dump mel spectrogram
    # mel_dump_dir = outmodeldir + '/mel_dump'
    # if not os.path.exists(mel_dump_dir):
    #     os.makedirs(mel_dump_dir)
    # mel_dump = mel_dump_dir + '/melspec_' + str(loop_count).zfill(2) + '.npy'
    # np.save(mel_dump, mel)
    # mel_dump = mel_dump_dir + '/melspec_' + str(loop_count).zfill(2) + '.png'
    # save_melspec(mel_dump, mel, conf) # 先に path を作って画像保存すると train.py でエラー発生


    # out mel shape -> (time) * (num_mels)
    return mel



def read_data(file_csv, conf, outmodeldir, meldump):
    if not os.path.exists(file_csv):
        print('Error: no data_csv: ' + file_csv)
        exit()

    df = pd.read_csv(file_csv)

    audio_feat_type = conf['FEAT']['audio_feat']

    file_id = []
    ecogs = []
    ecogs_fs = []
    audio_feat = []
    transcripts = []
    electrodes = []

    for index, row in df.iterrows():
        # print('index: ' + str(index))
        # print('index_type: '  + str(type(index)))

        print('loop:', index+1)

        ecog_file = row['ecog_filename']
        ecog_fs = row['ecog_fs']
        wav_file = row['wav_filename']
        transcript = row['transcripts']
        electrode = row['electrodes']

        ecogs.append(np.load(ecog_file).transpose())
        ecogs_fs.append(int(ecog_fs))
        if audio_feat_type == 'mfcc':
            mfcc = make_mfcc(wav_file, conf)
            print('mfcc shape:\t', mfcc.shape)
            audio_feat.append(mfcc)
        elif audio_feat_type == 'melspec':
            mel = make_melspec(wav_file, conf, outmodeldir, index, meldump)
            # save_melspec('./mel_test.png',mel,conf)
            print('mel spec shape:\t', mel.shape)
            audio_feat.append(mel)
        else:
            print('Error: incorrect audio_feat')
            exit()
        transcripts.append(transcript)
        electrodes.append(electrode)

        file_id.append(ecog_file)
    return file_id, ecogs, ecogs_fs, audio_feat, transcripts, electrodes



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





def ecog_preprocess(ecog, fs, chs, refs, car, ch_sum, part_sum, band_sum):
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

    # decimateのローパスフィルタの設定が適していないため修正    
    # ecog_lowpass = mf.lowpassFilter(ecog_part, fs, 200, axis=1, order=7)
    # print('ecog_lowpass shape:\t', ecog_lowpass.shape)
    # ecog_400 = signal.decimate(ecog_lowpass, int(fs/400))
    # 時間軸はaxis=1(shape[-1])
    ecog_400 = signal.resample(ecog_part, int(ecog_part.shape[-1] * 400 / fs), axis=1)
    # print('ecog_400 shape:\t', ecog_400.shape)

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
        
    # decimateのローパスフィルタの設定が適していないため修正
    # ecog_200 = signal.decimate(ecog_ave, int(400/200))
    # 時間軸はaxis=1(shape[-1])
    ecog_200 = signal.resample(ecog_ave, int(ecog_ave.shape[-1] * 200 / 400), axis=1)
    # print('ecog_200 shape:\t', ecog_200.shape)

    ecog_z = stats.zscore(ecog_200, axis=-1)


    return ecog_z


# ecogと音響特徴量それぞれにおけるトライアルごとの長さの違いを0埋め，listやtensor型をndarray型に変換
# 本来はecogと音声を切り出す時点で長さを揃えているため，ここでは実質型変換のみ実行されている？
def mk_ndarray_padding_with_zero(nd_list):
    max_len = max([nd.shape[0] for nd in nd_list])
    print('max_len:\t', max_len)

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


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)




def prep_data(ecogs_preprocess, audio_feat, transcripts, strides, shuffle=False):
    ecog_data = mk_ndarray_padding_with_zero(ecogs_preprocess)
    audio_feat_data = mk_ndarray_padding_with_zero(audio_feat)

    #ecog_data = np.load('ecog_data.npy')
    #mfcc_data = np.load('mfcc_data.npy')

    if shuffle:
        ecog_data = shuffle_along_axis(ecog_data, 1)

    print('ecog_data shape:\t', ecog_data.shape)
    print('audio_feat_data shape:\t', audio_feat_data.shape)

    audio_feat_data = audio_feat_data[:,strides-1::strides,:]
    ecog_out_len = (ecog_data.shape[1] - 1) // strides + 1

    dec_input_data, dec_target_data, char2token, token2char = text_proc(transcripts)

    print('dec_input_data shape:\t', dec_input_data.shape,
          'dec_target_data shape:\t', dec_target_data.shape)


    if ecog_out_len > audio_feat_data.shape[1]:
        audio_feat_data = np.pad(audio_feat_data, [(0,0), (0, ecog_out_len - audio_feat_data.shape[1]), (0, 0)], 'constant')
    else:
        ecog_data = np.pad(ecog_data, [(0,0), (0, audio_feat_data.shape[1] * strides - ecog_data.shape[1]), (0, 0)], 'constant')

    print('ecog_data shape:\t', ecog_data.shape,
          'audio_feat_data shape:\t', audio_feat_data.shape)

    return ecog_data, audio_feat_data, dec_input_data, dec_target_data, transcripts, char2token, token2char

def prep_dataX(ecogs_preprocess, audio_feat, transcripts, strides,conf, shuffle=False):
    ecog_data = mk_ndarray_padding_with_zero(ecogs_preprocess)
    audio_feat_data = mk_ndarray_padding_with_zero(audio_feat)

    #ecog_data = np.load('ecog_data.npy')
    #mfcc_data = np.load('mfcc_data.npy')

    if shuffle:
        ecog_data = shuffle_along_axis(ecog_data, 1)

    print('ecog_data shape:\t', ecog_data.shape)
    print('audio_feat_data shape:\t', audio_feat_data.shape)

    # audio_feat_data = audio_feat_data[:,strides/2-1::strides/2,:] # TODO
    print('audio_feat_data strides:\t',audio_feat_data.shape)
    ecog_out_len = (ecog_data.shape[1] - 1) // strides + 1
    print('ecog_out_len\t',ecog_out_len)

    tokens, tokenizer = text_procX(transcripts)


    audio_feat_padding = None
    if ecog_out_len > audio_feat_data.shape[1]:
        audio_feat_padding = ecog_out_len - audio_feat_data.shape[1]
        audio_feat_data = np.pad(audio_feat_data, [(0,0), (0, audio_feat_padding), (0, 0)], 'constant')
    else:
        padding_len = audio_feat_data.shape[1] * strides - ecog_data.shape[1]
        ecog_data = np.pad(ecog_data, [(0,0), (0, padding_len), (0, 0)], 'constant')

    print(ecog_data.shape, audio_feat_data.shape)


    # np.save('audio-feat-after-pad.npy', audio_feat_data[0])
    # save_fig('audio-feat-after-pad.png', audio_feat_data[0])
    # save_melspec('mel-after-pad.png', audio_feat_data[0],conf)

    return ecog_data, audio_feat_data, tokens, tokenizer, audio_feat_padding



def preprocessing(data_csv, conf, car, ch_sum, part_sum, band_sum):

    file_id, ecogs, ecogs_fs, audio_feat, transcripts_, electrodes = read_data(data_csv)



    ecogs_preprocess = []
    for ecog, fs, electrode in zip (ecogs, ecogs_fs, electrodes):
        elecs, refs = read_channels(electrode)

        ecog_prop = ecog_preprocess(ecog, fs, elecs, refs, car, ch_sum, part_sum, band_sum)
        ecogs_preprocess.append(ecog_prop.transpose())


    return prep_data(ecogs_preprocess, audio_feat, transcripts_, int(conf['INPUTCONV']['strides']))

def preprocessingX(data_csv, conf, car, ch_sum, part_sum, band_sum, outmodeldir=None, meldump=False):

    file_id, ecogs, ecogs_fs, audio_feat, transcripts_, electrodes = read_data(data_csv, conf, outmodeldir, meldump)



    ecogs_preprocess = []
    loop_count = 0
    for ecog, fs, electrode in zip (ecogs, ecogs_fs, electrodes):
        elecs, refs = read_channels(electrode)

        ecog_prop = ecog_preprocess(ecog, fs, elecs, refs, car, ch_sum, part_sum, band_sum)
        print('loop:', loop_count+1)
        print('ecog prepro shape:\t',ecog_prop.transpose().shape)
        ecogs_preprocess.append(ecog_prop.transpose())

        loop_count += 1

    return prep_dataX(ecogs_preprocess, audio_feat, transcripts_, int(conf['INPUTCONV']['strides']),conf)


