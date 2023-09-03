#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io.wavfile import write
import codecs
import os
import random
from parallel_wavegan.utils import download_pretrained_model


TrainExe = './ecog2audio_train.py'
TestExe = './ecog2audio_eval.py'
Conf_no = 'xA001'
Conf = './conf/config_' + Conf_no + '.ini'
AudioExe = './audio_reconstruction.py'

NumTrain = 64 # The number of training
NumTest = 16 # The number of test
Length = 3 # sec
Freq = 9600 # sampling rate
NumElectrodes = 64 # The number of electrodes
Vocab = [u'わがい', u'わがむ', u'わしい', u'わしむ', u'きがい', u'きがむ', u'きしい', u'きしむ']

def make_dummy_data(Num, mode, elecfile):
    if not os.path.exists('sample_data'):
        os.makedirs('sample_data')

    ofp = codecs.open('sample_list_' + mode + '.csv', 'w', 'utf-8')
    ofp.write('ecog_filename,ecog_fs,wav_filename,transcripts,electrodes\n')

    for n in range(Num):
        n_str = str(n).zfill(4)
        # making sample ecog data
        ECoGFileName = 'sample_data/ecog_' + mode + '_' + n_str + '.npy'
        ECoG = np.random.rand(Length*Freq, NumElectrodes)
        np.save(ECoGFileName, ECoG)

        # making sample wav data
        WavFileName = 'sample_data/wav_' + mode + '_' + n_str + '.wav'
        Wav = np.random.normal(0, 10000, Length*Freq)
        write(WavFileName, Freq, Wav.astype(np.int16))

        # add to list file
        ofp.write(ECoGFileName + ',' + str(Freq) + ',' + WavFileName + ',' + random.choice(Vocab) + ',' + elecfile + '\n')

    ofp.close()
    return 'sample_list_' + mode + '.csv'

if __name__ == '__main__':
    # prepare electrode file
    ofp = codecs.open('sample_elec.csv', 'w', 'utf-8')
    line = ','.join([str(i+1) for i in range(NumElectrodes)])
    ofp.write(line + '\n')
    ofp.close()

    # prepare data
    list_ts = make_dummy_data(NumTest, 'test', 'sample_elec.csv')
    list_tr = make_dummy_data(NumTrain, 'train', 'sample_elec.csv')

    # # make sample exe (for ecog2text)
    # ofp = codecs.open('sample_run.sh', 'w', 'utf-8')
    # ofp.write('#!/bin/bash\n\n')
    # ofp.write(TrainExe + ' ' + list_tr + ' ' + Conf + ' --model sample\n\n')
    # ofp.write(TestExe + ' ' + list_ts + ' exp/xA000_sample_train/sample\n')
    # ofp.close()

    # make sample exe (for ecog2audio)
    ofp = codecs.open('sample_run_audio.sh', 'w', 'utf-8')
    ofp.write('#!/bin/bash\n\n')
    ofp.write(TrainExe + ' ' + list_tr + ' ' + Conf + ' --model sample\n\n')
    ofp.write(TestExe + ' ' + list_ts + ' exp/' + Conf_no + '_sample_train/sample\n\n')
    ofp.write(AudioExe + ' ' + list_ts + ' exp/' + Conf_no + '_sample_train/sample\n')
    ofp.close()


    # download parallel wavegan pretrained model
    download_pretrained_model("jsut_parallel_wavegan.v1", "pretrained_model")


