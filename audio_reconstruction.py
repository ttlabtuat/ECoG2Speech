#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import subprocess
import sys
import glob
import argparse
import os
import configparser
import logging
import h5py

def parse_args():
    parser = argparse.ArgumentParser(description='Reconstruct Audio from Mel Spectrogram')
    parser.add_argument('data_csv')
    parser.add_argument('model')
    parser.add_argument('--chs', default=None)
    parser.add_argument('--car', action='store_true')

    return parser.parse_args()


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
    result_folder = result_base + '_evallog'

    # name_pattern = result_folder + '/*.h5'
    # files = sorted(glob.glob(name_pattern))
    # for file in files:
    #     print(file)
    #     mel = read_hdf5(file, 'feats')
    #     exit()

    audiodir = modelpath + '/recon_audio'
    if not os.path.exists(audiodir):
        os.makedirs(audiodir)
    # cp = subprocess.run(['parallel-wavegan-decode', '--checkpoint ../pretrained_model/jsut_parallel_wavegan.v1/checkpoint-400000steps.pkl', '--dumpdir ' + result_folder, '--normalize-before', '--outdir ' + audiodir])
    cp = subprocess.run([
        'parallel-wavegan-decode',
        '--checkpoint',
        './pretrained_model/jsut_parallel_wavegan.v1/checkpoint-400000steps.pkl',
        '--dumpdir',
        result_folder,
        '--normalize-before',
        '--outdir',
        audiodir
    ])



