# coding: utf-8
import matplotlib.pyplot as plt
import cmath
import numpy as np
import pandas as pd
import glob
import seaborn as sns
from scipy import signal
from scipy import fftpack
import math
import scipy as sp
from scipy import stats
import h5py
import statistics
import pickle
import sys
import os
import csv
import wave, array
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
#import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,roc_curve,auc
# Scikit-learn(Sklearn)のインポート
#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
sys.path.append("../")
#from my_module import my_function
import scipy.linalg
from sklearn.base import TransformerMixin, BaseEstimator
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

# --------------------音声切り出し関数群---------------------------------------
def getsections_dummy(trg):
    # sections = [[Tri1],[Tri2],...[Tri16]]
    # Tri1 = [lis_st,lis_end,im_st,im_end,sp_st,sp_end]
    flag = 0
    session = 0
    task = 0
    sections = []
    duration = []
    for t in range(len(trg) - 1):
        if trg[t] == 3 and trg[t+1] == 2 and flag == 0:
            task = (task+1)%3
            flag = 1
            duration.append(t)
        elif trg[t] == 1 and trg[t+1] == 3 and flag == 1:
            task = (task+1)%3
            flag = 0
            duration.append(t)
            if task == 0:
                session = session + 1
                sections.append(duration)
                duration = []
    #bp()
    return sections

def getsections(trg):
    # sections = [[Tri1],[Tri2],...[Tri16]]
    # Tri1 = [lis_st,lis_end,im_st,im_end,sp_st,sp_end]
    flag = 0
    session = 0
    task = 0
    sections = []
    duration = []
    for t in range(len(trg) - 1):
        if trg[t] == 2 and trg[t+1] == 1 and flag == 0:
            task = (task+1)%3
            flag = 1
            duration.append(t)
        elif trg[t] == 1 and trg[t+1] == 3 and flag == 1:
            task = (task+1)%3
            flag = 0
            duration.append(t)
            if task == 0:
                session = session + 1
                sections.append(duration)
                duration = []
    sections_fixed = []
    for i,set in enumerate(sections):
        sections_fixed_1 = []
        lisen_length = set[1]-set[0]
        image_length = set[3]-set[2]
        speak_length = set[5]-set[4]
        max_len = max(lisen_length,image_length,speak_length)
        sections_fixed_1.append(set[0])
        sections_fixed_1.append(set[0]+max_len)
        sections_fixed_1.append(set[2])
        sections_fixed_1.append(set[2]+max_len)
        sections_fixed_1.append(set[4])
        sections_fixed_1.append(set[4]+max_len)
        sections_fixed.append(sections_fixed_1)

    return sections_fixed

def cutout_voice(raw, sct, exp, num, label, mergin_f, mergin_b, outdir, Fs):
    num_ = 1001 + num
    exp_ = exp.replace('_','-')
    mergin_ms_f = int((mergin_f*1000)/Fs)
    mergin_ms_b = int((mergin_b*1000)/Fs)
    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b) + '-voice' + '-t3.wav'

    print (outfile)

    # sct[0]:sct[1]→傾聴
    # sct[2]:sct[3]→想像
    # sct[4]:sct[5]→発声
    out = raw[(int(sct[4])-mergin_f):(int(sct[5])+mergin_b)]
    #np.save("out_voice.npy",out)
    #print("raw.shape=",raw.shape)
    #print("out.shape=",out.shape)

    w = wave.Wave_write(outfile)
    w.setparams((
        1,#nchanels
        2,#sampwidth
        Fs,#framerate
        len(raw),#nframes
        "NONE",#comptype
        "not compressed"#compname
    ))
    #np.savez(outfile, out)
    w.writeframes(array.array('h', out).tostring())
    w.close()
    #return out
    #bp()

def cutout_sound(raw, sct, exp, num, label, mergin_f,mergin_b, outdir, Fs):
    num_ = 1001 + num
    exp_ = exp.replace('_','-')
    mergin_ms_f = int((mergin_f*1000)/Fs)
    mergin_ms_b = int((mergin_b*1000)/Fs)
    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b) + '-sound' + '-t1.wav'

    print (outfile)

    # sct[0]:sct[1]→傾聴
    # sct[2]:sct[3]→想像
    # sct[4]:sct[5]→発声
    out = raw[(int(sct[0])-mergin_f):(int(sct[1])+mergin_b)]
    #np.save("out_sound.npy",out)
    w = wave.Wave_write(outfile)
    w.setparams((
        1,#nchanels
        2,#sampwidth
        Fs,#framerate
        len(raw),#nframes
        "NONE",#comptype
        "not compressed"#compname
    ))
    #np.savez(outfile, out)
    w.writeframes(array.array('h', out).tostring())
    w.close()
    #return out

def cutout_ECoG(raw, sct, exp, num, label, mergin_f,mergin_b, outdir, Fs):
    num_ = 1001 + num
    exp_ = exp.replace('_','-')
    mergin_ms_f = int((mergin_f*1000)/Fs)
    mergin_ms_b = int((mergin_b*1000)/Fs)
    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b)  + '-ECoG' + '-t1.npy'
    print (outfile)
    # sct[0]:sct[1]→傾聴
    # sct[2]:sct[3]→想像
    # sct[4]:sct[5]→発声
    out = raw[(int(sct[0])-mergin_f):(int(sct[1])+mergin_b),:]
    print("ecog.shape",out.shape)
    np.save(outfile,out)

    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b) + '-ECoG' + '-t2.npy'
    print (outfile)
    out = raw[(int(sct[2])-mergin_f):(int(sct[3])+mergin_b),:]
    np.save(outfile,out)

    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b) + '-ECoG' + '-t3.npy'
    print (outfile)
    out = raw[(int(sct[4])-mergin_f):(int(sct[5])+mergin_b),:]
    np.save(outfile,out)

def cutout_EOG(raw, sct, exp, num, label, mergin_f,mergin_b, outdir, Fs):
    num_ = 1001 + num
    exp_ = exp.replace('_','-')
    mergin_ms_f = int((mergin_f*1000)/Fs)
    mergin_ms_b = int((mergin_b*1000)/Fs)
    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b) + '-EOG' + '-t1.npy'
    print (outfile)
    # sct[0]:sct[1]→傾聴
    # sct[2]:sct[3]→想像
    # sct[4]:sct[5]→発声
    out = raw[(int(sct[0])-mergin_f):(int(sct[1])+mergin_b),:]
    print("EOG.shape",out.shape)
    np.save(outfile,out)

    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b) + '-EOG' + '-t2.npy'
    print (outfile)
    out = raw[(int(sct[2])-mergin_f):(int(sct[3])+mergin_b),:]
    np.save(outfile,out)

    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b) + '-EOG' + '-t3.npy'
    print (outfile)
    out = raw[(int(sct[4])-mergin_f):(int(sct[5])+mergin_b),:]
    np.save(outfile,out)

def cutout_event(raw, sct, exp, num, label, mergin_f,mergin_b, outdir, Fs):
    num_ = 1001 + num
    exp_ = exp.replace('_','-')
    mergin_ms_f = int((mergin_f*1000)/Fs)
    mergin_ms_b = int((mergin_b*1000)/Fs)
    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b) + '-event' + '-t1.npy'
    print (outfile)
    # sct[0]:sct[1]→傾聴
    # sct[2]:sct[3]→想像
    # sct[4]:sct[5]→発声
    out = raw[(int(sct[0])-mergin_f):(int(sct[1])+mergin_b)]
    print("event.shape",out.shape)
    np.save(outfile,out)

    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b) + '-event' + '-t2.npy'
    print (outfile)
    out = raw[(int(sct[2])-mergin_f):(int(sct[3])+mergin_b)]
    np.save(outfile,out)

    outfile = outdir + '/' + exp_ + '-' + str(num_) + '-' + str(label) + '-mf' + str(mergin_ms_f) + '-mb' + str(mergin_ms_b) + '-event' + '-t3.npy'
    print (outfile)
    out = raw[(int(sct[4])-mergin_f):(int(sct[5])+mergin_b)]
    np.save(outfile,out)

#サンプル数が150以下の幅の変化を削除
def trg_recover_1(trg_mat,point):
    array = trg_mat
    flag = False
    for i in range(1,len(array)):

        if array[i-1] != array[i]:
            if flag == False:
                flag = True
                time_1 = i
                T = array[i-1]
                F = array[i]
            else:
                flag = False
                array[time_1 : i] = T
        elif flag == True and T ==3 and F == 0:
            pass
        elif flag == True and i - time_1 >point:#基本は150
            flag = False
        else:
            pass
    return array


#スタートを覗く０の値を持つ区間を３に変更
def trg_recover_2(array):
    where_are_three, = np.where(array==3)
    first_trg = where_are_three[0]
    arr_cp = array.copy()

    idxs = np.arange(arr_cp.shape[0])
    arr_cp[np.logical_and(arr_cp==0, idxs>first_trg)] = 3
    return arr_cp

def trg_recover_3(array,point):
    arr_cp = array.copy()
    count = 0
    flag_1 = False
    flag_2 = False
    flag_3 = False
    for i in range(1, len(arr_cp)):
#モザイクによるトリガ変化を訂正
        if arr_cp[i-1] == 2 and arr_cp[i] == 1:
            count = count + 1
        if count == 3 and arr_cp[i-1] == 1 and arr_cp[i] == 3:
            arr_cp[i : i + point] = 3 #訂正範囲指定，基本は2s分のptを指定
            count =0
#3→2→3で変化している区間を3に変更
        if arr_cp[i-1] == 3 and arr_cp[i] == 2:
            flag_1 = True
            time_1 = i-1
        if arr_cp[i-1] == 2 and arr_cp[i] == 1 and flag_1:
            flag_1 = False
        if arr_cp[i-1] == 2 and arr_cp[i] == 3 and flag_1:
            arr_cp[time_1:i] = 3
            flag_1 = False
#1→2→3で変化している区間を3に変更
        if arr_cp[i-1] == 1 and arr_cp[i] == 2:
            flag_2 = True
            time_2 = i-1
        if arr_cp[i-1] == 2 and arr_cp[i] == 1 and flag_2:
            flag_2 = False
        if arr_cp[i-1] == 2 and arr_cp[i] == 3 and flag_2:
            arr_cp[time_2:i] = 3
            flag_2 = False
#3→1→3で変化している区間を3に変更
        if arr_cp[i-1] == 3 and arr_cp[i] == 1:
            flag_3 = True
            time_3 = i-1
        if arr_cp[i-1] == 1 and arr_cp[i] == 2 and flag_3:
            flag_3 = False
        if arr_cp[i-1] == 1 and arr_cp[i] == 3 and flag_3:
            arr_cp[time_3:i] = 3
            flag_3 = False
    return arr_cp

def cutdata(matfile, label, outdir, Fs, mergin_f,mergin_b, ECoG_ch, trial):
    arrays = {}
    f = h5py.File(matfile, 'r')
    for k, v in f.items():
        arrays[k] = np.array(v)

# 文字列jsX_exp_Y を取得
    exp = os.path.basename(matfile).replace('.mat', '')
# mergin[ms] をm_sample[pt] に変換
# 切り出し区間[m:n]を[m-mergin_f:n+mergin_b]に拡張するため
    m_sample_f = int(Fs * (mergin_f/1000))
    m_sample_b = int(Fs * (mergin_b/1000))
# 出力先がない場合は作成
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
# イベント修正
    sound_ch = 133
    voice_ch = 137
    trg_ch = 139
    EOG_H = 129
    EOG_V = 131
    Data = arrays['ans'].transpose(1,0)
    print("Data.shape",Data.shape)
    sound_ = (Data[sound_ch].transpose())*0.01
    voice_ = Data[voice_ch].transpose()
    trg_ = Data[trg_ch].transpose()
    ECoG = Data[1:(ECoG_ch+1),::].transpose()
    EOG = Data[(EOG_H,EOG_V),::].transpose()
    recoverd1 = trg_recover_1(trg_,(Fs/2))#0.5s以下の幅を削除
    recoverd2 = trg_recover_3(recoverd1,(Fs*3))
    recoverd3 = trg_recover_2(recoverd2)
    recoverd4 = trg_recover_3(recoverd3,(Fs*3))
    recoverd5 = trg_recover_3(recoverd4,(Fs*3))
    sound = np.array(sound_, dtype='int16')
    voice = np.array(voice_, dtype='int16')
    trg = np.array(recoverd5, dtype='int16')
    print("ECoG_shape",ECoG.shape)
    print("voice_shape",voice.shape)
    sct = getsections(trg)  #len(sct)=18,len(sct[0])=6
    #print (sct)
    fl = open(label, 'r')
    labels = fl.readlines()
    label = labels[4][:-1].split(' ')
    cnt = 0
    offset_ =int(label[0])#.csvのtriの数に対応ささせるため(2セット目は17-)
    #print(labels)
    for l in range(len(labels)):
       #print(labels[l][:-1])
        label = labels[l][:-1].split(' ')
        if len(label) == 3:  #.csvファイル中の日時等のデータは無視する
            if 'Tri'not in label[0] and 'y' in label[1]:  #yesの文章のみ書き出す
                #print("hoge")
                #print('hoge', int(label[0]) - offset_)
                #print(cnt, label[2],  sct[int(label[0])-1])
                #cutout(voice, sct[int(label[0])-offset_], exp, cnt, label[2], m_sample, outdir, Fs)
                #label[0]=トライアル番号
                #label[1]=yes or no
                #label[2]=文章番号
                print(len(label))
                print(label[0])
                print(label[2])
                print(len(sct))
                print("offset=",offset_)
                #print(sct[int(label[0])-offset_])
                cutout_voice(voice, sct[int(label[0])-offset_], exp, cnt, label[2], m_sample_f, m_sample_b, outdir, Fs)
                cutout_sound(sound, sct[int(label[0])-offset_], exp, cnt, label[2], m_sample_f, m_sample_b, outdir, Fs)

                #voice = cutout_voice(voice, sct[int(label[0])-1], exp, cnt, label[2], m_sample, outdir, Fs)
                #sound = cutout_sound(sound, sct[int(label[0])-1], exp, cnt, label[2], m_sample, outdir, Fs)
                cutout_ECoG(ECoG, sct[int(label[0])-offset_], exp, cnt, label[2], m_sample_f, m_sample_b, outdir, Fs)
                cutout_EOG(EOG, sct[int(label[0])-offset_], exp, cnt, label[2], m_sample_f, m_sample_b, outdir, Fs)
                cutout_event(trg, sct[int(label[0])-offset_], exp, cnt, label[2], m_sample_f, m_sample_b, outdir, Fs)
                cnt += 1
    f.close()
    fl.close()
    return trg_,trg
    #bp()

# ----------------------------Makin前処理関数群-------------------------------
#バイポーラリファレンス用の関数
def bi_ref(ecog,grid_type,grid_list):
    grid_num = len(grid_type)# グリッドの枚数を取得
    #グリッドの電極のみ取り出す
    for i in range(int(len(grid_list)/2)):
        if i==0:
            ecog_ = ecog[grid_list[i]-1:grid_list[i+1]-1+1,:]
        else:
            ecog_ = np.vstack((ecog_,ecog[grid_list[i*2]-1:grid_list[i*2+1]-1+1,:]))
    array = ecog_
    # 各listはリファレンス時の引き算対象となる電極番号
    # グリッドの番号の振り方が2種類しかないためtypeは2種類のみ
    listA_type_0 = [0, 1, 2, 3, 4,
                    0, 1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10,
                    6, 7, 8, 9, 10, 11,
                   12, 13, 14, 15, 16,
                   12, 13, 14, 15, 16, 17,
                   18, 19, 20, 21, 22]
    listB_type_0 = [1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10, 11,
                    7, 8, 9, 10, 11,
                   12, 13, 14, 15, 16, 17,
                   13, 14, 15, 16, 17,
                   18, 19, 20, 21, 22,	23,
                   19, 20, 21, 22,	23]

    listA_type_1 = [0, 1, 2, 3, 4,
                    6, 7, 8, 9, 10, 11,
                    6, 7, 8, 9, 10,
                   12, 13, 14, 15, 16, 17,
                   12, 13, 14, 15, 16,
                   18, 19, 20, 21, 22, 23,
                   18, 19, 20, 21, 22]
    listB_type_1 = [1, 2, 3, 4, 5,
                    0, 1, 2, 3, 4, 5,
                    7, 8, 9, 10, 11,
                    6, 7, 8, 9, 10, 11,
                   13, 14, 15, 16, 17,
                   12, 13, 14, 15, 16, 17,
                   19, 20, 21, 22, 23]

    if grid_num==2:#グリッドが2枚だった場合
        array_24 = array[0:24]
        array_48 = array[24:48]
        if grid_type[0]==0:#1枚目のグリッドが右上カウントだった場合の処理
            array_first = array_24[listA_type_0]-array_24[listB_type_0]
        if grid_type[0]==1:#1枚目のグリッドが左下カウントだった場合の処理
            array_first = array_24[listA_type_1]-array_24[listB_type_1]
        if grid_type[1]==0:#2枚目のグリッドが右上カウントだった場合の処理
            array_second = array_48[listA_type_0]-array_48[listB_type_0]
        if grid_type[1]==1:#2枚目のグリッドが左下カウントだった場合の処理
            array_second = array_48[listA_type_1]-array_48[listB_type_1]
        array_connect = np.append(array_first,array_second,0)
        ecog_remain = np.delete(ecog,slice(grid_list[0]-1,grid_list[1]-1+1),0)
        ecog_remain = np.delete(ecog_remain,slice(grid_list[2]-1-24,grid_list[3]-1+1-24),0)
        ecog_biref_and_remain = np.vstack((array_connect,ecog_remain))
    if grid_num==3:#グリッドが3枚だった場合
        array_24 = array[0:24]
        array_48 = array[24:48]
        array_72 = array[48:72]
        if grid_type[0]==0:#1枚目のグリッドが右上カウントだった場合の処理
            array_first = array_24[listA_type_0]-array_24[listB_type_0]
        if grid_type[0]==1:#1枚目のグリッドが左下カウントだった場合の処理
            array_first = array_24[listA_type_1]-array_24[listB_type_1]
        if grid_type[1]==0:#2枚目のグリッドが右上カウントだった場合の処理
            array_second = array_48[listA_type_0]-array_48[listB_type_0]
        if grid_type[1]==1:#2枚目のグリッドが左下カウントだった場合の処理
            array_second = array_48[listA_type_1]-array_48[listB_type_1]
        if grid_type[2]==0:#3枚目のグリッドが右上カウントだった場合の処理
            array_third = array_72[listA_type_0]-array_72[listB_type_0]
        if grid_type[2]==1:#3枚目のグリッドが左下カウントだった場合の処理
            array_third = array_72[listA_type_1]-array_72[listB_type_1]
        array_first_second = np.append(array_first,array_second,0)
        array_connect = np.append(array_first_second,array_third,0)
        ecog_remain = np.delete(ecog,slice(grid_list[0]-1,grid_list[1]-1+1),0)
        ecog_remain = np.delete(ecog_remain,slice(grid_list[2]-1-24,grid_list[3]-1+1-24),0)
        ecog_remain = np.delete(ecog_remain,slice(grid_list[4]-1-48,grid_list[5]-1+1-48),0)
        ecog_biref_and_remain = np.vstack((array_connect,ecog_remain))

    return ecog_biref_and_remain,array_connect,ecog_remain

# 8帯域別のバンドパスフィルター
def band_pass_iir_8(data,fs,cut_list):
    ch_data = data.shape[0]
    len_data = data.shape[1]
    array = np.zeros((8,ch_data,len_data))
    for i in range(8):
        array[i] = bandpass_iir(data, fs,
                                round(cut_list[i*2],2),
                                round(cut_list[i*2+1],2),
                                axis=1, show=False)
    return array

# 2帯域別のバンドパスフィルター
def band_pass_2(data,fs,cut_list,order):
    ch_data = data.shape[0]
    len_data = data.shape[1]
    array = np.zeros((2,ch_data,len_data))
    for i in range(2):
        array[i] = bandpass_iir(data, fs,
                                round(cut_list[i*2],2),
                                round(cut_list[i*2+1],2),
                                order, axis=1, show=False)
    return array

def band_pass_fir_8(data,fs,cut_list,numtaps):
    ch_data = data.shape[0]
    len_data = data.shape[1]
    array = np.zeros((8,ch_data,len_data))
    for i in range(8):
        array[i] = bandpass_fir(data, fs,
                                round(cut_list[i*2],2),
                                round(cut_list[i*2+1],2),
                                numtaps, axis=1)
    return array

# Bandpass filter using Butterworth Filter
def bandpass_iir(data, fs, f_low, f_high, axis=1, show=False):
   data = np.asarray(data)
   # Design filter
   fd = 10
   wp = np.array([f_low, f_high])/(fs/2)
   ws = np.array([f_low-fd, f_high+fd])/(fs/2)
   ord, wn = signal.buttord(wp, ws, 3, 40)
   sos = signal.butter(ord, wn, btype='bandpass', output='sos')
   if show:
       plot_freq_response_sos(sos, fs)
   return signal.sosfiltfilt(sos, data, axis=axis)

# Bandpass filter using Butterworth Filter
def bandpass_fir(data, fs, f_low, f_high, numtaps, axis=1):

    data = np.asarray(data)
    wp = np.array([f_low, f_high])
    #wp = np.array([f_low, f_high])/(fs/2)

    # Design filter
    #print([f_low, f_high])
    #print(wp)
    #print(fs)
    b = signal.firwin(numtaps, wp, pass_zero=False,fs=fs)
    lp = signal.firwin(numtaps, f_low/(fs/2), pass_zero=False, fs=fs)
    hp = -signal.firwin(numtaps, f_high/(fs/2), pass_zero=False, fs=fs)
    #print(hp.shape)
    hp[int(numtaps / 2)] = hp[int(numtaps / 2)] + 1
    bp = - (lp + hp)
    bp[int(numtaps/2)] = bp[int(numtaps/2)] + 1

    

    #print(b.shape)
    #print(b)
    #return scipy.signal.lfilter(b, 1, data, axis)
    #outdata = np.zeros(data.shape)

    #for i in range(len(data)):
    #    print(i)
    #    print(data[i])

    return signal.filtfilt(b, 1, data, axis)   
    #filtfiltは位相がずれない

# Z-scoreによる標準化
# length_window[s]
def z_scored(ecog,length_window,fs):
    array = ecog
    pt_window = int(length_window*fs) # zスコアを計算する区間内のサンプル数
    array_zero = np.zeros(array.shape,dtype=np.complex)
    #分割しきれなかった箇所の一番目のインデックス
    num_last = int(array.shape[1]/pt_window)*(pt_window-1)
    for i in range(int(array.shape[1]/pt_window)):
        array_zero[::,(pt_window-1)*i:(pt_window-1)*(i+1)] = \
        stats.zscore(array[::,(pt_window-1)*i:(pt_window-1)*(i+1)],axis=1)
    array_zero[::,num_last::] = \
    stats.zscore(array[::,num_last::],axis=1)
    return array_zero

# ヒルベルト変換
def hilbert(data, axis):
    array = data
    axis = axis
    array_hilbert = signal.hilbert(array,axis)
    return array_hilbert

#脳波読み込み
#elect_first=ECOG測定開始電極CH，elect_last＝削除開始チャネル,elect_exclude＝除外チャネル
def select_electrode(wavepath,list):
    arrays = {}
    f = h5py.File(wavepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
    ECOG = arrays["ans"].transpose(1,0)
    for i in range(int(len(list)/2)):
        if i==0:
            ecog1 = ECOG[list[i]:list[i+1]+1,:]
        else:
            ecog1 = np.vstack((ecog1,ECOG[list[i*2]:list[i*2+1]+1,:]))
    ecog2 = ecog1[:,100:]
    return ecog2,ECOG

def read_wave(wavepath,elect_first,elect_last,elect_exclude):
    arrays = {}
    f = h5py.File(wavepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
    ECOG = arrays["ans"].transpose(1,0)
    ECOG___ = np.delete(ECOG, np.s_[elect_last:], 0)#後半の電極削除（トリガ等を含む）
    ECOG__ = np.delete(ECOG___, [elect_exclude], 0)#詰まった電極削除
    ECOG_ = ECOG__[elect_first:,100:] #最初のノイズ除去，tumor,inhemi電極削除
    ECOG_ref = np.zeros(ECOG_.shape,dtype = float)
    ECOG_org = ECOG[:,100:]
    return ECOG_,ECOG_org

def read_trg_wave(wavepath,rate,list):
    arrays = {}
    f = h5py.File(wavepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
#脳波からansを取り出し，そこからトリガを取り出し，int型に変更
    array_all = arrays['ans'].transpose(1,0)
    array_down = signal.decimate(array_all,rate)#ダウンサンプリング
    trg = array_all[139,::rate]#トリガはダウンサンプリングせずに記録
    trg_int = trg.astype(int)
    trg_mat = trg_int[100:]
    for i in range(int(len(list)/2)):
        if i==0:
            ecog1 = array_down[list[i]:list[i+1]+1,:]
        else:
            ecog1 = np.vstack((ecog1,array_down[list[i*2]:list[i*2+1]+1,:]))
    ecog2 = ecog1[:,100:]
    return trg_mat,ecog2

#トリガ読み込み
def read_trg(trgpath):
    global input_csv,trg_
    input_csv = np.loadtxt(trgpath,delimiter=',')
    #input_csv = pd.read_csv(filepath_or_buffer=trgpath, encoding="shift_jis", sep=",")
    #trg_ = input_csv.values
    #trg = trg_.transpose((1,0))
    #trg = trg.reshape(-1)
    trg = input_csv[100:]
    return trg
# 脳波，眼電図，傾聴音声，発声音声，イベント抽出(各データ最初の100サンプルは削除)
def read_ECoG_EOG_sound_event(wavepath,list):
    arrays = {}
    f = h5py.File(wavepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
#脳波からansを取り出し，そこからトリガを取り出し，int型に変更
    array_all = arrays['ans'].transpose(1,0)
    trg = array_all[139]
    trg_int = trg.astype(int)
    trg_mat = trg_int[100:]
    for i in range(int(len(list)/2)):
        if i==0:
            ecog1 = array_all[list[i]:list[i+1]+1,:]
        else:
            ecog1 = np.vstack((ecog1,array_all[list[i*2]:list[i*2+1]+1,:]))
    ecog2 = ecog1[:,100:]
    eog = array_all[129:133,100:]
    sound_L = array_all[133:135,100:]
    sound_R = array_all[135:137,100:]
    voice = array_all[137:139,100:]
    return trg_mat,ecog2,eog,sound_L,sound_R,voice,array_all[:,100:]


#リファレンス適用（全電極平均）
def reference(ECOG_):
    s = sum(ECOG_)
    N = len(ECOG_)
    reference = np.mean(data, axis=ch_axis)
    ECOG_ref = ECOG_-reference
    return ECOG_ref


# In[7]:


#トリガ内の３→２，１→３になるタイミングのインデックスを取得
def trg_tim_all(trg,before_1,after_1,before_2,after_2 ):
    trg_change_ = []
    for i in range(len(trg)-1):
        if trg[i]==before_1 and trg[i+1]==after_1:
            trg_change_.append(i)
        if trg[i]==before_2 and trg[i+1]==after_2:
            trg_change_.append(i)
    trg_change = np.array(trg_change_)
    return trg_change


# In[8]:


#trg_allから，startとendのタイミングを抽出
#傾聴タスク_st=0,ed=1想像タスク_st=2,ed=3発声タスク_st＝4、ed=5
def trg_st_ed(trg_all,task_type):
    start_ = []
    end_ = []
    start_ = trg_all[task_type::6]
    end_ = trg_all[(task_type + 1)::6]
    start = np.array(start_)
    end = np.array(end_)
    start_rest = start - 2700
    task_num = len(start-end)
    return start,start_rest,end,task_num

#タスクを２つ選択して，そのタスク間の脳波とラベルを出力する，ex）傾聴ー発声ならtype_num_0=0,type_num_1 =2
def select_task(wave,trg,trg_list,type_num_0,type_num_1):
    global x_0,x_1,x_01,y_0,y_1,y_01
    x = wave.copy()
    y = trg.copy()
    range_list = ["trg_list[6*i]","trg_list[6*i+1]",#傾聴
                "trg_list[6*i+2]","trg_list[6*i+3]",#想像
                "trg_list[6*i+4]","trg_list[6*i+5]"]#発声
    x_01 = {}
    y_01 = []
    for i in range(int(len(trg_list)/6)):#trg_listを1/2するとタスクの回数，さらに1/3すると１サイクルの数
        if i == 0:
            x_0 = x[:,eval(range_list[2*type_num_0]):eval(range_list[2*type_num_0+1])]
            y_0 = np.zeros(y[eval(range_list[2*type_num_0]):eval(range_list[2*type_num_0+1])].shape,int)
            x_1 = x[:,eval(range_list[2*type_num_1]):eval(range_list[2*type_num_1+1])]
            y_1 = np.ones(y[eval(range_list[2*type_num_1]):eval(range_list[2*type_num_1+1])].shape,int)
            x_01 = np.hstack((x_0,x_1))
            y_01 = np.hstack((y_0,y_1))
        else:
            x_0 = x[:,eval(range_list[2*type_num_0]):eval(range_list[2*type_num_0+1])]
            y_0 = np.zeros(y[eval(range_list[2*type_num_0]):eval(range_list[2*type_num_0+1])].shape,int)
            x_1 = x[:,eval(range_list[2*type_num_1]):eval(range_list[2*type_num_1+1])]
            y_1 = np.ones(y[eval(range_list[2*type_num_1]):eval(range_list[2*type_num_1+1])].shape,int)
            x_01 = np.hstack((x_01,x_0,x_1))
            y_01 = np.hstack((y_01,y_0,y_1))
    return x_01, y_01

# In[9]:


#スペクトログラムの計算、タスク間のノーマライズパワー計算
def spec_50ms(ECOG_filt,ch,fs):
    width = 0.05*fs #fs時に50ms区間内の[pt]
    lap = 0.025*fs #fs時の25ms[pt]
    task = []
    if fs == 1200:
        for i in range(ch):
         #frontal=0,temporal=24,occipital=48
                data_ = ECOG_filt[i]
                f, t, Sxx = signal.spectrogram(data_, fs, nperseg=width, noverlap=lap , window=np.hanning(width), scaling = 'spectrum')
                y = sum(Sxx[3:9,:])#計算するパワーの周波数帯域を決定する(60-180)
                if i == 0:
                    task = np.zeros((0,len(y)))
                    power = np.vstack((task, y))
                else:
                    power = np.vstack((power, y))
    if fs == 600:
        for i in range(ch):
         #frontal=0,temporal=24,occipital=48
                data_ = ECOG_filt[i]
                f, t, Sxx = signal.spectrogram(data_, fs, nperseg=width, noverlap=lap , window=np.hanning(width), scaling = 'spectrum')
                y = sum(Sxx[6:18,:])#計算するパワーの周波数帯域を決定する(60-180)
                if i == 0:
                    task = np.zeros((0,len(y)))
                    power = np.vstack((task, y))
                else:
                    power = np.vstack((power, y))
    return power

#スペクトログラムの計算、タスク間のノーマライズパワー計算
def spec_250ms(ECOG_filt,ch,fs):
    width = 0.25*fs#１区間２５＊１０ms
    lap = 0.225*fs#オーバーラップ225ms
    task = []
    if fs == 1200:
        for i in range(ch):
         #frontal=0,temporal=24,occipital=48
                data_ = ECOG_filt[i]
                f, t, Sxx = signal.spectrogram(data_, fs, nperseg=width, noverlap=lap , window=np.hanning(width), scaling = 'spectrum')
                y = sum(Sxx[15:45,:])#計算するパワーの周波数帯域を決定する(60-180)
                if i == 0:
                    task = np.zeros((0,len(y)))
                    power = np.vstack((task, y))
                else:
                    power = np.vstack((power, y))
    if fs == 600:
        for i in range(ch):
         #frontal=0,temporal=24,occipital=48
                data_ = ECOG_filt[i]
                f, t, Sxx = signal.spectrogram(data_, fs, nperseg=width, noverlap=lap , window=np.hanning(width), scaling = 'spectrum')
                y = sum(Sxx[30:90,:])#計算するパワーの周波数帯域を決定する(60-180)
                if i == 0:
                    task = np.zeros((0,len(y)))
                    power = np.vstack((task, y))
                else:
                    power = np.vstack((power, y))
    return power
#各セットごとのarrayを列方向に結合
def connect_array(array,ch):
    for i in range(5):
        if i == 0:
            array_ = np.zeros((ch,0))
            array_connect = np.hstack((array_,array[i]))
        else:
            array_connect = np.hstack((array_connect,array[i]))
    return array_connect

#各セットごとのlistを列方向に結合
def connect_list(list):
    a = list[0].tolist()
    b = list[1].tolist()
    c = list[2].tolist()
    d = list[3].tolist()
    e = list[4].tolist()
    list_connect = np.hstack((a,b,c,d,e))
    #list_connect = np.hstack((a,b,c))
    return list_connect

#発声，無発声の正解ラベル生成
def make_label_50(start,end,trg,task_num,op):
    no_in_block = op#パワー計算において，重なる要素の数，ex)0.075*1200=90
    label_long = np.zeros((trg.shape), dtype = int)
    label_short = np.zeros((int(trg.shape[0]/no_in_block)-1,), dtype = int)
    j=0
    #最初に発声区間のインデックスに１を格納
    for i in range(task_num):
        label_long[start[i] : end[i]+1] = 1
    #圧縮前のラベルから，60pt区間で30ptずらしながら，区間内の値を計算
    for i in range(0,trg.shape[0],no_in_block):
        if sum(label_long[i:i+no_in_block*2])>=no_in_block/2:#閾値(区間の半分以上が１の場合)
            label_short[j]=1
        j=j+1
    return label_long,label_short

#タスク-休憩の正解ラベル生成
def make_label_250(start,end,trg,task_num,op,pt):
    overlap = op#パワー計算において，重なる要素の数，ex)0.075*1200=90
    point = pt#1200*0.25
    remain_overlap = point-overlap
    label_long = np.zeros((trg.shape), dtype = int)
    trial_num = int((trg.shape[0]-point)//(remain_overlap))+1
    label_short = np.zeros(trial_num)
    #最初にタスク区間のインデックスに１を格納
    for i in range(task_num):
        label_long[start[i] : end[i]+1] = 1

    j = 0
    for i in range(0,label_long.shape[0]-point,remain_overlap):
        label_short[j] = Decimal(str(sum(label_long[i:i+point])/point)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
        j=j+1
    return label_long, label_short

#０，１でできたラベルを区間ごとに多数決をとって短くする
def make_label_X(label_l,pt,op):
    overlap = op#パワー計算において，重なる要素の数，250ms区間の場合２７０pt
    point = pt#２５０ms区間だと３００pt
    remain_overlap = point-overlap
    label_long = label_l.copy()
    trial_num = int((label_long.shape[0]-point)//(remain_overlap))+1
    label_short = np.zeros(trial_num)
    j = 0
    for i in range(0,label_long.shape[0]-point,remain_overlap):
        label_short[j] = Decimal(str(sum(label_long[i:i+point])/point)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
        j=j+1
    return label_short


#ノルムを用いた正規化
def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2

#発声区間と無発声区間を１：１にする
#具体的には発声区間の前に発声区間と同じ長さの無発声区間を設ける
def equaler(power,label,idx,):
    idx_se = idx[1::2]-idx[0::2]#各発声区間の長さを計算

    for i in range(len(idx_se)):
        if i == 0:
            array_connect_p = power[:,idx[0]-(int(idx_se[0]/2)):idx[1]+(int(idx_se[0]/2))]
            array_connect_l = label[idx[0]-(int(idx_se[0]/2)):idx[1]+(int(idx_se[0]/2))]
        else:
            array_connect_p = np.hstack((array_connect_p,power[:,idx[i*2]-(int(idx_se[i]/2)):idx[i*2+1]+(int(idx_se[i]/2))]))
            array_connect_l = np.hstack((array_connect_l,label[idx[i*2]-(int(idx_se[i]/2)):idx[i*2+1]+(int(idx_se[i]/2))]))
    return array_connect_p, array_connect_l

#行列の結合
def make_hankel(fv,label,ch):
    X = np.zeros((ch*9,((fv.shape[1]-1)-4)-4+1))
    Y = np.zeros(((fv.shape[1]-1)-4)-4+1)
    for i in range(fv.shape[1]-1-4-4+1):
        for j in range(9):
            X[j*ch:(j+1)*ch,i] = fv[:,i+j]
        Y[i] = label[4+i]
    return X,Y

#トリガの短い変化を平滑化（Xのポイント分が基準）
def marumekomi(pred,X):
    array = pred.copy()
    flag = False
    global start
    for i in range(1,len(array)):
        if array[i-1] != array[i]:
            if flag == False:
                before = array[i-1]
                after = array[i]
                start = i
                flag = True
            else:
                end = i-1
                if end - start < X:
                    array[start:i] = before
                    flag = False
                else:
                    flag = False
    return array
#分類結果にマージンをつける
def margin(pred,X):
    array = pred.copy()
    flag = False
    count = 0
    global start
    for i in range(1,len(array)):

        if array[i-1] != array[i]:
            if array[i-1]==0 and array[i]==1:
                start = i-X
                flag = False
            elif array[i-1]==1 and array[i]==0:
                end = i
                array[start:end] = 1
                flag = True

        elif flag == True:
            if count<X:
                array[i]=1
                count = count+1
            else:
                flag = False
                count = 0
    return array
"""
        if array[i-1]==0 and array[i]==1:
            start = i-X
            flag = False

        elif array[i-1]==1 and array[i]==0:
            end = i
            array[start:end] = 1
            flag = True

        elif flag == True:
            if count<X:
                array[i]=1
                count = count+1
            else:
                flag = False
                count = 0
"""
#脳波をtime[ms]区間ごとにoverlap[ms]のオーバーラップで分割し
#区間ごとで並べた３次元配列にする
#k.modeは多数決を取っているのでどんな値でも可能
#下のやつはラベルの生成方法は足し算して平均を取っているので，0,1のラベルを入力すること！
def make_reshape(ecog_, label_, time, overlap_, fs):
    global ecog, label, point, trial_num, overlap, remain_overlap
    ecog = ecog_.copy()
    label = label_.copy()
    point = int(time*10**-3*fs)
    overlap = int(overlap_*10**-3*fs)
    remain_overlap = point-overlap
    trial_num = int((ecog.shape[1]-point)//remain_overlap)+1
    ch = ecog.shape[0]
    re_ecog = np.zeros((trial_num,ch,point))
    re_label = np.zeros(trial_num)
    j = 0
    for i in range(0,ecog.shape[1]-point,remain_overlap):
        re_ecog[j,:,:] = ecog[:,i:i+point]
        k = 0
        k = stats.mode(label[i:i+point], axis=None)
        re_label[j] = k.mode[0]
        #re_label[j] = Decimal(str(sum(label[i:i+point])/point)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
        j=j+1
    return re_ecog, re_label

#CSPによる特徴抽出
def csp(ecog_,label_,n_features):
    ecog = ecog_.copy()
    label = label_.copy()
    X1 = ecog[np.where(label==1)]#csp_１２の場合想像＝０，発声＝１
    c1_all = [np.cov(X1[i], bias=True) for i in range(len(X1))]
    c1 = np.mean(c1_all, axis=0)

    X0 = ecog[np.where(label==0)]
    c0_all = [np.cov(X0[i], bias=True) for i in range(len(X0))]
    c0 = np.mean(c0_all, axis=0)

    w, v = scipy.linalg.eig(c1, c0)
    sort_idx = np.argsort(w)[::-1]
    w, v = w[sort_idx], v[:, sort_idx]

    v_filt = np.concatenate(
            (v[:, :n_features // 2], v[:, -n_features // 2:]),
            axis=1)
    f = np.var(np.tensordot(v_filt.T, ecog, axes=(1, 1)), axis=2)
    f = np.log(f)
    return w,v,f,sort_idx#ベクトルはソート済み

#発声区間，無発声区間を１：１にするコードの３次元配列版（発声区間の前後を抽出）
def equaler_3dim(ecog_,label_,idx,):
    ecog = ecog_.copy()
    label = label_.copy()
    idx_se = idx[1::2]-idx[0::2]#各発声区間の長さを計算

    for i in range(len(idx_se)):
        if i == 0:
            array_connect_e = ecog[idx[0]-(int(idx_se[0]/2)):idx[1]+(int(idx_se[0]/2)),:,:]
            array_connect_l = label[idx[0]-(int(idx_se[0]/2)):idx[1]+(int(idx_se[0]/2))]
        else:
            array_connect_e = np.vstack((array_connect_e,ecog[idx[i*2]-(int(idx_se[i]/2)):idx[i*2+1]+(int(idx_se[i]/2)),:,:]))
            array_connect_l = np.hstack((array_connect_l,label[idx[i*2]-(int(idx_se[i]/2)):idx[i*2+1]+(int(idx_se[i]/2))]))
    return array_connect_e, array_connect_l

#rateずつ間を開けてサンプリング
def downsampling_wave(data,rate):
    x = data.copy()
    x_down =  x[:,::rate]
    return x_down

def downsampling_label(data,rate):
    x = data.copy()
    x_down = x[::rate]
    return x_down

# フィルタ関連
def lowpassFilter(data, fs, fc_low, axis, order):
    b,a =signal.butter(order, (fc_low * 1.0) / (fs * 0.5), btype ='lowpass')
    filtdata = signal.filtfilt(b, a, data, axis)
    return filtdata

def highpassFilter(data, fs, fc_high):
    b,a = signal.butter(3, (fc_high * 1.0) / (fs * 0.5), btype='highpass')
    filtdata = signal.filtfilt(b, a, data, axis = 0)
    return filtdata

def notchFilter(data,fs,freq):
    Q = 30.0
    b,a = signal.iirnotch(freq/(fs/2),Q)
    filtdata = signal.filtfilt(b, a, data, axis = 0)
    return filtdata

# Plot the frequency response.
def plot_freq_response_ba(b, a, fs):
    w, h = signal.freqz(b, a, worN=8000)
    plt.plot(fs / 2 * w / np.pi, 20 * np.log10(np.abs(h) + 1e-200), 'b')
    plt.title('Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(0, fs / 2)
    plt.ylabel('Gain [dB]')
    plt.grid()
    plt.show()


def plot_freq_response_sos(sos, fs):
    w, h = signal.sosfreqz(sos, worN=8000)
    plt.plot(fs / 2 * w / np.pi, 20 * np.log10(np.abs(h) + 1e-200), 'b')
    plt.title('Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(0, fs / 2)
    plt.ylabel('Gain [dB]')
    plt.grid()
    plt.show()


# Lowpass filter using Butterworth Filter
def lowpass_iir(data, fs, fc, order=3, axis=0, show=False):
    data = np.asarray(data)

    fc_norm = fc / (fs / 2)

    # Design filter
    sos = signal.butter(order, fc_norm, btype='lowpass', output='sos')

    if show:
        plot_freq_response_sos(sos, fs)

    return signal.sosfiltfilt(sos, data, axis=axis)


# Lowpass filter using FIR Filter
def lowpass_fir(data, fs, fc, taps, axis=0, show=False):
    data = np.asarray(data)

    fc_norm = fc / (fs / 2)
    b = signal.firwin(taps, fc_norm, window='hann')
    a = [1]

    if show:
        plot_freq_response_ba(b, a, fs)

    return signal.filtfilt(b, a, data, axis=axis)


# Highpass filter using Butterworth Filter
def highpass_iir(data, fs, fc, order=3, axis=0, show=False):
    data = np.asarray(data)

    fc_norm = fc / (fs / 2)

    # Design filter
    sos = signal.butter(order, fc_norm, btype='highpass', output='sos')

    if show:
        plot_freq_response_sos(sos, fs)

    return signal.sosfiltfilt(sos, data, axis=axis)






# Notch filter
def notch_iir(data, fs, fn, Q=30, axis=1, show=False):
    data = np.asarray(data)

    fn_norm = fn / (fs / 2)

    # Design filter
    b, a = signal.iirnotch(fn_norm, Q)

    if show:
        plot_freq_response_ba(b, a, fs)

    return signal.filtfilt(b, a, data, axis=axis)



####################################################################################################################################################
#トリガ読み込み
def read_trg_mat(read_trgpath):
    arrays = {}
    f = h5py.File(read_trgpath)
    for k, v in f.items():
        arrays[k] = np.array(v)
#脳波からansを取り出し，そこからトリガを取り出し，int型に変更
    array_tr = arrays['ans'].transpose(1,0)
    js1_trg_1 = array_tr[139,:]
    trg_mat = js1_trg_1.astype(int)
    return trg_mat


# In[4]:



"""
#モザイクによるトリガ変化を訂正
def trg_recover_3(arr_cp,point):
    count = 0
    for i in range(1, len(arr_cp)):
        if arr_cp[i-1] == 2 and arr_cp[i] == 1:
            count = count + 1
        if count == 3 and arr_cp[i-1] == 1 and arr_cp[i] == 3:
            arr_cp[i : i + point] = 3 #訂正範囲指定，基本は６０００
            count =0
#3→2→3で変化している区間を3に変更
    flag = False
    for i in range(1,len(arr_cp)):
        if arr_cp[i-1] == 3 and arr_cp[i] == 2:
            flag = True
            time = i-1
        if arr_cp[i-1] == 2 and arr_cp[i] == 1 and flag == True:
            flag = False
        if arr_cp[i-1] == 2 and arr_cp[i] == 3 and flag == True:
            arr_cp[time:i] = 3
            flag = False
    return arr_cp

"""
# In[7]:


#保存前の波形確認
def trg_confilm(arr_cp):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20,4))
    ax1.plot(arr_cp[0:200000], linewidth=2)
    ax2.plot(arr_cp[200000:400000], linewidth=2)
    ax3.plot(arr_cp[400000:600000], linewidth=2)
    ax4.plot(arr_cp[600000:900000], linewidth=2)
    plt.show()


###松井さんコード
def plot_16ch(data,fs,ch_list,start=None,stop=None,title=None,vspace=80,color_excess=0):
#data=(ch*n_sample)
    vmargin = vspace
    n_ch, n_sample = data.shape
    time = np.linspace(0, n_sample / fs, n_sample)
    with plt.style.context(('default', 'seaborn-whitegrid')):
        fig, ax = plt.subplots(figsize=(15, 12))
        for ch in range(n_ch):
            num = (n_ch - 1) - ch
            if color_excess and np.any(np.abs(data[num]) > color_excess):
                # idx = np.where(np.abs(data[fs*start:fs*stop,num]) > 75)[0] + fs*start
                # ax.plot(time,data[:,num]+75*ch,color="hotpink")
                # ax.hlines(100+75*ch,time[0],time[-1],color="lightpink")
                ax.plot(
                    time,
                    data[num] + vspace * ch,
                    color="cornflowerblue",
                    zorder=1)
                # ax.scatter(time[idx],data[idx, num]+75*ch, color="deeppink", zorder=2, marker=".")
            else:
                ax.plot(time, data[num] + vspace * ch, color="blue", zorder=1)
        ymin = -vmargin
        ymax = vspace * (n_ch - 1) + vmargin
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.linspace(vspace * (n_ch - 1), 0, n_ch, endpoint=True))
        ax.set_yticklabels(ch_list)
        if start is not None:
            ax.axvline(start, linestyle="dashed")
        if stop is not None:
            ax.axvline(stop, linestyle="dashed")
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        return fig

def eegplot(data,fs,ch_list=None,per_page=16,start=None,stop=None,path=None,title=None,vspace=80,color_excess=0):
    #This function plots EEG data (scale: uV)
    #Arguments:
        #data: ndarray of shepe (samples, ch)
    n_ch = len(data)
    if ch_list is None:
        ch_list = list(range(1, n_ch + 1))
    elif len(ch_list) != n_ch:
        raise ValueError(
            f'Number of channels mismatch (Label:{len(ch_list)} Data:{n_ch})')
    begins = np.arange(0, n_ch, per_page)
    ends = begins + per_page
    ends[-1] = n_ch
    pages = len(begins)
    for n, (begin, end) in enumerate(zip(begins, ends)):
        fig = plot_16ch(data[begin:end].T, fs, ch_list[begin:end], start, stop,'hoge', vspace, color_excess)
        if path:
            print("path=on")
            path_base, path_ext = os.path.splitext(path)
            #plt.savefig(f'{path_base}_{n+1}{path_ext}' if pages > 1 else path)
        else:
            print("else")
            plt.show()
        plt.close(fig)
# In[10]:

if __name__ == '__main__':
    pass
