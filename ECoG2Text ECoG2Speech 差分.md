# ECoG2TextとECoG2Speechの差分

## top

o：ファイル有  
x：ファイル無

| ファイル | ECoG2Text | ECoG2Speech | 備考(差分) |
| --- | --- | --- | --- |
| ./bin/sclite | o   | o   | \-  |
| ./conf/config\_xA000.ini | o   | x   | [conf](#conf) |
| ./conf/config\_xA001.ini | x   | o   | [conf](#conf) |
| ./conf/config\_xA002.ini | x   | o   | [conf](#conf) |
| ./ecog2text/my\_module/\_\_init\_\_.py | o   | o   | \-  |
| ./ecog2text/my\_module/my\_function\_kome.py | o   | o   | [my\_func](#my_function) |
| ./ecog2text/my\_module/my\_function\_orig.py | o   | o   | [my\_func](#my_function) |
| ./ecog2text/my\_module/my\_function.py | o   | o   | [my\_func](#my_function) |
| ./ecog2text/my\_module/script.py | o   | x   | 不使用のため削除 |
| ./ecog2text/my\_module/script.sh | o   | x   | 不使用のため削除 |
| ./ecog2text/my\_module/script0.py | o   | x   | 不使用のため削除 |
| ./ecog2text/my\_module/script00.py | o   | x   | 不使用のため削除 |
| ./ecog2text/python\_speech\_features/\_\_init\_\_.py | o   | o   | \-  |
| ./ecog2text/python\_speech\_features/base.py | o   | o   | \-  |
| ./ecog2text/python\_speech\_features/sigproc.py | o   | o   | \-  |
| ./ecog2text/model.py | o   | o   | \-  |
| ./ecog2text/mytransformer.py | o   | o   | \-  |
| ./ecog2text/preprocess.py | o   | o   | [prepro](#preprocess) |
| ./ecog2text/preprocess\_orig.py | x   | o   | [prepro](#preprocess) |
| audio\_reconstruction.py | x   | o   | ParallelWaveGANで音声再合成するコードを作成    |
| Dockerfile | o   | o   | [dockerfile](#dockerfile)    |
| ecog2audio\_eval.py | x   | o   | [eval](#eval)    |
| ecog2audio\_train.py | x   | o   | [train](#train)    |
| ecog2text\_eval.py | o   | o   | [eval](#eval)    |
| ecog2text\_train.py | o   | o   | [train](#train)    |
| make\_sample\_data.py | o   | o   | [make_sample](#make_sample_data)    |
| README.md | o   | o   | \-    |
| requirements.txt | o   | x   | 未作成 |

## conf

- （xA000とxA001の差分）
    - 音声特徴量（MFCCとMelSpectrogram）を選択できるように項目を追加  
        ![スクリーンショット 2023-09-03 15.10.58.png](../_resources/スクリーンショット%202023-09-03%2015.10.58.png)
    - MFCCからaudio\_featに変更
    - L1 loss の項目を追加  
        ![スクリーンショット 2023-09-03 15.11.15.png](../_resources/スクリーンショット%202023-09-03%2015.11.15.png)
- （xA001とxA002の差分）
    - trf\_flagを変更してBLSTM(=False)とTransformer(=True)を切り替え  
        ![スクリーンショット 2023-09-03 15.11.34.png](../_resources/スクリーンショット%202023-09-03%2015.11.34.png)

[↑ top](#top)

## my\_function

（メモ）my\_functionとmy\_function\_origは(他のところで呼び出していないなら)削除しても問題なさそう

- ECoG2TextとECoG2Speech間の差分はなし
    
- （my\_function\_origとmy\_functionの差分
    
    - import 内容の変更  
        ![スクリーンショット 2023-09-03 15.34.50.png](../_resources/スクリーンショット%202023-09-03%2015.34.50.png)
- （my\_function\_origとmy\_function\_komeの差分）
    
    - import 内容の変更  
        ![スクリーンショット 2023-09-03 15.27.44.png](../_resources/スクリーンショット%202023-09-03%2015.27.44.png)
    - バンドパスフィルタを掛ける部分のコードの追記  
        ![スクリーンショット 2023-09-03 15.28.00.png](../_resources/スクリーンショット%202023-09-03%2015.28.00.png)

[↑ top](#top)

## preprocess

- ECoG2Text の ./ecog2text/preprocess.py と ECoG2Speech の ./ecog2text/preprocess\_orig.py は差分なし
    
- （ECoG2Text の ./ecog2text/preprocess.py と ECoG2Speech の ./ecog2text/preprocess.py の差分）
    
    - import 内容の変更  
        ![スクリーンショット 2023-09-03 15.58.39.png](../_resources/スクリーンショット%202023-09-03%2015.58.39.png)
    - mel spectrogram の図を作るためのコードを追加  
        ![スクリーンショット 2023-09-03 15.58.47.png](../_resources/スクリーンショット%202023-09-03%2015.58.47.png)  
        ![スクリーンショット 2023-09-03 15.58.50.png](../_resources/スクリーンショット%202023-09-03%2015.58.50.png)
    - mfccのパラメータをconfファイルへ移動したことに伴うコードの変更  
        ![スクリーンショット 2023-09-03 15.58.55.png](../_resources/スクリーンショット%202023-09-03%2015.58.55.png)
    - mel spectrogram を計算するコードを追加  
        ![スクリーンショット 2023-09-03 15.58.59.png](../_resources/スクリーンショット%202023-09-03%2015.58.59.png)  
        ![スクリーンショット 2023-09-03 15.59.02.png](../_resources/スクリーンショット%202023-09-03%2015.59.02.png)  
        ![スクリーンショット 2023-09-03 15.59.09.png](../_resources/スクリーンショット%202023-09-03%2015.59.09.png)  
        ![スクリーンショット 2023-09-03 15.59.19.png](../_resources/スクリーンショット%202023-09-03%2015.59.19.png)  
        ![スクリーンショット 2023-09-03 15.59.27.png](../_resources/スクリーンショット%202023-09-03%2015.59.27.png)
    - read\_dataでmfccとmel-specの選ばれた方の処理ができるように変更  
        ![スクリーンショット 2023-09-03 15.59.32.png](../_resources/スクリーンショット%202023-09-03%2015.59.32.png)  
        ![スクリーンショット 2023-09-03 15.59.37.png](../_resources/スクリーンショット%202023-09-03%2015.59.37.png)
    - ターミナルにprintされる内容が何かわかるように文言を追加  
        ![スクリーンショット 2023-09-03 15.59.58.png](../_resources/スクリーンショット%202023-09-03%2015.59.58.png)
    - shuffleできるようにするためのコード（不使用）  
        ![スクリーンショット 2023-09-03 16.00.22.png](../_resources/スクリーンショット%202023-09-03%2016.00.22.png)
    - prep\_dataの変更（恐らく不使用）。print内容やmfcc→audio\_featという表記の変更。  
        ![スクリーンショット 2023-09-03 16.01.25.png](../_resources/スクリーンショット%202023-09-03%2016.01.25.png)
    - prep\_dataXの変更。print内容やmfcc→audio\_featという表記の変更。データの長さのずれ（audio\_feat\_padding）を取得する処理を追加。  
        ![スクリーンショット 2023-09-03 16.01.29.png](../_resources/スクリーンショット%202023-09-03%2016.01.29.png)
    - preprocessing（不使用）内のmfcc→audio\_featの変更。
    - preprocessingX内のmfcc→audio\_featの変更。printの追加。引数の追加。
![スクリーンショット 2023-09-03 16.01.34.png](../_resources/スクリーンショット%202023-09-03%2016.01.34.png)

[↑ top](#top)


## dockerfile
- （ECoG2TextとECoG2Speechの差分）
	- ubuntuのバージョンを変更
	- installするモジュールを追加
	- parallel wavegan の git を clone する処理を追加
![スクリーンショット 2023-09-03 16.32.12.png](../_resources/スクリーンショット%202023-09-03%2016.32.12.png)

[↑ top](#top)

## train
- ECoG2Text の ecog2text\_train.py と ECoG2Speech の ecog2text\_train.py は差分なし
- （ecog2text\_train.py と ecog2audio\_train.py の差分）
	- import 内容の追加
![スクリーンショット 2023-09-03 17.25.41.png](../_resources/スクリーンショット%202023-09-03%2017.25.41.png)
	- train（BLSTM）の中身の書き換え。デコーダの削除、データ長を揃える処理の追加。
![スクリーンショット 2023-09-03 17.31.50.png](../_resources/スクリーンショット%202023-09-03%2017.31.50.png)
![スクリーンショット 2023-09-03 17.31.57.png](../_resources/スクリーンショット%202023-09-03%2017.31.57.png)
	- デコーダの削除、mfcc→audio_featへ変更
![スクリーンショット 2023-09-03 17.27.08.png](../_resources/スクリーンショット%202023-09-03%2017.27.08.png)
	- trainTRF（Transformer）の中身の書き換え。デコーダの削除、データ長を揃える処理の追加。
![スクリーンショット 2023-09-03 17.27.21.png](../_resources/スクリーンショット%202023-09-03%2017.27.21.png)
![スクリーンショット 2023-09-03 17.27.37.png](../_resources/スクリーンショット%202023-09-03%2017.27.37.png)
![スクリーンショット 2023-09-03 17.27.54.png](../_resources/スクリーンショット%202023-09-03%2017.27.54.png)
	- cross validation 1 回目のみdumpする処理を追加
![スクリーンショット 2023-09-03 17.28.11.png](../_resources/スクリーンショット%202023-09-03%2017.28.11.png)
	- デコーダの処理削除
![スクリーンショット 2023-09-03 17.28.29.png](../_resources/スクリーンショット%202023-09-03%2017.28.29.png)
	- print内容の変更、引数の追加
![スクリーンショット 2023-09-03 17.28.48.png](../_resources/スクリーンショット%202023-09-03%2017.28.48.png)
	- epoch収束の様子の図を作成するコードを追加
![スクリーンショット 2023-09-03 17.28.53.png](../_resources/スクリーンショット%202023-09-03%2017.28.53.png)







[↑ top](#top)



## eval
- ECoG2Text の ./ecog2text\_eval.py と ECoG2Speech の ./ecog2text\_eval.py は差分なし
- （ecog2text\_eval.py と ecog2audio\_eval.py の差分）
	- import 内容の追加
![スクリーンショット 2023-09-03 16.46.14.png](../_resources/スクリーンショット%202023-09-03%2016.46.14.png)
	- evaluate と smap の中身を大きく削除。（ECoG2Speechではsmapは不使用。evaluateも不使用？）
![スクリーンショット 2023-09-03 16.55.06.png](../_resources/スクリーンショット%202023-09-03%2016.55.06.png)
![スクリーンショット 2023-09-03 16.47.00.png](../_resources/スクリーンショット%202023-09-03%2016.47.00.png)
![スクリーンショット 2023-09-03 16.47.09.png](../_resources/スクリーンショット%202023-09-03%2016.47.09.png)
![スクリーンショット 2023-09-03 16.47.26.png](../_resources/スクリーンショット%202023-09-03%2016.47.26.png)
	- mel spectrogram の図を作成する処理を追加
![スクリーンショット 2023-09-03 16.47.36.png](../_resources/スクリーンショット%202023-09-03%2016.47.36.png)
![スクリーンショット 2023-09-03 16.47.40.png](../_resources/スクリーンショット%202023-09-03%2016.47.40.png)
![スクリーンショット 2023-09-03 16.47.43.png](../_resources/スクリーンショット%202023-09-03%2016.47.43.png)
	- hdf5の read write の関数を追加
![スクリーンショット 2023-09-03 16.47.43のコピー.png](../_resources/スクリーンショット%202023-09-03%2016.47.43のコピー.png)
![スクリーンショット 2023-09-03 16.47.47.png](../_resources/スクリーンショット%202023-09-03%2016.47.47.png)
![スクリーンショット 2023-09-03 16.47.50.png](../_resources/スクリーンショット%202023-09-03%2016.47.50.png)
	- evalの結果(log)の図を作成するsave_evallogsを変更
![スクリーンショット 2023-09-03 16.48.05.png](../_resources/スクリーンショット%202023-09-03%2016.48.05.png)
![スクリーンショット 2023-09-03 16.48.11.png](../_resources/スクリーンショット%202023-09-03%2016.48.11.png)
![スクリーンショット 2023-09-03 16.48.16.png](../_resources/スクリーンショット%202023-09-03%2016.48.16.png)
	- mseを計算する関数を追加
![スクリーンショット 2023-09-03 16.48.24.png](../_resources/スクリーンショット%202023-09-03%2016.48.24.png)
	- mel spectrogram の長さを揃えるための関数を追加
![スクリーンショット 2023-09-03 16.48.24のコピー.png](../_resources/スクリーンショット%202023-09-03%2016.48.24のコピー.png)
	- 変数を追加
![スクリーンショット 2023-09-03 16.48.34.png](../_resources/スクリーンショット%202023-09-03%2016.48.34.png)
	- cross validation 1 回目のみdumpする処理を追加
![スクリーンショット 2023-09-03 16.48.41.png](../_resources/スクリーンショット%202023-09-03%2016.48.41.png)
	- mfcc → audio_featへ変更
![スクリーンショット 2023-09-03 16.48.53.png](../_resources/スクリーンショット%202023-09-03%2016.48.53.png)
	- decoder を使わないように変更
![スクリーンショット 2023-09-03 16.49.00.png](../_resources/スクリーンショット%202023-09-03%2016.49.00.png)
	- mel spectrogram のmse評価結果を保存するファイル名指定
![スクリーンショット 2023-09-03 16.49.06.png](../_resources/スクリーンショット%202023-09-03%2016.49.06.png)
	- mel spectrogram を mse で評価するコードを追加
![スクリーンショット 2023-09-03 16.49.09.png](../_resources/スクリーンショット%202023-09-03%2016.49.09.png)
![スクリーンショット 2023-09-03 16.49.12.png](../_resources/スクリーンショット%202023-09-03%2016.49.12.png)


[↑ top](#top)


## make_sample_data
- （ECoG2TextとECoG2Speechの差分）
	- import 内容の追加、ファイルパスの変数追加
![スクリーンショット 2023-09-03 17.46.58.png](../_resources/スクリーンショット%202023-09-03%2017.46.58.png)
	- shell script に記述する内容の変更、parallel wavegan の pretrained model をダウンロードする処理の追加
![スクリーンショット 2023-09-03 17.47.14.png](../_resources/スクリーンショット%202023-09-03%2017.47.14.png)


[↑ top](#top)

