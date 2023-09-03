# ECoG2Audio

## デモ
1. docker image を作成
  `user ~ % docker build -t ${tag} . # タグをつけてimage作成(例:ecog2audio)`
2. コンテナを実行
  `user ~ % docker run -v ${pwd}:/workspace --rm -it ${tag}`
3. (コンテナ上)ワークディレクトリへ移動
  `container:/# cd workspace`
4. (コンテナ上)サンプルデータを作成
  `container:/workspace# python3 make_sample_data.py`
5. (コンテナ上)サンプルデータを使ってモデルの学習〜評価〜音声再合成
  `container:/workspave# sh sample_run_audio.sh`
  （再合成音声の音量レベルは調整する必要あり）

## 主なファイルの説明

### ./Dockerfile
dockerのイメージ作成用の設定ファイル。

### ./make_sample_data.py
サンプルデータ作成用のコード。
### ./conf/
モデルのパラメータを設定するコード。（デバッグのため epoch=5 に設定。）
<dl>
  <!-- <dt>config_xA000.ini</dt> -->
  <!-- <dd>ECoG2Text実行時</dd> -->
  <dt>config_xA001.ini</dt>
  <dd>BLSTMエンコーダを使用する場合</dd>
  <dt>config_xA002.ini</dt>
  <dd>Transformerエンコーダを使用する場合</dd>
</dl>

### ./ecog2text/
ECoGの前処理をするコード。
<dl>
  <dt>preprocess.py</dt>
  <dd>ecog2textからecog2audio用に書き換えた、前処理用のモジュール</dd>
</dl>

### ./ecog2audio_XXX
モデルの学習〜評価をするコード。ecog2textからecog2audio用に書き換えたもの。
<dl>
  <dt>ecog2audio_train.py</dt>
  <dd>学習用</dd>
  <dt>ecog2audio_eval.py</dt>
  <dd>評価用</dd>
</dl>

### ./ecog2text_XXX
モデルの学習〜評価をするコード。ecog2text用。
<dl>
  <dt>ecog2text_train.py</dt>
  <dd>学習用</dd>
  <dt>ecog2audio_eval.py</dt>
  <dd>評価用</dd>
</dl>

### ./audio_reconstruction.py
音声再合成用のコード。


