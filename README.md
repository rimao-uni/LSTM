# LSTM

## Description
※研究室の勉強の一環として実装した。  
LSTM（Long Short-Term Memory）は、回帰結合型ニューラルネットワーク（RNN）の一種であり、時系列データを扱うためのモデルである。LSTMは、系列データ内の長期的な依存関係を学習し、勾配消失問題を軽減することができる。  
LSTMセルは、以下の主要な部分から構成される。  

・入力ゲート（Input Gate）: 新しい情報をどれだけセルの状態に追加するかを制御  
・忘却ゲート（Forget Gate）: セルの状態からどれだけ情報を忘れるかを制御  
・出力ゲート（Output Gate）: 隠れ状態をどれだけ次の時間ステップの入力に反映させるかを制御  

以下にLSTMセルの図と更新式を示す。  
<p align="center">
<img src="https://github.com/rimao-uni/LSTM/assets/117995370/4f666e9a-8f5c-473f-9979-34bbf557b0b9" >
</p>

$$i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})$$

$$f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})$$

$$g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})$$

$$o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})$$

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$

$$h_t = o_t \odot \tanh(c_t)$$

従来のRNNでは、時間ステップごとに隠れ状態が更新され、情報が時間を経て消失することが問題であったが、LSTMでは、セルの状態というメモリセルを導入することで、長期的な情報を保持することを可能にした。  

## Requirement
```
torch==1.9.0
pandas==1.3.3
matplotlib==3.4.3
```

## References
Christopher Olah氏のブログ記事 : [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
LONG SHORT-TERM MEMORY : [Paper](https://blog.xpgreat.com/file/lstm.pdf)
