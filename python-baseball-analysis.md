
# Python プロ野球データ分析

Atsutya Kobayashi 2018-08-12

---

### index

- 1. [`Numpy/Pandas`でのデータ構造など](#sec1)


- 2. [`Matplotlib`データの可視化：変数間の関係性の俯瞰](#sec2)


- 3. [`Numpy`での単回帰分析](#sec3)


- 4. [`scikit-learn`での多変量解析：変数選択，モデル選択](#sec4)


- 5. [GLM(一般化線形モデル)，正則化，Ridge/Lasso/erastic netによる勝率予想](#sec5)


- 6. [SVM(サポートベクターマシン)による分類(SVC)，交差検証，評価，可視化](#sec6)


- 7. [ランダムフォレストによる分類，可視化](#sec7)


- 8. [ニューラルネットワーク (パーセプトロン，MLP，DL)](#sec8)


- 9. [`TensorFlow`による深層学習モデル構築 ](#sec9)

---

#### References/Documentations

- [Numpy/Scipy](https://docs.scipy.org/doc/)
- [Pandas](https://pandas.pydata.org/)
- [matplotlib.pyplot](https://matplotlib.org/api/pyplot_summary.html)
- [scikit-learn](http://scikit-learn.org/stable/index.html)
- [tensorflow](https://www.tensorflow.org/)

---

#### About Data

`getNPB.py` : NPBの公式サイトのデータ (http://npb.jp/bis/yearly/) からシーズンプロ野球データを取得するプログラム(3系)．`BeautifulSoup4`と`requests`を用いて作成した。引数に年を入れるとその年のチーム毎の結果を返す．引数に何も指定しなかった場合は，1950年から2017年までのデータを全て取得し，`central_league.txt`と`pacific_league.txt`を出力する。

---

<span id="sec1"></span>  

## 1. Numpy/Pandasでのデータ確認

`.csv`データを読み込み，ベクトルデータとして扱う。


```python
%ls
```

    central_league.csv              pacificleague.csv
    centralleague_withsave.csv      pacificleague_withsave.csv
    getNPB.py                       python-baseball-analysis.ipynb
    pacific_league.csv              [34mvar[m[m/


データのCSVが用意されているので，`Pandas`で読み込んで見る。


```python
# typical
import numpy as np
import pandas as pd

# module for progress-bar
from tqdm import tqdm_notebook

# ignore insignificant warnings
import warnings
warnings.filterwarnings('ignore')
```

モジュールをインポート。


```python
# set visible max columns = 50
pd.set_option('display.max_columns', 50)
```

列の表示が途中で省略されてしまうので，最大表示列数を50まで上げておく。


```python
# read csv data as Pandas.DataFrame
data_pacific = pd.read_csv("./pacific_league.csv")
data_central = pd.read_csv("./central_league.csv")

print("pacific league data ,size is {}".format(data_pacific.shape))
print("central league data ,size is {}".format(data_central.shape))

"data imported!"
```

    pacific league data ,size is (831, 22)
    central league data ,size is (412, 22)





    'data imported!'



上記がセ／パ両リーグの，**1950年~2017年**のシーズンデータ。

---

### 各変数の説明

`pandas.DataFrame.columns`で列のラベル，データ・タイプを取得可能。


```python
# get columns data(tuple)
# for i in data_central.columns: 
#   print(i)

data_central.columns.values
```




    array(['Year', 'League', 'Rank', 'Team', 'All-Games', 'Win', 'Lose',
           'Draw', 'Win-Prob', 'Batting-Average', 'Bats', 'Points', 'Hits',
           'Double', 'Triple', 'HR', 'RBI', 'Steal', 'Protection-Ratio',
           'Whole-Pitch', 'Strike-Outs', 'Lost-Points'], dtype=object)



データの内容は，  

- Year : シーズン年度
- League : セ／パ
- Rank : シーズン順位
- Team : チーム名
- All-Games : 試合数
- Win : 勝利数
- Lose : 敗北数
- Draw : 引き分け数
- Win-Prob : 最終勝率
- Batting-Average : チーム打率
- Bats : チーム打数
- Points : チーム得点数
- Hits : チーム安打数
- Double : チーム二塁打数
- Triple : チーム三塁打数
- HR : チームホームラン数
- RBI : チーム打点数
- Steal : チーム盗塁数
- Protection-Ratio : チーム防御率
- Whole-Pitch : チーム完投数
- Strike-Outs : チーム脱三振数
- Lost-Points : チーム失点数


---

### セ・パ両データを結合


セ・リーグとパ・リーグのデータを結合し，チーム一覧も表示する。


```python
# integrate both leagues data
data_all = data_central.append(data_pacific).sort_values(["Year","Rank"]).drop_duplicates()


print(data_all.drop_duplicates().shape)

data_all["Team"].unique()
```

    (831, 22)





    array(['松竹ロビンス', '毎日オリオンズ', '中日ドラゴンズ', '南海ホークス', '読売ジャイアンツ', '大映スターズ',
           '大阪タイガース', '阪急ブレーブス', '大洋ホエールズ', '西鉄クリッパース', '西日本パイレーツ',
           '東急フライヤーズ', '国鉄スワローズ', '近鉄パールス', '広島カープ', '名古屋ドラゴンズ', '西鉄ライオンズ',
           '大洋松竹ロビンス', '高橋ユニオンズ', '東映フライヤーズ', 'トンボユニオンズ', '大映ユニオンズ',
           '毎日大映オリオンズ', '近鉄バファロー', '阪神タイガース', '近鉄バファローズ', '東京オリオンズ',
           'サンケイスワローズ', 'サンケイアトムズ', '広島東洋カープ', 'ロッテ・オリオンズ', 'アトムズ',
           'ヤクルトアトムズ', '太平洋クラブ・ライオンズ', '日拓ホーム・フライヤーズ', 'ヤクルトスワローズ',
           '日本ハム・ファイターズ', 'クラウンライター・ライオンズ', '横浜大洋ホエールズ', '西武ライオンズ',
           'オリックス・ブレーブス', '福岡ダイエーホークス', 'オリックス・ブルーウェーブ', '千葉ロッテマリーンズ',
           '横浜ベイスターズ', '大阪近鉄バファローズ', '北海道日本ハムファイターズ', '福岡ソフトバンクホークス',
           'オリックス・バファローズ', '東北楽天ゴールデンイーグルス', '東京ヤクルトスワローズ', '埼玉西武ライオンズ',
           '横浜DeNAベイスターズ'], dtype=object)



---

### 各データについてソート

- `.head(n)` で上からn項目を取得


- `.tail(n)` で下からn項目を取得


- `pandas.DataFrame.sort_values`[(ドキュメント)](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html) を用いて，ソート。



```python
data_all.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>League</th>
      <th>Rank</th>
      <th>Team</th>
      <th>All-Games</th>
      <th>Win</th>
      <th>Lose</th>
      <th>Draw</th>
      <th>Win-Prob</th>
      <th>Batting-Average</th>
      <th>Bats</th>
      <th>Points</th>
      <th>Hits</th>
      <th>Double</th>
      <th>Triple</th>
      <th>HR</th>
      <th>RBI</th>
      <th>Steal</th>
      <th>Protection-Ratio</th>
      <th>Whole-Pitch</th>
      <th>Strike-Outs</th>
      <th>Lost-Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1950</td>
      <td>central</td>
      <td>1</td>
      <td>松竹ロビンス</td>
      <td>137</td>
      <td>98</td>
      <td>35</td>
      <td>4</td>
      <td>0.737</td>
      <td>0.287</td>
      <td>4939</td>
      <td>908</td>
      <td>1417</td>
      <td>179</td>
      <td>49</td>
      <td>179</td>
      <td>825</td>
      <td>223</td>
      <td>3.23</td>
      <td>71</td>
      <td>438</td>
      <td>524</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1950</td>
      <td>pacific</td>
      <td>1</td>
      <td>毎日オリオンズ</td>
      <td>120</td>
      <td>81</td>
      <td>34</td>
      <td>5</td>
      <td>0.704</td>
      <td>0.286</td>
      <td>4245</td>
      <td>713</td>
      <td>1212</td>
      <td>209</td>
      <td>43</td>
      <td>124</td>
      <td>640</td>
      <td>195</td>
      <td>3.42</td>
      <td>58</td>
      <td>462</td>
      <td>512</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1950</td>
      <td>central</td>
      <td>2</td>
      <td>中日ドラゴンズ</td>
      <td>137</td>
      <td>89</td>
      <td>44</td>
      <td>4</td>
      <td>0.669</td>
      <td>0.274</td>
      <td>4787</td>
      <td>745</td>
      <td>1311</td>
      <td>229</td>
      <td>54</td>
      <td>144</td>
      <td>693</td>
      <td>179</td>
      <td>3.73</td>
      <td>72</td>
      <td>558</td>
      <td>597</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_all.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>League</th>
      <th>Rank</th>
      <th>Team</th>
      <th>All-Games</th>
      <th>Win</th>
      <th>Lose</th>
      <th>Draw</th>
      <th>Win-Prob</th>
      <th>Batting-Average</th>
      <th>Bats</th>
      <th>Points</th>
      <th>Hits</th>
      <th>Double</th>
      <th>Triple</th>
      <th>HR</th>
      <th>RBI</th>
      <th>Steal</th>
      <th>Protection-Ratio</th>
      <th>Whole-Pitch</th>
      <th>Strike-Outs</th>
      <th>Lost-Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>826</th>
      <td>2017</td>
      <td>pacific</td>
      <td>5</td>
      <td>北海道日本ハムファイターズ</td>
      <td>143</td>
      <td>60</td>
      <td>83</td>
      <td>0</td>
      <td>0.420</td>
      <td>0.242</td>
      <td>4749</td>
      <td>509</td>
      <td>1147</td>
      <td>189</td>
      <td>18</td>
      <td>108</td>
      <td>482</td>
      <td>86</td>
      <td>3.82</td>
      <td>6</td>
      <td>968</td>
      <td>596</td>
    </tr>
    <tr>
      <th>408</th>
      <td>2017</td>
      <td>central</td>
      <td>6</td>
      <td>東京ヤクルトスワローズ</td>
      <td>143</td>
      <td>45</td>
      <td>96</td>
      <td>2</td>
      <td>0.319</td>
      <td>0.234</td>
      <td>4728</td>
      <td>473</td>
      <td>1108</td>
      <td>166</td>
      <td>19</td>
      <td>95</td>
      <td>449</td>
      <td>50</td>
      <td>4.21</td>
      <td>6</td>
      <td>1011</td>
      <td>653</td>
    </tr>
    <tr>
      <th>827</th>
      <td>2017</td>
      <td>pacific</td>
      <td>6</td>
      <td>千葉ロッテマリーンズ</td>
      <td>143</td>
      <td>54</td>
      <td>87</td>
      <td>2</td>
      <td>0.383</td>
      <td>0.233</td>
      <td>4718</td>
      <td>479</td>
      <td>1098</td>
      <td>215</td>
      <td>29</td>
      <td>95</td>
      <td>455</td>
      <td>78</td>
      <td>4.22</td>
      <td>11</td>
      <td>939</td>
      <td>647</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("勝率TOP3")
display(data_all.sort_values(
    ["Win-Prob"], ascending=False, kind="quicksort").head(3))

print("打率TOP3")
display(data_all.sort_values(
    ["Batting-Average"], ascending=False, kind="quicksort").head(3))

print("防御率TOP3")
display(data_all.sort_values(
    ["Protection-Ratio"], ascending=True, kind="quicksort").head(3))
```

    勝率TOP3



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>League</th>
      <th>Rank</th>
      <th>Team</th>
      <th>All-Games</th>
      <th>Win</th>
      <th>Lose</th>
      <th>Draw</th>
      <th>Win-Prob</th>
      <th>Batting-Average</th>
      <th>Bats</th>
      <th>Points</th>
      <th>Hits</th>
      <th>Double</th>
      <th>Triple</th>
      <th>HR</th>
      <th>RBI</th>
      <th>Steal</th>
      <th>Protection-Ratio</th>
      <th>Whole-Pitch</th>
      <th>Strike-Outs</th>
      <th>Lost-Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>419</th>
      <td>1951</td>
      <td>pacific</td>
      <td>1</td>
      <td>南海ホークス</td>
      <td>104</td>
      <td>72</td>
      <td>24</td>
      <td>8</td>
      <td>0.750</td>
      <td>0.276</td>
      <td>3660</td>
      <td>496</td>
      <td>1010</td>
      <td>152</td>
      <td>45</td>
      <td>48</td>
      <td>441</td>
      <td>191</td>
      <td>2.40</td>
      <td>36</td>
      <td>376</td>
      <td>322</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1950</td>
      <td>central</td>
      <td>1</td>
      <td>松竹ロビンス</td>
      <td>137</td>
      <td>98</td>
      <td>35</td>
      <td>4</td>
      <td>0.737</td>
      <td>0.287</td>
      <td>4939</td>
      <td>908</td>
      <td>1417</td>
      <td>179</td>
      <td>49</td>
      <td>179</td>
      <td>825</td>
      <td>223</td>
      <td>3.23</td>
      <td>71</td>
      <td>438</td>
      <td>524</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1951</td>
      <td>central</td>
      <td>1</td>
      <td>読売ジャイアンツ</td>
      <td>114</td>
      <td>79</td>
      <td>29</td>
      <td>6</td>
      <td>0.731</td>
      <td>0.291</td>
      <td>3950</td>
      <td>702</td>
      <td>1151</td>
      <td>191</td>
      <td>34</td>
      <td>92</td>
      <td>632</td>
      <td>192</td>
      <td>2.62</td>
      <td>63</td>
      <td>436</td>
      <td>381</td>
    </tr>
  </tbody>
</table>
</div>


    打率TOP3



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>League</th>
      <th>Rank</th>
      <th>Team</th>
      <th>All-Games</th>
      <th>Win</th>
      <th>Lose</th>
      <th>Draw</th>
      <th>Win-Prob</th>
      <th>Batting-Average</th>
      <th>Bats</th>
      <th>Points</th>
      <th>Hits</th>
      <th>Double</th>
      <th>Triple</th>
      <th>HR</th>
      <th>RBI</th>
      <th>Steal</th>
      <th>Protection-Ratio</th>
      <th>Whole-Pitch</th>
      <th>Strike-Outs</th>
      <th>Lost-Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>745</th>
      <td>2003</td>
      <td>pacific</td>
      <td>1</td>
      <td>福岡ダイエーホークス</td>
      <td>140</td>
      <td>82</td>
      <td>55</td>
      <td>3</td>
      <td>0.599</td>
      <td>0.297</td>
      <td>4926</td>
      <td>822</td>
      <td>1461</td>
      <td>276</td>
      <td>33</td>
      <td>154</td>
      <td>794</td>
      <td>147</td>
      <td>3.94</td>
      <td>26</td>
      <td>1126</td>
      <td>588</td>
    </tr>
    <tr>
      <th>301</th>
      <td>1999</td>
      <td>central</td>
      <td>3</td>
      <td>横浜ベイスターズ</td>
      <td>135</td>
      <td>71</td>
      <td>64</td>
      <td>0</td>
      <td>0.526</td>
      <td>0.294</td>
      <td>4788</td>
      <td>711</td>
      <td>1408</td>
      <td>246</td>
      <td>20</td>
      <td>140</td>
      <td>688</td>
      <td>74</td>
      <td>4.44</td>
      <td>15</td>
      <td>868</td>
      <td>639</td>
    </tr>
    <tr>
      <th>751</th>
      <td>2004</td>
      <td>pacific</td>
      <td>2</td>
      <td>福岡ダイエーホークス</td>
      <td>133</td>
      <td>77</td>
      <td>52</td>
      <td>4</td>
      <td>0.597</td>
      <td>0.292</td>
      <td>4654</td>
      <td>739</td>
      <td>1359</td>
      <td>244</td>
      <td>28</td>
      <td>183</td>
      <td>706</td>
      <td>84</td>
      <td>4.58</td>
      <td>19</td>
      <td>923</td>
      <td>651</td>
    </tr>
  </tbody>
</table>
</div>


    防御率TOP3



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>League</th>
      <th>Rank</th>
      <th>Team</th>
      <th>All-Games</th>
      <th>Win</th>
      <th>Lose</th>
      <th>Draw</th>
      <th>Win-Prob</th>
      <th>Batting-Average</th>
      <th>Bats</th>
      <th>Points</th>
      <th>Hits</th>
      <th>Double</th>
      <th>Triple</th>
      <th>HR</th>
      <th>RBI</th>
      <th>Steal</th>
      <th>Protection-Ratio</th>
      <th>Whole-Pitch</th>
      <th>Strike-Outs</th>
      <th>Lost-Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>1955</td>
      <td>central</td>
      <td>1</td>
      <td>読売ジャイアンツ</td>
      <td>130</td>
      <td>92</td>
      <td>37</td>
      <td>1</td>
      <td>0.713</td>
      <td>0.266</td>
      <td>4436</td>
      <td>579</td>
      <td>1179</td>
      <td>156</td>
      <td>38</td>
      <td>84</td>
      <td>540</td>
      <td>133</td>
      <td>1.75</td>
      <td>61</td>
      <td>635</td>
      <td>291</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1956</td>
      <td>central</td>
      <td>2</td>
      <td>大阪タイガース</td>
      <td>130</td>
      <td>79</td>
      <td>50</td>
      <td>1</td>
      <td>0.612</td>
      <td>0.224</td>
      <td>4240</td>
      <td>386</td>
      <td>950</td>
      <td>149</td>
      <td>44</td>
      <td>54</td>
      <td>341</td>
      <td>165</td>
      <td>1.77</td>
      <td>31</td>
      <td>665</td>
      <td>283</td>
    </tr>
    <tr>
      <th>460</th>
      <td>1956</td>
      <td>pacific</td>
      <td>1</td>
      <td>西鉄ライオンズ</td>
      <td>154</td>
      <td>96</td>
      <td>51</td>
      <td>7</td>
      <td>0.646</td>
      <td>0.254</td>
      <td>5075</td>
      <td>611</td>
      <td>1288</td>
      <td>213</td>
      <td>56</td>
      <td>95</td>
      <td>567</td>
      <td>165</td>
      <td>1.87</td>
      <td>42</td>
      <td>902</td>
      <td>372</td>
    </tr>
  </tbody>
</table>
</div>


なんかセリーグのcsvデータが重複して2こずつになってたので`pandas.DataFrame.drop_duplicates()`で削除。

`Single : 単打数`を追加し，`loc`[(document)](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html)で列名指定して順番を入れ替える。


```python
# add column "Singles" data, counting the num of single hits
data_all["Singles"] = data_all["Hits"] \
- (data_all["Double"] + data_all["Triple"] + data_all["HR"])

# check data mat size
print(data_all.shape)

# change columns order set "Singles" col between "Hits" and "Doubles"
data_all = data_all.loc[:,['Year','Team', 'League', 'Rank', 
                           'All-Games', 'Win', 'Lose',
                           'Draw', 'Win-Prob', 'Batting-Average', 
                           'Bats', 'Points', 'Hits', 'Singles',
                           'Double', 'Triple', 'HR', 'RBI', 'Steal', 
                           'Protection-Ratio','Whole-Pitch', 
                           'Strike-Outs', 'Lost-Points']]

# view top 5 rows
data_all.head(5)
```

    (831, 23)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Team</th>
      <th>League</th>
      <th>Rank</th>
      <th>All-Games</th>
      <th>Win</th>
      <th>Lose</th>
      <th>Draw</th>
      <th>Win-Prob</th>
      <th>Batting-Average</th>
      <th>Bats</th>
      <th>Points</th>
      <th>Hits</th>
      <th>Singles</th>
      <th>Double</th>
      <th>Triple</th>
      <th>HR</th>
      <th>RBI</th>
      <th>Steal</th>
      <th>Protection-Ratio</th>
      <th>Whole-Pitch</th>
      <th>Strike-Outs</th>
      <th>Lost-Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1950</td>
      <td>松竹ロビンス</td>
      <td>central</td>
      <td>1</td>
      <td>137</td>
      <td>98</td>
      <td>35</td>
      <td>4</td>
      <td>0.737</td>
      <td>0.287</td>
      <td>4939</td>
      <td>908</td>
      <td>1417</td>
      <td>1010</td>
      <td>179</td>
      <td>49</td>
      <td>179</td>
      <td>825</td>
      <td>223</td>
      <td>3.23</td>
      <td>71</td>
      <td>438</td>
      <td>524</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1950</td>
      <td>毎日オリオンズ</td>
      <td>pacific</td>
      <td>1</td>
      <td>120</td>
      <td>81</td>
      <td>34</td>
      <td>5</td>
      <td>0.704</td>
      <td>0.286</td>
      <td>4245</td>
      <td>713</td>
      <td>1212</td>
      <td>836</td>
      <td>209</td>
      <td>43</td>
      <td>124</td>
      <td>640</td>
      <td>195</td>
      <td>3.42</td>
      <td>58</td>
      <td>462</td>
      <td>512</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1950</td>
      <td>中日ドラゴンズ</td>
      <td>central</td>
      <td>2</td>
      <td>137</td>
      <td>89</td>
      <td>44</td>
      <td>4</td>
      <td>0.669</td>
      <td>0.274</td>
      <td>4787</td>
      <td>745</td>
      <td>1311</td>
      <td>884</td>
      <td>229</td>
      <td>54</td>
      <td>144</td>
      <td>693</td>
      <td>179</td>
      <td>3.73</td>
      <td>72</td>
      <td>558</td>
      <td>597</td>
    </tr>
    <tr>
      <th>412</th>
      <td>1950</td>
      <td>南海ホークス</td>
      <td>pacific</td>
      <td>2</td>
      <td>120</td>
      <td>66</td>
      <td>49</td>
      <td>5</td>
      <td>0.574</td>
      <td>0.279</td>
      <td>4232</td>
      <td>645</td>
      <td>1181</td>
      <td>839</td>
      <td>211</td>
      <td>43</td>
      <td>88</td>
      <td>583</td>
      <td>225</td>
      <td>3.38</td>
      <td>38</td>
      <td>469</td>
      <td>495</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1950</td>
      <td>読売ジャイアンツ</td>
      <td>central</td>
      <td>3</td>
      <td>140</td>
      <td>82</td>
      <td>54</td>
      <td>4</td>
      <td>0.603</td>
      <td>0.268</td>
      <td>4831</td>
      <td>724</td>
      <td>1297</td>
      <td>936</td>
      <td>208</td>
      <td>27</td>
      <td>126</td>
      <td>673</td>
      <td>212</td>
      <td>2.90</td>
      <td>89</td>
      <td>632</td>
      <td>522</td>
    </tr>
  </tbody>
</table>
</div>



これである程度は，pandas.dataframeをキレイにした気がする。

---

<span id="sec2"></span>  



## 2. Matplotlib.pyplot でのデータ可視化

Matplotlibのコマンドの意味を理解する最も簡単な方法は、対応する概念の名前を覚えることです。  
今回は以下の事項だけ覚えておきましょう：

- Figure：1枚の図全体 (**複数のプロットを持つことができる**)
- Axes: グリッドとデータ点を持つプロット (≠axis)
- Line: 直線プロット (曲がっているように見えますが、**細かく見ると直線プロットの集積です**)
- Scatter: 散布図プロット
- X/Y axis label: X軸/Y軸のラベル名
- Title: グラフタイトル
- Legend: 凡例 (各線・点の説明あるいは記述)

![](https://matplotlib.org/_images/anatomy1.png)

[ドキュメント](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)

いつもの。インポート


```python
# magic
%matplotlib inline
import matplotlib.pyplot as plt
```

---

#### IPAフォントの導入

まず，IPAフォントを用いて日本語化。　[ブラウザからのダウンロードページ](https://ipafont.ipa.go.jp/old/ipafont/download.html)


```python
import matplotlib
from matplotlib import rc
print(matplotlib.get_cachedir()) # 削除するためのキャッシュの場所を確認

# 使用するフォントはIPAのものに。
font = {'family':'IPAGothic'}
rc('font', **font)
```

    /Users/atsuya/.matplotlib


- on bash

```
$ cd
$ curl -o ipafont.zip http://ipafont.ipa.go.jp/old/ipafont/IPAfont00303.php
$ unzip ipafont.zip
$ mv IPAfont00303/*.ttf ~/.pyenv/versions/anaconda3-5.1.0/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf
$ rm .matplotlib/*.cache
```

これにより日本語が`matplotlib`で使える。はず

---

#### 備考/追記

http://ipafont.ipa.go.jp/old/ipafont/IPAfont00303.php のcurlではダウンロードできなかったので，サイトから`.zip`ファイルを落として該当ディレクトリに`.ttf`ファイルを入れた方がはやい。`~/.jupyter/custom/`を設定している人は，`~/.jupyter/custom/fonts`内に`.ttf`ファイルを入れておく。

#### 出力画像サイズを変更しておく

`matplotlib.pyplot`でのインライン出力イメージが小さいので，設定の変更をしておく。  
また，フォントのサイズも変更しておく。その他，スタイル設定や色設定など変えておく。


```python
plt.style.use('ggplot')
plt.rcParams['ytick.color'] = '111111'
plt.rcParams['xtick.color'] = '111111'
plt.rcParams['axes.labelcolor'] = '111111'
plt.rcParams['figure.figsize'] = (15.0, 10.0)
plt.rcParams['font.size'] = 15
```

---

### matplotlib.pyplot.plot 基本確認

[ロジスティック関数(標準シグモイド関数)](https://ja.wikipedia.org/wiki/%E3%82%B7%E3%82%B0%E3%83%A2%E3%82%A4%E3%83%89%E9%96%A2%E6%95%B0)をプロットしてみる。

$$y=\frac{1}{(1+e^{-x})}$$


```python
# Casual style
X = np.linspace(-10, 10, 100)  # from -10 to 10, 100 elements
Y = 1/(1+np.exp(-X))

plt.figure()  # init of figure object
plt.plot(X, Y)  # plot
plt.show()  # show
```


![png](output_32_0.png)


上は簡単な描画の方法なので，`figure`の上に`axis`=軸を乗せ，そこに描画するという本来の書き方(フォーマルな書き方と呼ぶ)で，インボリュートを書いてみる。

$$x = cos\theta + \theta sin\theta$$
$$y = sin\theta + \theta cos\theta$$
$$(-8\pi < \theta < 8\pi)$$


```python
# formal style plotting

d = np.linspace(0, 8*np.pi, 1000) # from 0 to 8π, 1000 elements
d2 = np.linspace(-8*np.pi, 0, 1000) # from -8π to 0, 1000 elements
X = np.cos(d) + d * np.sin(d)
Y = np.sin(d) - d * np.cos(d)
x = np.cos(d2) + d2 * np.sin(d2)
y = np.sin(d2) - d2 * np.cos(d2)

# make figure
fig = plt.figure()
print(type(fig))

# make axis objects
ax = fig.add_subplot(221)  # make 2x2 space=1 axis obj on figure obj
ax.plot(X, Y, alpha=.5, color="black")  # plot on axis obj

ax2 = fig.add_subplot(222)  # make 2x2 space=2 axis obj on figure obj
ax2.plot(Y, X, alpha=.5, color="red")  # plot on axis obj

ax3 = fig.add_subplot(223)  # make 2x2 space=3 axis obj on figure obj
ax3.plot(x, y, alpha=.5, color="green")  # plot on axis obj

ax4 = fig.add_subplot(224)  # make 2x2 space=4 axis obj on figure obj
ax4.plot(y, x, alpha=.5, color="blue")  # plot on axis obj

plt.show()  # show figure
```

    <class 'matplotlib.figure.Figure'>



![png](output_34_1.png)


---

### `pandas.DataFrame.plot()` を用いたデータフレームの可視化

`pandas`からダイレクトに利用できる`matplotlib`の機能。

> Default Palameters  
>
> *DataFrame.plot(x=None, y=None, kind='line', ax=None, subplots=False, sharex=None, sharey=False, layout=None, figsize=None, use_index=True, title=None, grid=None, legend=True, style=None, logx=False, logy=False, loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, fontsize=None, colormap=None, table=False, yerr=None, xerr=None, secondary_y=False, sort_columns=False, **kwds)*

---

#### 描画可能な表タイプ (kind=)

- `line` : line plot (default) = 折れ線グラフ
- `bar` : vertical bar plot = 棒グラフ
- `barh` : horizontal bar plot = 横棒グラフ
- `hist` : histogram = ヒストグラム
- `box` : boxplot = 箱ひげ図
- `kde` : Kernel Density Estimation plot = カーネル密度推定
- `density` : same as ‘kde’ = カーネル密度推定
- `area` : area plot = エリア:色を塗った折れ線のような
- `pie` : pie plot = 円グラフ
- `scatter` : scatter plot = 散布図
- `hexbin` : hexbin plot = ビン散布:密度を色で

`DataFrame`の1次元版みたいなヤツ，各一つのlabelを抜き出し`pandas.Series`型にして，
折れ線グラフを書いてみる。

---

#### Documents

- [`pyplot.plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)


- [`pandas.DataFrame.plot`](http://pandas.pydata.org/pandas-docs/stable//generated/pandas.DataFrame.plot.html)

---


```python
# make specific team Series datas
chunichi_dragons = data_all[(data_all["Team"] == "中日ドラゴンズ")].set_index("Year")
yomiuri_giants = data_all[(data_all["Team"] == "読売ジャイアンツ")].set_index("Year")
hanshin_tigers = data_all[(data_all["Team"] == "阪神タイガース")].set_index("Year")

chunichi_dragons["Batting-Average"].plot(color="blue")
yomiuri_giants["Batting-Average"].plot(color="orange")
hanshin_tigers["Batting-Average"].plot(color="yellow", grid=True, title="NPBチーム別シーズン打率")

plt.show()
```


![png](output_36_0.png)


`df[df["Label"]=="value"]`により，ラベルの値の真偽の行列に基づいてデータフレームの一部を抜き出せる。  
各チームのデータを抜き出した後，x軸に表示するために年`Year`をインデックスにし，`matplotlib.pyplot.plot()`でプロット。`show()`で表示。

でも，実は`DataFrame`でもそのまま簡単にグラフをかける。


```python
chunichi_dragons.loc[:,
                     ["Singles","Double","Triple","HR"]
                    ].plot(title="中日ドラゴンズ安打数推移",kind="line", grid=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1e274128>




![png](output_39_1.png)


---

### 列データにチーム毎の色のデータを追加

プロットの色を変えたいので，pandasのチーム名ごとに色を指定してみる。

`pandas.Series.replace()`を用いる。

> Series.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad', axis=None)[source]

やっぱり，　`pandas.Series.where`のほうがいいかも。

> *Series.where(cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False, raise_on_error=None)*

`where()`で，ブールのリストに基づいた値を`other`に入れ替える。`inplace`で上書きかどうかを指定。ブールのリストは，`Team`に各チーム名(ドラゴンズとか)が含まれているデータのみ`False`になっている。ここでもとデータの上書きやら何やらで小一時間ハマった。

[(参考)](https://deepage.net/features/pandas-where.html)


---

### matplotlib で指定できる色

- [ドキュメント](https://matplotlib.org/examples/color/named_colors.html)

![](https://matplotlib.org/_images/named_colors.png)

---


せっかくなのでfigを横並びにできるよう，([参考](https://matplotlib.org/gallery/subplots_axes_and_figures/subplot.html#sphx-glr-gallery-subplots-axes-and-figures-subplot-py))`matplotlib.pyplot.subplot`を使い，`matplotlib.pyplot.scatter` を用いて点描してみる。[(ドキュメント)](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html)

> *matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None, **kwargs)[source]*

---

あとついでに，`DataFrame[label]` でも，`DataDrame.label`でも行けるらしいので後者で。

---

### matplotlib.figure.Figure.add_subplot

フォーマルな方法で描画するときの，`add_subplot()`([ドキュメント](add_subplot()https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot))は，matplotlibのFigureクラスのメソッド。毎回の`fig = plt.figure()`は，コイツのインスタンス生成。

> *class matplotlib.figure.Figure(figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None, constrained_layout=None)*

生成した`figure`上の`matplotlib.axes.Axes`オブジェクトは，`matplotlib.axes.Axes.set_title`で様々設定できるらしい。[(ドキュメント)](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_title.html)

> *Axes.set_title(label, fontdict=None, loc='center', pad=None, **kwargs)*


```python
# casual protting
# data_all.plot(title="RBI / Win Probabiloity", x="Win-Prob", y="RBI",kind="scatter",grid=1)
# data_all.plot(title="Protection Ratio / Win Probabiloity" ,x="Win-Prob", y="Protection-Ratio",kind="scatter",grid=1)

# formal plotting
fig = plt.figure(edgecolor="k")

# add a column for colors
data_all.colors = ["black"] * data_all.Team.size # 1 x 831

def team2color(t, col):
    
    bool_list = ~data_all.Team.str.contains(t) # inverse T/F with tilde
    # renew
    data_all.color.where(cond=bool_list, other=col, inplace=True)
    print("set {} to {}".format(col, t))

# central league
team2color("ドラゴンズ","blue")
team2color("ジャイアンツ","darkorange")
team2color("タイガース","gold")
team2color("ヤクルト","navy")
team2color("ベイスターズ","deepskyblue")
team2color("広島","red")

# pacific league
team2color("ホークス","yellow")
team2color("ファイターズ","grey")
team2color("ライオンズ","aqua")
team2color("バファローズ","indianred")
team2color("マリーンズ","k")
team2color("阪","navajowhite")

# plotting on subplot objs

ax_l = fig.add_subplot(121) # add axis on 1st space of 1x2 figure
ax_l.scatter(x=data_all["RBI"], y=data_all["Win-Prob"],
              c=data_all.color, alpha=.5
             )
# by setting data_all.color to parameter "c", set team color on each plots

# set title on left axis obj (ax_l)
ax_l.set_title(label="RBI / Win Probability")

ax_r = fig.add_subplot(122) # add axis on 2nd space of 1x2 figure
ax_r.scatter(x=data_all["Protection-Ratio"], y=data_all["Win-Prob"], 
             c=data_all.color, alpha=.5
            )
# by setting data_all.color to parameter "c", set team color on each plots

# set title on right axis obj (ax_r)
ax_r.set_title(label="Protection Ratio / Win Probability")

plt.ylabel("Win Prob.")

fig.show()
```

    set blue to ドラゴンズ
    set darkorange to ジャイアンツ
    set gold to タイガース
    set navy to ヤクルト
    set deepskyblue to ベイスターズ
    set red to 広島
    set yellow to ホークス
    set grey to ファイターズ
    set aqua to ライオンズ
    set indianred to バファローズ
    set k to マリーンズ
    set navajowhite to 阪



![png](output_41_1.png)



```python
data_all[(data_all["Rank"] == 1)]["Team"].value_counts()\
.plot(kind="pie", figsize=(12,12), title="チーム別リーグ優勝回数")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2583bfd0>




![png](output_42_1.png)


---

<span id="sec3"></span>  



## 3. Numpyでの単回帰 Linear Regression
