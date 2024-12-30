# AI Final Project
# 使用 TensorFlow 進行時間序列預測

本教程是使用 TensorFlow 進行時間序列預測的簡介。它構建了幾種不同樣式的模型，包括卷積神經網路（CNN）和循環神經網路（RNN）。

## 本教程包括兩個主要部分，每個部分包含若干小節：

### 預測單個時間步驟：

- **單個特徵**

- **所有特徵**

### 預測多個時間步驟：

- **單次**：一次做出所有預測

- **自回歸**：一次做出一個預測，並將輸出饋送回模型

## *安裝*
```py
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
```
![圖片](pic/001.png)

## 天氣數據集

本教程使用由[馬克斯·普朗克生物地球化學研究所](https://www.bgc-jena.mpg.de/wetter/)記錄的[天氣時間序列數據集](https://tensorflow.google.cn/tutorials/structured_data/time_series?hl=zh_cn#%E5%A4%A9%E6%B0%94%E6%95%B0%E6%8D%AE%E9%9B%86)。

此數據集包含了 14 個不同特徵，例如氣溫、氣壓和濕度。 自 2003 年起，這些數據每 10 分鐘就會被收集一次。為了提高效率，您將僅使用 2009 至 2016 年之間收集的數據。數據集的這一部分由 François Chollet 為他的[《Deep Learning with Python》](https://www.manning.com/books/deep-learning-with-python)一書所準備。
```py
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
```

本教程僅處理**每小時預測**，因此先從 10 分鐘間隔到 1 小時對數據進行下採樣：
```py
df = pd.read_csv(csv_path)
# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
```
讓我們看一下數據。 下面是前幾行：
```py
df.head()
```
下面是一些特徵隨時間的演變：
```py
plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)
```
### 檢查和清理
 接下來，看一下數據集的統計數據：
```py
df.describe().transpose()
```
### 風速
值得注意的一件事是風速 （） 的 值和最大值 （） 列。 這個可能是錯誤的。``` wv (m/s)minmax. wv (m/s)-9999 ```

有一個單獨的風向列，因此速度應大於零 （）。 將其替換為零：```>=0```
```py
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame.
df['wv (m/s)'].min()
```
### 特徵工程
在潛心構建模型之前，務必瞭解數據並確保傳遞格式正確的數據。

風數據的最後一列 以度為單位給出了風向。 角度不是很好的模型輸入：$360°$ 和 $ 0°$ 應該會彼此接近，並平滑換行。 如果不吹風，方向則無關緊要。```wd (deg)```

現在，風數據的分佈狀況如下：

```py
plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
```

但是，如果將風向和風速列轉換為風向量，模型將更容易解釋：

```py
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)
```
模型正確解釋風向量的分佈要簡單得多：
```py
plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
ax = plt.gca()
ax.axis('tight')
```
### 時間
同樣， 列非常有用，但不是以這種字串形式。 首先將其轉換為秒：```Date Time```
```py
timestamp_s = date_time.map(pd.Timestamp.timestamp)
```
與風向類似，以秒為單位的時間不是有用的模型輸入。 作為天氣數據，它有清晰的每日和每年週期性。 可以通過多種方式處理週期性。

您可以通過使用正弦和餘弦變換為清晰的「一天中的時間」和「一年中的時間」信號來獲得可用的信號：
```py
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
```
```py
plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
```
這使模型能夠訪問最重要的頻率特徵。 在這種情況下，您提前知道了哪些頻率很重要。

如果您沒有該資訊，則可以通過使用快速傅里葉變換提取特徵來確定哪些頻率重要。 要檢驗假設，下面是溫度隨時間變化的 tf.signal.rfft。 請注意 和 附近頻率的明顯峰值：```1/year1/day```

```py
fft = tf.signal.rfft(df['T (degC)'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['T (degC)'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
```

### 拆分數據
您將使用 拆分出訓練集、驗證集和測試集。 請注意，在拆分前數據**沒有**隨機打亂順序。 這有兩個原因：```(70%, 20%, 10%)```
1. 確保仍然可以將數據切入連續樣本的視窗

2. 確保訓練后在收集的數據上對模型進行評估，驗證/測試結果更加真實。
```py
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
```
### 歸一化數據
在訓練神經網路之前縮放特徵很重要。 歸一化是進行此類縮放的常見方式：減去平均值，然後除以每個特徵的標準偏差。

平均值和標準偏差應僅使用訓練數據進行計算，從而使模型無法訪問驗證集和測試集中的值。

有待商榷的是：模型在訓練時不應訪問訓練集中的未來值，以及應該使用移動平均數來進行此類規範化。 這不是本教程的重點，驗證集和測試集會確保我們獲得（某種程度上）可靠的指標。 因此，為了簡單起見，本教程使用的是簡單平均數。
```py
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
```
現在看一下這些特徵的分佈。 部分特徵的尾部確實很長，但沒有類似 風速值的明顯錯誤. ```-9999```
```py 
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
```
## 數據視窗化
本教程中的模型將基於來自數據連續樣本的窗口進行一組預測。

輸入視窗的主要特徵包括：
- . 輸入和標籤視窗的寬度（時間步驟數量）。
- 它們之間的時間偏移量。
- 用作輸入、標籤或兩者的特徵。
本教程構建了各種模型（包括線性、DNN、CNN 和 RNN 模型），並將它們用於以下兩種情況：
- 單輸出和多輸出預測。
- 單時間步驟和多時間步驟預測。
本部分重點介紹實現數據視窗化，以便將其重用到上述所有模型。

根據任務和模型類型，您可能需要生成各種資料視窗。 下面是一些範例：
1. 例如，要在給定 24 小時歷史記錄的情況下對未來 24 小時作出一次預測，可以定義如下視窗：
![alt text](image.png)
1. 給定 6 小時的歷史記錄，對未來 1 小時作出一次預測的模型將需要類似下面的視窗：
![alt text](image-1.png)

本部分的剩餘內容會定義 類。 此類可以：```WindowGenerator```
1. 處理如上圖所示的索引和偏移量。
2. 將特徵視窗拆分為 對。```(features, labels)```
3. 繪製結果視窗的內容。
4. 使用 [tf.data.Dataset](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn) 從訓練、評估和測試數據高效生成這些視窗的批次。

### 1. 索引和偏移量
首先創建 類。 方法包含輸入和標籤索引的所有必要邏輯。```WindowGenerator__init__```

它還將訓練、評估和測試 DataFrame 作為輸出。 這些稍後將被轉換為視窗的 [tf.data.Dataset。](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn)
```py 
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
```
下面是建立本部份開頭圖表中所示的兩個視窗的代碼：
```py
w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=['T (degC)'])
w1
```
```py
w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['T (degC)'])
w2
```
### 2. 拆分
給定一個連續輸入的清單， 方法會將它們轉換為輸入視窗和標籤視窗。```split_window```

您之前定義的範例 將按以下方式分割：```w2```

![alt text](image-2.png)

此圖不顯示數據的 軸，但此 函數還會處理 ，因此可以將其用於單輸出和多輸出樣本。```featuressplit_windowlabel_columns```
```py
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window
```
試試以下代碼：
```py
# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')
```
通常，TensorFlow 中的數據會被打包到陣列中，其中最外層索引是交叉樣本（“批次”維度）。 中間索引是“時間”和“空間”（寬度、高度）維度。 最內層索引是特徵。

上面的代碼使用了三個 7 時間步驟視窗的批次，每個時間步驟有 19 個特徵。 它將其拆分成一個 6 時間步驟的批次、19 個特徵輸入和一個 1 時間步驟 1 特徵的標籤。 該標籤僅有一個特徵，因為 已使用 進行了初始化。 最初，本教程將構建預測單個輸出標籤的模型。```WindowGeneratorlabel_columns=['T (degC)']```

### 3.繪圖
下面是一個繪圖方法，可以對拆分視窗進行簡單可視化：
```py
w2.example = example_inputs, example_labels
```
```py
def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot = plot
```
此繪圖根據專案引用的時間來對齊輸入、標籤和（稍後的）預測：
```py
w2.plot()
```
您可以繪製其他列，但是樣本視窗 配置僅包含 列的標籤。```w2T (degC)```
```py 
w2.plot(plot_col='p (mbar)')
```
### 4. 創建 [tf.data.Dataset](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn)

最後，此```make_dataset```方法將獲取時間序列 DataFrame 並使用 [tf.keras.utils.timeseries_dataset_from_array](https://tensorflow.google.cn/api_docs/python/tf/keras/utils/timeseries_dataset_from_array?hl=zh-cn) 函數將其轉換為```(input_window, label_window)```對的 [tf.data.Dataset](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn)。

```py
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset
```
`WindowGenerator `物件包含訓練、驗證和測試數據。

使用您之前定義的 `make_dataset` 方法添加属性以作為 [tf.data.Dataset](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn) 訪問他們。此外，添加一個標準樣本批次以便於訪問和繪圖：


## 單步模型

基於此類數據能夠構建的最簡單模型，能夠僅根據當前條件預測單個特徵的值，即未來的一個時間步驟（1 小時）。

因此，從構建模型開始，預測未來 1 小時的值。T (degC)

### 預測下一個時間步驟

設定物件以產生下列單步對：`WindowGenerator(input, label)`

```
Total window size: 2  
Input indices: [0]  
Label indices: [1]  
Label column name(s): ['T (degC)']  
```

window 會根據訓練、驗證和測試集創建，使您可以輕鬆反覆運算數據批次。`tf.data.Datasets`

Inputs shape (batch, time, features): (32, 1, 19)  
Labels shape (batch, time, features): (32, 1, 1)  

### 基線

在構建可訓練模型之前，最好將性能基線作為與以後更複雜的模型進行比較的點。

第一個任務是在給定所有特徵的當前值的情況下，預測未來 1 小時的溫度。當前值包括當前溫度。

因此，從僅返回當前溫度作為預測值的模型開始，預測“無變化”。這是一個合理的基線，因為溫度變化緩慢。當然，如果您對更遠的未來進行預測，此基線的效果就不那麼好了。

#### 將輸入傳送到輸出

#### 實例化並評估此模型：
```
439/439 [==============================] - 1s 2ms/step - loss: 0.0128 - mean_absolute_error: 0.0785
```

上面的代碼列印了一些性能指標，但這些指標並沒有使您對模型的運行情況有所瞭解。

`WindowGenerator` 有一種繪製方法，但只有一個樣本，繪圖不是很有趣。

因此，創建一個更寬的來一次生成包含 24 小時連續輸入和標籤的視窗。新的變數不會更改模型的運算方式。模型仍會根據單個輸入時間步驟對未來 1 小時進行預測。這裡 `window` 軸的作用類似於 `time` 軸：每個預測都是獨立進行的，時間步驟之間沒有交互：
``
Total window size: 25  
Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]  
Label indices: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]  
Label column name(s): ['T (degC)']
``
#### 對未來 1 小時進行一次預測，每小時一次

Input shape: (32, 24, 19)  
Output shape: (32, 24, 1)  

通過繪製基線模型的預測值，可以注意到只是標籤向右移動了 1 小時：



#### 線性模型

可以應用於此任務的最簡單的可訓練模型是在輸入和輸出之間插入線性轉換。在這種情況下，時間步驟的輸出僅取決於該步驟：

沒有設置 `activation` 的 `tf.keras.layers.Dense` 層是線性模型。層僅會將數據的最後一個軸從 `(batch, time, inputs)` 轉換為 `(batch, time, units)`；它會單獨應用於 `batch` 和 `time` 軸的每個條目。

#### 訓練過程函數

本教程訓練許多模型，因此將訓練過程打包到一個函數中：

#### 訓練模型並評估其性能

