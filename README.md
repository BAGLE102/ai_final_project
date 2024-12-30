# AI Final Project
# 使用 TensorFlow 進行時間序列預測

本教程是使用 TensorFlow 進行時間序列預測的簡介。它構建了幾種不同樣式的模型，包括卷積神經網路（CNN）和循環神經網路（RNN）。

## 本教程包括兩個主要部分，每個部分包含若干小節：

### 預測單個時間步驟：

####   •**單個特徵**

####    •**所有特徵**

### 預測多個時間步驟：

#### •單次：一次做出所有預測

#### •自回歸：一次做出一個預測，並將輸出饋送回模型

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
## 檢查和清理
 接下來，看一下數據集的統計數據：
```py
df.describe().transpose()
```
#### 風速
值得注意的一件事是風速 （） 的 值和最大值 （） 列。 這個可能是錯誤的。wv (m/s)minmax. wv (m/s)-9999

有一個單獨的風向列，因此速度應大於零 （）。 將其替換為零：>=0
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
## 特徵工程
在潛心構建模型之前，務必瞭解數據並確保傳遞格式正確的數據。

風數據的最後一列 以度為單位給出了風向。 角度不是很好的模型輸入：360° 和 0° 應該會彼此接近，並平滑換行。 如果不吹風，方向則無關緊要。wd (deg)

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
同樣， 列非常有用，但不是以這種字串形式。 首先將其轉換為秒：Date Time
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

如果您沒有該資訊，則可以通過使用快速傅里葉變換提取特徵來確定哪些頻率重要。 要檢驗假設，下面是溫度隨時間變化的 tf.signal.rfft。 請注意 和 附近頻率的明顯峰值：1/year1/day

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
