# Diagnosis of Chest Conditions with CNN

### Problem Statement 

The diagnosis of chest conditions typically requires the use of medical imaging techniques like
computed tomography (CT) or X-ray scans. CT scans are favored over X-ray scans because of
their ability to produce high-fidelity 3D images with multiple X-ray scans from various angles.
However, cost differences between the two techniques are significant. CT scans are approximately 10x the cost of X-ray scans.  This project seeks solutions to accurately diagnose chest conditions based on lower-cost X-rays.  

The National Institute of Health's (NIH) chest X-rays dataset will be used, which consists of more than 100k labeled chest X-ray images.  Deep learning techniques like traditional (densely connected) and convolutional neural networks will be explored. 


## Preliminaries


```python
import numpy as np
import pandas as pd

import os
import glob
import random
import itertools
import math

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb

import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import scipy
from scipy.interpolate import interp1d

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
```

    3.1.1
    

## Exploratory Data Analysis + Data Cleaning


```python
# Read in data labels
raw_data = pd.read_csv('data/Data_Entry_2017.csv')
data_labels = raw_data.copy()
```


```python
data_labels.head()
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
      <th>Image Index</th>
      <th>Finding Labels</th>
      <th>Follow-up #</th>
      <th>Patient ID</th>
      <th>Patient Age</th>
      <th>Patient Gender</th>
      <th>View Position</th>
      <th>OriginalImage[Width</th>
      <th>Height]</th>
      <th>OriginalImagePixelSpacing[x</th>
      <th>y]</th>
      <th>Unnamed: 11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>00000001_000.png</td>
      <td>Cardiomegaly</td>
      <td>0</td>
      <td>1</td>
      <td>58</td>
      <td>M</td>
      <td>PA</td>
      <td>2682</td>
      <td>2749</td>
      <td>0.143</td>
      <td>0.143</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>00000001_001.png</td>
      <td>Cardiomegaly|Emphysema</td>
      <td>1</td>
      <td>1</td>
      <td>58</td>
      <td>M</td>
      <td>PA</td>
      <td>2894</td>
      <td>2729</td>
      <td>0.143</td>
      <td>0.143</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>00000001_002.png</td>
      <td>Cardiomegaly|Effusion</td>
      <td>2</td>
      <td>1</td>
      <td>58</td>
      <td>M</td>
      <td>PA</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.168</td>
      <td>0.168</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>00000002_000.png</td>
      <td>No Finding</td>
      <td>0</td>
      <td>2</td>
      <td>81</td>
      <td>M</td>
      <td>PA</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.171</td>
      <td>0.171</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>00000003_000.png</td>
      <td>Hernia</td>
      <td>0</td>
      <td>3</td>
      <td>81</td>
      <td>F</td>
      <td>PA</td>
      <td>2582</td>
      <td>2991</td>
      <td>0.143</td>
      <td>0.143</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_labels.describe()

# Observe the presence of outliers in patient age.  Remove records with patient ages of over 120
data_labels = data_labels[data_labels['Patient Age'] <= 120]
```


```python
data_labels.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 112104 entries, 0 to 112119
    Data columns (total 12 columns):
    Image Index                    112104 non-null object
    Finding Labels                 112104 non-null object
    Follow-up #                    112104 non-null int64
    Patient ID                     112104 non-null int64
    Patient Age                    112104 non-null int64
    Patient Gender                 112104 non-null object
    View Position                  112104 non-null object
    OriginalImage[Width            112104 non-null int64
    Height]                        112104 non-null int64
    OriginalImagePixelSpacing[x    112104 non-null float64
    y]                             112104 non-null float64
    Unnamed: 11                    0 non-null float64
    dtypes: float64(3), int64(5), object(4)
    memory usage: 11.1+ MB
    


```python
# Determine list of all possible chest conditions
def get_conditions(df):
    conditions = df['Finding Labels'].map(lambda x: x.split('|')).tolist()
    conditions = list(itertools.chain(*conditions))
    conditions = np.unique(conditions)
    return(conditions)

all_conditions = get_conditions(data_labels)
all_conditions
```




    array(['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
           'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
           'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
           'Pneumothorax'], dtype='<U18')




```python
# Perform one-hot encoding for conditions
def oh_encode(df, conditions):
    for condition in conditions:
        df[condition] = df['Finding Labels'].map(lambda x: 1 if condition in x else 0)
    return None

oh_encode(data_labels, all_conditions)
```


```python
data_labels.head()
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
      <th>Image Index</th>
      <th>Finding Labels</th>
      <th>Follow-up #</th>
      <th>Patient ID</th>
      <th>Patient Age</th>
      <th>Patient Gender</th>
      <th>View Position</th>
      <th>OriginalImage[Width</th>
      <th>Height]</th>
      <th>OriginalImagePixelSpacing[x</th>
      <th>...</th>
      <th>Emphysema</th>
      <th>Fibrosis</th>
      <th>Hernia</th>
      <th>Infiltration</th>
      <th>Mass</th>
      <th>No Finding</th>
      <th>Nodule</th>
      <th>Pleural_Thickening</th>
      <th>Pneumonia</th>
      <th>Pneumothorax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>00000001_000.png</td>
      <td>Cardiomegaly</td>
      <td>0</td>
      <td>1</td>
      <td>58</td>
      <td>M</td>
      <td>PA</td>
      <td>2682</td>
      <td>2749</td>
      <td>0.143</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>00000001_001.png</td>
      <td>Cardiomegaly|Emphysema</td>
      <td>1</td>
      <td>1</td>
      <td>58</td>
      <td>M</td>
      <td>PA</td>
      <td>2894</td>
      <td>2729</td>
      <td>0.143</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>00000001_002.png</td>
      <td>Cardiomegaly|Effusion</td>
      <td>2</td>
      <td>1</td>
      <td>58</td>
      <td>M</td>
      <td>PA</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.168</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>00000002_000.png</td>
      <td>No Finding</td>
      <td>0</td>
      <td>2</td>
      <td>81</td>
      <td>M</td>
      <td>PA</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.171</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>00000003_000.png</td>
      <td>Hernia</td>
      <td>0</td>
      <td>3</td>
      <td>81</td>
      <td>F</td>
      <td>PA</td>
      <td>2582</td>
      <td>2991</td>
      <td>0.143</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
# Evaluate distribution of conditions.  Observe a highly imbalanced dataset.  Remove records with 'No Finding' label
data_labels = data_labels[data_labels['No Finding']==0]

def show_dist(df, colname, n):
    counts = df[colname].value_counts()[:n]
    
    x_val = range(n)
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    ax.bar(x_val, counts)
    ax.set_xticks(x_val)
    ax.set_xticklabels(counts.index, rotation=90)
    plt.title('Distribution of Chest Conditions')
    plt.ylabel('Number of Records')
    return None

show_dist(data_labels, 'Finding Labels', 10)
```


![png](output_13_0.png)



```python
# Drop 'No Finding' column--no longer relevant
data_labels.drop(labels=['No Finding'], axis=1, inplace=True)
data_labels.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 51751 entries, 0 to 112115
    Data columns (total 26 columns):
    Image Index                    51751 non-null object
    Finding Labels                 51751 non-null object
    Follow-up #                    51751 non-null int64
    Patient ID                     51751 non-null int64
    Patient Age                    51751 non-null int64
    Patient Gender                 51751 non-null object
    View Position                  51751 non-null object
    OriginalImage[Width            51751 non-null int64
    Height]                        51751 non-null int64
    OriginalImagePixelSpacing[x    51751 non-null float64
    y]                             51751 non-null float64
    Unnamed: 11                    0 non-null float64
    Atelectasis                    51751 non-null int64
    Cardiomegaly                   51751 non-null int64
    Consolidation                  51751 non-null int64
    Edema                          51751 non-null int64
    Effusion                       51751 non-null int64
    Emphysema                      51751 non-null int64
    Fibrosis                       51751 non-null int64
    Hernia                         51751 non-null int64
    Infiltration                   51751 non-null int64
    Mass                           51751 non-null int64
    Nodule                         51751 non-null int64
    Pleural_Thickening             51751 non-null int64
    Pneumonia                      51751 non-null int64
    Pneumothorax                   51751 non-null int64
    dtypes: float64(3), int64(19), object(4)
    memory usage: 10.7+ MB
    


```python
# Get list of all possible chest conditions
all_conditions = get_conditions(data_labels)
all_conditions
```




    array(['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
           'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
           'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
           'Pneumothorax'], dtype='<U18')




```python
# Generate co-occurrence matrix of conditions
cooccur_mat = data_labels[all_conditions].T.dot(data_labels[all_conditions])
cooccur_mat
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
      <th>Atelectasis</th>
      <th>Cardiomegaly</th>
      <th>Consolidation</th>
      <th>Edema</th>
      <th>Effusion</th>
      <th>Emphysema</th>
      <th>Fibrosis</th>
      <th>Hernia</th>
      <th>Infiltration</th>
      <th>Mass</th>
      <th>Nodule</th>
      <th>Pleural_Thickening</th>
      <th>Pneumonia</th>
      <th>Pneumothorax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Atelectasis</td>
      <td>11558</td>
      <td>370</td>
      <td>1223</td>
      <td>221</td>
      <td>3275</td>
      <td>424</td>
      <td>220</td>
      <td>40</td>
      <td>3264</td>
      <td>739</td>
      <td>590</td>
      <td>496</td>
      <td>262</td>
      <td>774</td>
    </tr>
    <tr>
      <td>Cardiomegaly</td>
      <td>370</td>
      <td>2776</td>
      <td>169</td>
      <td>127</td>
      <td>1063</td>
      <td>44</td>
      <td>52</td>
      <td>7</td>
      <td>587</td>
      <td>102</td>
      <td>108</td>
      <td>111</td>
      <td>41</td>
      <td>49</td>
    </tr>
    <tr>
      <td>Consolidation</td>
      <td>1223</td>
      <td>169</td>
      <td>4667</td>
      <td>162</td>
      <td>1287</td>
      <td>103</td>
      <td>79</td>
      <td>4</td>
      <td>1221</td>
      <td>610</td>
      <td>428</td>
      <td>251</td>
      <td>123</td>
      <td>223</td>
    </tr>
    <tr>
      <td>Edema</td>
      <td>221</td>
      <td>127</td>
      <td>162</td>
      <td>2302</td>
      <td>593</td>
      <td>30</td>
      <td>9</td>
      <td>3</td>
      <td>981</td>
      <td>129</td>
      <td>131</td>
      <td>64</td>
      <td>340</td>
      <td>33</td>
    </tr>
    <tr>
      <td>Effusion</td>
      <td>3275</td>
      <td>1063</td>
      <td>1287</td>
      <td>593</td>
      <td>13316</td>
      <td>359</td>
      <td>188</td>
      <td>21</td>
      <td>4000</td>
      <td>1254</td>
      <td>912</td>
      <td>849</td>
      <td>268</td>
      <td>996</td>
    </tr>
    <tr>
      <td>Emphysema</td>
      <td>424</td>
      <td>44</td>
      <td>103</td>
      <td>30</td>
      <td>359</td>
      <td>2516</td>
      <td>36</td>
      <td>4</td>
      <td>449</td>
      <td>215</td>
      <td>115</td>
      <td>151</td>
      <td>23</td>
      <td>747</td>
    </tr>
    <tr>
      <td>Fibrosis</td>
      <td>220</td>
      <td>52</td>
      <td>79</td>
      <td>9</td>
      <td>188</td>
      <td>36</td>
      <td>1686</td>
      <td>8</td>
      <td>345</td>
      <td>117</td>
      <td>166</td>
      <td>176</td>
      <td>11</td>
      <td>80</td>
    </tr>
    <tr>
      <td>Hernia</td>
      <td>40</td>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>21</td>
      <td>4</td>
      <td>8</td>
      <td>227</td>
      <td>33</td>
      <td>25</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <td>Infiltration</td>
      <td>3264</td>
      <td>587</td>
      <td>1221</td>
      <td>981</td>
      <td>4000</td>
      <td>449</td>
      <td>345</td>
      <td>33</td>
      <td>19891</td>
      <td>1157</td>
      <td>1546</td>
      <td>750</td>
      <td>605</td>
      <td>946</td>
    </tr>
    <tr>
      <td>Mass</td>
      <td>739</td>
      <td>102</td>
      <td>610</td>
      <td>129</td>
      <td>1254</td>
      <td>215</td>
      <td>117</td>
      <td>25</td>
      <td>1157</td>
      <td>5779</td>
      <td>906</td>
      <td>451</td>
      <td>71</td>
      <td>431</td>
    </tr>
    <tr>
      <td>Nodule</td>
      <td>590</td>
      <td>108</td>
      <td>428</td>
      <td>131</td>
      <td>912</td>
      <td>115</td>
      <td>166</td>
      <td>10</td>
      <td>1546</td>
      <td>906</td>
      <td>6331</td>
      <td>411</td>
      <td>70</td>
      <td>341</td>
    </tr>
    <tr>
      <td>Pleural_Thickening</td>
      <td>496</td>
      <td>111</td>
      <td>251</td>
      <td>64</td>
      <td>849</td>
      <td>151</td>
      <td>176</td>
      <td>8</td>
      <td>750</td>
      <td>451</td>
      <td>411</td>
      <td>3384</td>
      <td>48</td>
      <td>289</td>
    </tr>
    <tr>
      <td>Pneumonia</td>
      <td>262</td>
      <td>41</td>
      <td>123</td>
      <td>340</td>
      <td>268</td>
      <td>23</td>
      <td>11</td>
      <td>3</td>
      <td>605</td>
      <td>71</td>
      <td>70</td>
      <td>48</td>
      <td>1430</td>
      <td>41</td>
    </tr>
    <tr>
      <td>Pneumothorax</td>
      <td>774</td>
      <td>49</td>
      <td>223</td>
      <td>33</td>
      <td>996</td>
      <td>747</td>
      <td>80</td>
      <td>9</td>
      <td>946</td>
      <td>431</td>
      <td>341</td>
      <td>289</td>
      <td>41</td>
      <td>5301</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot co-occurrence matrix
plt.subplots(1, 1, figsize=(10,10))
sb.heatmap(cooccur_mat.applymap(math.sqrt), cmap='Purples', cbar=False)
plt.title('Co-occurrence of Chest Conditions')
plt.ylim(-0.05, len(all_conditions)+0.05)
plt.show()
```


![png](output_17_0.png)



```python
cooccur_mat.applymap(math.log10)
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
      <th>Atelectasis</th>
      <th>Cardiomegaly</th>
      <th>Consolidation</th>
      <th>Edema</th>
      <th>Effusion</th>
      <th>Emphysema</th>
      <th>Fibrosis</th>
      <th>Hernia</th>
      <th>Infiltration</th>
      <th>Mass</th>
      <th>Nodule</th>
      <th>Pleural_Thickening</th>
      <th>Pneumonia</th>
      <th>Pneumothorax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Atelectasis</td>
      <td>4.062883</td>
      <td>2.568202</td>
      <td>3.087426</td>
      <td>2.344392</td>
      <td>3.515211</td>
      <td>2.627366</td>
      <td>2.342423</td>
      <td>1.602060</td>
      <td>3.513750</td>
      <td>2.868644</td>
      <td>2.770852</td>
      <td>2.695482</td>
      <td>2.418301</td>
      <td>2.888741</td>
    </tr>
    <tr>
      <td>Cardiomegaly</td>
      <td>2.568202</td>
      <td>3.443419</td>
      <td>2.227887</td>
      <td>2.103804</td>
      <td>3.026533</td>
      <td>1.643453</td>
      <td>1.716003</td>
      <td>0.845098</td>
      <td>2.768638</td>
      <td>2.008600</td>
      <td>2.033424</td>
      <td>2.045323</td>
      <td>1.612784</td>
      <td>1.690196</td>
    </tr>
    <tr>
      <td>Consolidation</td>
      <td>3.087426</td>
      <td>2.227887</td>
      <td>3.669038</td>
      <td>2.209515</td>
      <td>3.109579</td>
      <td>2.012837</td>
      <td>1.897627</td>
      <td>0.602060</td>
      <td>3.086716</td>
      <td>2.785330</td>
      <td>2.631444</td>
      <td>2.399674</td>
      <td>2.089905</td>
      <td>2.348305</td>
    </tr>
    <tr>
      <td>Edema</td>
      <td>2.344392</td>
      <td>2.103804</td>
      <td>2.209515</td>
      <td>3.362105</td>
      <td>2.773055</td>
      <td>1.477121</td>
      <td>0.954243</td>
      <td>0.477121</td>
      <td>2.991669</td>
      <td>2.110590</td>
      <td>2.117271</td>
      <td>1.806180</td>
      <td>2.531479</td>
      <td>1.518514</td>
    </tr>
    <tr>
      <td>Effusion</td>
      <td>3.515211</td>
      <td>3.026533</td>
      <td>3.109579</td>
      <td>2.773055</td>
      <td>4.124374</td>
      <td>2.555094</td>
      <td>2.274158</td>
      <td>1.322219</td>
      <td>3.602060</td>
      <td>3.098298</td>
      <td>2.959995</td>
      <td>2.928908</td>
      <td>2.428135</td>
      <td>2.998259</td>
    </tr>
    <tr>
      <td>Emphysema</td>
      <td>2.627366</td>
      <td>1.643453</td>
      <td>2.012837</td>
      <td>1.477121</td>
      <td>2.555094</td>
      <td>3.400711</td>
      <td>1.556303</td>
      <td>0.602060</td>
      <td>2.652246</td>
      <td>2.332438</td>
      <td>2.060698</td>
      <td>2.178977</td>
      <td>1.361728</td>
      <td>2.873321</td>
    </tr>
    <tr>
      <td>Fibrosis</td>
      <td>2.342423</td>
      <td>1.716003</td>
      <td>1.897627</td>
      <td>0.954243</td>
      <td>2.274158</td>
      <td>1.556303</td>
      <td>3.226858</td>
      <td>0.903090</td>
      <td>2.537819</td>
      <td>2.068186</td>
      <td>2.220108</td>
      <td>2.245513</td>
      <td>1.041393</td>
      <td>1.903090</td>
    </tr>
    <tr>
      <td>Hernia</td>
      <td>1.602060</td>
      <td>0.845098</td>
      <td>0.602060</td>
      <td>0.477121</td>
      <td>1.322219</td>
      <td>0.602060</td>
      <td>0.903090</td>
      <td>2.356026</td>
      <td>1.518514</td>
      <td>1.397940</td>
      <td>1.000000</td>
      <td>0.903090</td>
      <td>0.477121</td>
      <td>0.954243</td>
    </tr>
    <tr>
      <td>Infiltration</td>
      <td>3.513750</td>
      <td>2.768638</td>
      <td>3.086716</td>
      <td>2.991669</td>
      <td>3.602060</td>
      <td>2.652246</td>
      <td>2.537819</td>
      <td>1.518514</td>
      <td>4.298657</td>
      <td>3.063333</td>
      <td>3.189209</td>
      <td>2.875061</td>
      <td>2.781755</td>
      <td>2.975891</td>
    </tr>
    <tr>
      <td>Mass</td>
      <td>2.868644</td>
      <td>2.008600</td>
      <td>2.785330</td>
      <td>2.110590</td>
      <td>3.098298</td>
      <td>2.332438</td>
      <td>2.068186</td>
      <td>1.397940</td>
      <td>3.063333</td>
      <td>3.761853</td>
      <td>2.957128</td>
      <td>2.654177</td>
      <td>1.851258</td>
      <td>2.634477</td>
    </tr>
    <tr>
      <td>Nodule</td>
      <td>2.770852</td>
      <td>2.033424</td>
      <td>2.631444</td>
      <td>2.117271</td>
      <td>2.959995</td>
      <td>2.060698</td>
      <td>2.220108</td>
      <td>1.000000</td>
      <td>3.189209</td>
      <td>2.957128</td>
      <td>3.801472</td>
      <td>2.613842</td>
      <td>1.845098</td>
      <td>2.532754</td>
    </tr>
    <tr>
      <td>Pleural_Thickening</td>
      <td>2.695482</td>
      <td>2.045323</td>
      <td>2.399674</td>
      <td>1.806180</td>
      <td>2.928908</td>
      <td>2.178977</td>
      <td>2.245513</td>
      <td>0.903090</td>
      <td>2.875061</td>
      <td>2.654177</td>
      <td>2.613842</td>
      <td>3.529430</td>
      <td>1.681241</td>
      <td>2.460898</td>
    </tr>
    <tr>
      <td>Pneumonia</td>
      <td>2.418301</td>
      <td>1.612784</td>
      <td>2.089905</td>
      <td>2.531479</td>
      <td>2.428135</td>
      <td>1.361728</td>
      <td>1.041393</td>
      <td>0.477121</td>
      <td>2.781755</td>
      <td>1.851258</td>
      <td>1.845098</td>
      <td>1.681241</td>
      <td>3.155336</td>
      <td>1.612784</td>
    </tr>
    <tr>
      <td>Pneumothorax</td>
      <td>2.888741</td>
      <td>1.690196</td>
      <td>2.348305</td>
      <td>1.518514</td>
      <td>2.998259</td>
      <td>2.873321</td>
      <td>1.903090</td>
      <td>0.954243</td>
      <td>2.975891</td>
      <td>2.634477</td>
      <td>2.532754</td>
      <td>2.460898</td>
      <td>1.612784</td>
      <td>3.724358</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create column with labels for model training
data_labels['conditions'] = data_labels[all_conditions].values.tolist()
```

## Data Preparation


```python
# Create mapping from image names to file paths
path_pattern = os.path.join('data', 'images_*', 'images', '*.png')
img_path_list = glob.glob(path_pattern)
img_path_dict = {os.path.basename(img_path): img_path for img_path in img_path_list}
```


```python
# Add image path for each record in dataset
data_labels['img_path'] = data_labels['Image Index'].map(lambda img_name: img_path_dict[img_name])
```


```python
# Split data into training, validation, and test sets
train_data, test_data = train_test_split(data_labels, test_size=0.1, shuffle=True,
                                        stratify = data_labels['Finding Labels'].map(lambda x: x[:3]))
```


```python
train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=True,
                                       stratify = train_data['Finding Labels'].map(lambda x: x[:3]))
```


```python
print('Train data dimensions:', train_data.shape)
print('Validation data dimensions:', val_data.shape)
print('Test data dimensions:', test_data.shape)
```

    Train data dimensions: (37260, 28)
    Validation data dimensions: (9315, 28)
    Test data dimensions: (5176, 28)
    

### Data Generator Setup 


```python
# Create data generator to read images in batches
all_conditions = list(all_conditions)
IMG_SIZE=(256, 256)
datagen = ImageDataGenerator()

train_datagen = datagen.flow_from_dataframe(train_data,
                                           x_col='img_path',
                                           y_col=all_conditions,
                                           target_size=IMG_SIZE,
                                           class_mode='raw', 
                                           batch_size=32)

val_datagen = datagen.flow_from_dataframe(val_data,
                                          x_col='img_path',
                                          y_col=all_conditions,
                                          target_size=IMG_SIZE,
                                          class_mode='raw',
                                          batch_size=64)

test_datagen = datagen.flow_from_dataframe(test_data,
                                           x_col='img_path',
                                           y_col=all_conditions,
                                           target_size=IMG_SIZE,
                                           class_mode='raw',
                                           batch_size=1024)


```

    Found 37260 validated image filenames.
    Found 9315 validated image filenames.
    Found 5176 validated image filenames.
    


```python
# Visualize first batch of images
train_batch = next(train_datagen)[:9]

fig, axs = plt.subplots(3,3, figsize=(16,16))
axs = axs.flatten()

for (img_x, img_y, img_ax) in zip(train_x, train_y, axs):
    img_ax.imshow(img_x[:,:,0], cmap='Greys')
    img_x_labels = ' + '.join([cond for cond, label in zip(all_conditions, img_y) if label==1])
    img_ax.set_title(img_x_labels)
    img_ax.axis('off')

```


![png](output_28_0.png)


## Build Models 

### CNN 


```python
# Define optimizer
opt = keras.optimizers.adam_v2.Adam(learning_rate=0.001)
topk = keras.metrics.TopKCategoricalAccuracy(k=5)
```


```python
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                                input_shape=(train_x.shape[1:])))
model.add(keras.layers.MaxPool2D((2,2), strides=2))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPool2D((2,2), strides=2))
model.add(keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPool2D((2,2), strides=2))
model.add(keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPool2D((2,2), strides=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(len(all_conditions), activation='sigmoid'))
print(model.summary())

```

    Model: "sequential_12"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_46 (Conv2D)           (None, 256, 256, 32)      896       
    _________________________________________________________________
    max_pooling2d_46 (MaxPooling (None, 128, 128, 32)      0         
    _________________________________________________________________
    conv2d_47 (Conv2D)           (None, 128, 128, 64)      18496     
    _________________________________________________________________
    max_pooling2d_47 (MaxPooling (None, 64, 64, 64)        0         
    _________________________________________________________________
    conv2d_48 (Conv2D)           (None, 64, 64, 128)       73856     
    _________________________________________________________________
    max_pooling2d_48 (MaxPooling (None, 32, 32, 128)       0         
    _________________________________________________________________
    conv2d_49 (Conv2D)           (None, 32, 32, 256)       295168    
    _________________________________________________________________
    max_pooling2d_49 (MaxPooling (None, 16, 16, 256)       0         
    _________________________________________________________________
    flatten_11 (Flatten)         (None, 65536)             0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 14)                917518    
    =================================================================
    Total params: 1,305,934
    Trainable params: 1,305,934
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[topk])
```


```python
# Run for 2 epochs to conserve computing resources
history = model.fit(train_datagen, 
                    steps_per_epoch=100,
                    validation_steps=50,
                    validation_data=val_datagen, 
                    epochs=2)
```

    Epoch 1/2
    100/100 [==============================] - 188s 2s/step - loss: 6.7952 - top_k_categorical_accuracy: 0.6546 - val_loss: 0.3059 - val_top_k_categorical_accuracy: 0.7350
    Epoch 2/2
    100/100 [==============================] - 186s 2s/step - loss: 0.3024 - top_k_categorical_accuracy: 0.7387 - val_loss: 0.3062 - val_top_k_categorical_accuracy: 0.7359
    


```python
test_x, test_y = next(test_datagen)
```

## Model Evaluation 


```python
# Predict model 
y_pred = model.predict(test_x)
```


```python
y_pred[:5]
```




    array([[0.20166901, 0.07214871, 0.05906156, 0.02727956, 0.30680323,
            0.01325548, 0.06997937, 0.0037351 , 0.25562257, 0.20107701,
            0.12394708, 0.10428512, 0.01202098, 0.14555046],
           [0.24328488, 0.10327926, 0.08492112, 0.0667983 , 0.33003476,
            0.0241212 , 0.02279302, 0.0064249 , 0.33249992, 0.10642788,
            0.09942654, 0.05880937, 0.02358162, 0.06386936],
           [0.25620204, 0.05903396, 0.10174575, 0.04916012, 0.24418822,
            0.06929016, 0.0540528 , 0.01102525, 0.32861528, 0.19259602,
            0.17040068, 0.09244871, 0.03352752, 0.08919248],
           [0.17195037, 0.03534457, 0.14248595, 0.04134786, 0.28238672,
            0.05141163, 0.01547927, 0.00231597, 0.3844045 , 0.11465839,
            0.11554116, 0.06628197, 0.01883331, 0.04506251],
           [0.29151464, 0.06840333, 0.08437636, 0.04442558, 0.2645261 ,
            0.06935498, 0.04514971, 0.00934643, 0.36389688, 0.1545091 ,
            0.14699984, 0.09233651, 0.03074557, 0.09404346]], dtype=float32)




```python
x = range(len(all_conditions))

# Proportion of conditions predicted by model
pred_prop = np.mean(y_pred, axis=0)

# Actual proportion of conditions in test data
actual_prop = np.mean(test_y, axis=0)


```


```python
# Smooth distribution curve 
x_new = np.arange(0, len(all_conditions) - 1, 0.1)

interp_pred = interp1d(x, pred_prop, kind=2)
interp_actual = interp1d(x, actual_prop, kind=2)

pred_prop_interp = interp_pred(x_new)
actual_prop_interp = interp_actual(x_new)
```


```python
# Plot distributions
fig, ax = plt.subplots(1, 1, figsize=(12,8))

ax.plot(x_new, pred_prop_interp, c='blue', label='Predicted')
ax.plot(x_new, actual_prop_interp, c='purple', label='Actual')
ax.set_title('Proportions of Chest Conditions - Predicted versus Actual')
ax.set_xticks(range(len(all_conditions)))
ax.set_xticklabels(all_conditions, rotation=90)
ax.legend()
plt.show()
```


![png](output_41_0.png)


## Retrospective - Observed Challenges and Potential Improvements 

Observed challenges:
* Multilabel classification metrics - metrics that accurately represent performance in multilabel classification problems are difficult to derive and implement.  Each patient (and hence X-ray) may have multiple conditions, so there will often be scenarios of partial correctness where most but not all of the true and predicted labels match. 
* Labeling accuracy - Labels are extracted from physician notes via NLP methods and may not be entirely accurate.
* Computing resource limitations - CNNs require large amounts of RAM and processing power to run efficiently.  Training iterations (i.e., epochs) were constrained.

Areas for additional improvement/exploration:
* Add new relevant features to model - X-ray image data can be supplemented by factors like patient age, gender, and medical history in predicting patient conditions.
* Resampling to mitigate class imbalance - Because of the rarity of some conditions (e.g., hernia), the dataset is very imbalanced and can result in training issues.  The data can be resampled to increase weights of rare conditions and decrease weights of common conditions.


```python

```
