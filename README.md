# Exploratory Data Analysis (EDA) Script

This Python script provides a set of functions for conducting exploratory data analysis (EDA) on a dataset. It utilizes popular data manipulation and visualization libraries such as NumPy, Pandas, Seaborn, and Matplotlib.

## Key Functions

1. **check_df**: Displays basic characteristics of a DataFrame.
2. **grab_col_names**: Identifies categorical, numeric, and categorical ordinal variables in the dataset.
3. **cat_summary**: Summarizes and optionally visualizes value counts and ratios for categorical variables.
4. **num_summary**: Provides descriptive statistics and optional visualization for numerical variables.
5. **target_summary_with_cat**: Computes the mean of the target variable based on a categorical column.
6. **target_summary_with_num**: Computes the mean of the target variable based on a numerical column.
7. **hig_correlated_cols**: Identifies columns with high correlation and optionally visualizes the correlation matrix.

## Example Usage

```python
# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading your dataset
df = pd.read_csv('your_dataset.csv')

# Applying functions
check_df(df, head=10)
cat_summary(df, 'category_column', plot=True)
num_summary(df, 'numeric_column', plot=True)
target_summary_with_cat(df, 'target_column', 'categorical_column')
hig_correlated_cols(df, plot=True, corr_th=0.85)
```

---

# Keşifsel Veri Analizi Script'i

Bu Python script'i, bir veri kümesi üzerinde keşifsel veri analizi yapmak için kullanılan bir dizi fonksiyon içermektedir. NumPy, Pandas, Seaborn ve Matplotlib gibi yaygın veri manipülasyonu ve görselleştirme kütüphanelerini kullanır.

## Temel Fonksiyonlar

1. **check_df**: Bir DataFrame'in temel özelliklerini gösterir.
2. **grab_col_names**: Veri kümesindeki kategorik, sayısal ve kategorik ordinal değişkenleri belirler.
3. **cat_summary**: Kategorik değişkenler için değer sayılarını ve oranları özetler ve isteğe bağlı olarak görselleştirir.
4. **num_summary**: Sayısal değişkenler için açıklayıcı istatistikleri ve isteğe bağlı olarak görselleştirmeyi sağlar.
5. **target_summary_with_cat**: Bir kategorik sütuna dayalı olarak hedef değişkenin ortalamasını hesaplar.
6. **target_summary_with_num**: Bir sayısal sütuna dayalı olarak hedef değişkenin ortalamasını hesaplar.
7. **hig_correlated_cols**: Yüksek korelasyona sahip sütunları belirler ve isteğe bağlı olarak korelasyon matrisini görselleştirir.

## Örnek Kullanım

```python
# Gerekli kütüphaneleri içe aktarma
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri kümenizi yükleme
df = pd.read_csv('veri_kumesi.csv')

# Fonksiyonları uygulama
check_df(df, head=10)
cat_summary(df, 'kategorik_sutun', plot=True)
num_summary(df, 'sayisal_sutun', plot=True)
target_summary_with_cat(df, 'hedef_sutun', 'kategorik_sutun')
hig_correlated_cols(df, plot=True, corr_th=0.85)
```

## Yazar
Mehmet Can Duru
