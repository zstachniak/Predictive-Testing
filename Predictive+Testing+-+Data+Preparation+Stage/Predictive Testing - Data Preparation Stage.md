
# Evaluating the Predictive Capabilities of "Predictive Testing"
## Predicting NCLEX Success for Nursing Students
Author: Alexander Stachniak

# Data Preparation Stage
Data preparation is an important part of any predictive modeling. The original data was gathered and anonymized on-site. In order to protect the privacy and confidentiality of the data, the data gathering stage will not be discussed in depth. No cleaning was done with the original data before it was pickled and saved.
## Goals
* Prepare a target dataset
* Clean the data and perform necessary preprocessing steps
* Review training data for covariance
* Perform data reduction (if warranted)

# Modules
All Python modules used are fairly common and will be installed as a default for users running Anaconda.


```python
# Standard modules
import pandas as pd
import numpy as np
import pickle
import datetime

# Plotting modules
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# Display plots within Jupyter Notebook
%matplotlib inline
# Set default plot size
plt.rcParams['figure.figsize'] = (14, 5)
# Seaborn used for its excellent correlation heatmap plot
import seaborn as sns

# SciKit Learn Modules
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import scale, Imputer, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# One module from SciPy used for calculating percentile scores
from scipy.stats import percentileofscore

# Caution: use only when certain of results
import warnings
warnings.filterwarnings('ignore')
```

## Loading the Data


```python
with open('NCLEX_data.pickle', 'rb') as file:
    nclex = pickle.load(file)
with open('Grades_data.pickle', 'rb') as file:
    grades = pickle.load(file)
with open('PA_data.pickle', 'rb') as file:
    pa = pickle.load(file)
```

## Preparing Target Variable
Our target variable is the result of the NCLEX exam, either "Pass" or "Fail." As our first data preparation step, we will ensure that our data is consistent.


```python
# Set ID as the index
nclex = nclex.set_index('Student ID')
# Sort by the index (ID)
nclex = nclex.sort_index()
```


```python
# View data
nclex.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Result</th>
    </tr>
    <tr>
      <th>Student ID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PASS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PASS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pass</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FAIL</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PASS</td>
    </tr>
  </tbody>
</table>
</div>




```python
# As we can see, we need to do account for some variance in how the target attribute is stored
nclex['Result'].unique().tolist()
```




    ['PASS', 'pass', 'FAIL', 'fail', 'Pass', 'Fail']




```python
# Convert Result to integer values
result_map = {'FAIL': 0,
              'Fail': 0,
              'fail': 0,
              'PASS': 1,
              'Pass': 1,
              'pass': 1}
nclex['Result'] = nclex['Result'].map(result_map)
```


```python
# View the duplicates to ensure we're not going to lose any data
nclex[nclex.index.duplicated(keep=False)]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Result</th>
    </tr>
    <tr>
      <th>Student ID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>71</th>
      <td>0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1411</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1411</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We can use the tilde (inverse) and duplicated method to drop duplicates
nclex = nclex[~nclex.index.duplicated(keep='first')]
```


```python
# View target data
nclex.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Result</th>
    </tr>
    <tr>
      <th>Student ID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Review class imbalance
nclex['Result'].value_counts()
```




    1    877
    0    104
    Name: Result, dtype: int64



## Preparing Traditional Grades Data


```python
# Set ID as the index
grades = grades.set_index('Student ID')
# Sort by the index (ID)
grades = grades.sort_index()
```


```python
# View the data (long format)
grades.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Term</th>
      <th>Grade</th>
      <th>Catalog</th>
      <th>Course Descr</th>
      <th>Tot Enrl</th>
      <th>Mode</th>
      <th>Course GPA</th>
    </tr>
    <tr>
      <th>Student ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0865</td>
      <td>A</td>
      <td>301</td>
      <td>INTRO/ART &amp; SCIENCE NURSING I</td>
      <td>10</td>
      <td>P</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0900</td>
      <td>A</td>
      <td>498</td>
      <td>PROFESSIONAL NURSE ROLE DEVELP</td>
      <td>28</td>
      <td>P</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0895</td>
      <td>A</td>
      <td>482</td>
      <td>INTRO TO EPIDEMIOLOGY</td>
      <td>30</td>
      <td>OL</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0870</td>
      <td>A</td>
      <td>422</td>
      <td>APPLIED PHYSIOLOGY</td>
      <td>31</td>
      <td>P</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0870</td>
      <td>A</td>
      <td>313</td>
      <td>TRENDS &amp; ISSUES IN NURSING</td>
      <td>23</td>
      <td>P</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# List of applicable courses
course_list = ['301', '302', '303', '307', '322', '332', '400', '401', '422', '426', '431', '440', '441', '442', '445', '460', '472', '481', '540', '598']
```


```python
# Drop course grades that are not applicable
grades = grades[grades['Catalog'].isin(course_list)]
```


```python
# Drop unnecessary fields
grades = grades[['Catalog', 'Course GPA']]
```


```python
# Convert from long to wide format
grades = grades.pivot(columns='Catalog')
# Drop unnecessary level in multi-index
grades.columns = grades.columns.droplevel(0)
```


```python
# Drop students who do not have all grades
grades.dropna(axis=0, how='any', inplace=True)
```


```python
# Scale numeric data using the MinMaxScaler
min_max_scaler = MinMaxScaler()
grades[course_list] = min_max_scaler.fit_transform(grades[course_list])
```


```python
# Review data
grades.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Catalog</th>
      <th>301</th>
      <th>302</th>
      <th>303</th>
      <th>307</th>
      <th>322</th>
      <th>332</th>
      <th>400</th>
      <th>401</th>
      <th>422</th>
      <th>426</th>
      <th>431</th>
      <th>440</th>
      <th>441</th>
      <th>442</th>
      <th>445</th>
      <th>460</th>
      <th>472</th>
      <th>481</th>
      <th>540</th>
      <th>598</th>
    </tr>
    <tr>
      <th>Student ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.00</td>
      <td>0.695652</td>
      <td>1.00</td>
      <td>0.823529</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.588235</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.65</td>
      <td>0.823529</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.85</td>
      <td>0.869565</td>
      <td>0.85</td>
      <td>1.000000</td>
      <td>0.65</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.434783</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.35</td>
      <td>0.823529</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.65</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.00</td>
      <td>0.869565</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>0.85</td>
      <td>0.7</td>
      <td>0.823529</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.869565</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>0.823529</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.00</td>
      <td>0.869565</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>0.823529</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.869565</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.65</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.85</td>
      <td>1.000000</td>
      <td>0.65</td>
      <td>1.000000</td>
      <td>0.85</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Reviewing for Correlations and Covariance
With 20 variables for grades, it is a good idea to check for evidence of high correlation or covariance which could negatively affect our predictions. Looking at a correlation matrix, it is clear that there is some correlation between many of the grades. But is it enough to be worried about?


```python
# Calculate correlation matrix
corr = grades.corr()
# Create a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up plot
f, ax = plt.subplots(figsize=(11,9))
# Generate a colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x218c3110588>




![png](output_26_1.png)


It can be hard to tell just how much correlation there is with a colormap. Below, we print the actual values using some basic CSS styling through Pandas. All correlations above 0.50 or below -0.50 are highlighted in red. This view makes it much more obvious that we don't have too much cause for concern.


```python
def high_corr_red(value):
    """Returns a string with the css property `'color: red'` for values over >= 0.50 or <= -0.50."""
    color = 'red' if value != 1 and (value >= 0.50 or value <= -0.50) else 'black'
    return 'color: %s' % color

# Chaining multiple styles together
corr_styled = corr.style.\
    applymap(high_corr_red).\
    format("{:.2}")
# Call to output
corr_styled
```




<style  type="text/css" >
    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col1 {
            color:  red;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col4 {
            color:  red;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col0 {
            color:  red;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col4 {
            color:  red;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col0 {
            color:  red;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col1 {
            color:  red;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col16 {
            color:  red;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col13 {
            color:  red;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col12 {
            color:  red;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col9 {
            color:  red;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col19 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col0 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col1 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col2 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col3 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col4 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col5 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col6 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col7 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col8 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col9 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col10 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col11 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col12 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col13 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col14 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col15 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col16 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col17 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col18 {
            color:  black;
        }    #T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col19 {
            color:  black;
        }</style>  
<table id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62" > 
<thead>    <tr> 
        <th class="index_name level0" >Catalog</th> 
        <th class="col_heading level0 col0" >301</th> 
        <th class="col_heading level0 col1" >302</th> 
        <th class="col_heading level0 col2" >303</th> 
        <th class="col_heading level0 col3" >307</th> 
        <th class="col_heading level0 col4" >322</th> 
        <th class="col_heading level0 col5" >332</th> 
        <th class="col_heading level0 col6" >400</th> 
        <th class="col_heading level0 col7" >401</th> 
        <th class="col_heading level0 col8" >422</th> 
        <th class="col_heading level0 col9" >426</th> 
        <th class="col_heading level0 col10" >431</th> 
        <th class="col_heading level0 col11" >440</th> 
        <th class="col_heading level0 col12" >441</th> 
        <th class="col_heading level0 col13" >442</th> 
        <th class="col_heading level0 col14" >445</th> 
        <th class="col_heading level0 col15" >460</th> 
        <th class="col_heading level0 col16" >472</th> 
        <th class="col_heading level0 col17" >481</th> 
        <th class="col_heading level0 col18" >540</th> 
        <th class="col_heading level0 col19" >598</th> 
    </tr>    <tr> 
        <th class="index_name level0" >Catalog</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row0" class="row_heading level0 row0" >301</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col0" class="data row0 col0" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col1" class="data row0 col1" >0.54</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col2" class="data row0 col2" >0.098</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col3" class="data row0 col3" >0.3</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col4" class="data row0 col4" >0.51</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col5" class="data row0 col5" >0.44</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col6" class="data row0 col6" >0.27</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col7" class="data row0 col7" >0.15</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col8" class="data row0 col8" >0.31</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col9" class="data row0 col9" >0.41</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col10" class="data row0 col10" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col11" class="data row0 col11" >0.38</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col12" class="data row0 col12" >0.4</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col13" class="data row0 col13" >0.36</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col14" class="data row0 col14" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col15" class="data row0 col15" >0.44</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col16" class="data row0 col16" >0.41</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col17" class="data row0 col17" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col18" class="data row0 col18" >0.22</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row0_col19" class="data row0 col19" >0.13</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row1" class="row_heading level0 row1" >302</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col0" class="data row1 col0" >0.54</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col1" class="data row1 col1" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col2" class="data row1 col2" >-0.049</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col3" class="data row1 col3" >0.36</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col4" class="data row1 col4" >0.5</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col5" class="data row1 col5" >0.27</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col6" class="data row1 col6" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col7" class="data row1 col7" >0.084</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col8" class="data row1 col8" >0.23</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col9" class="data row1 col9" >0.47</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col10" class="data row1 col10" >0.12</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col11" class="data row1 col11" >0.36</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col12" class="data row1 col12" >0.49</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col13" class="data row1 col13" >0.35</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col14" class="data row1 col14" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col15" class="data row1 col15" >0.29</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col16" class="data row1 col16" >0.43</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col17" class="data row1 col17" >0.35</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col18" class="data row1 col18" >0.17</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row1_col19" class="data row1 col19" >0.14</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row2" class="row_heading level0 row2" >303</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col0" class="data row2 col0" >0.098</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col1" class="data row2 col1" >-0.049</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col2" class="data row2 col2" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col3" class="data row2 col3" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col4" class="data row2 col4" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col5" class="data row2 col5" >0.38</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col6" class="data row2 col6" >0.064</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col7" class="data row2 col7" >0.093</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col8" class="data row2 col8" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col9" class="data row2 col9" >0.028</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col10" class="data row2 col10" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col11" class="data row2 col11" >-0.066</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col12" class="data row2 col12" >-0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col13" class="data row2 col13" >-0.086</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col14" class="data row2 col14" >0.089</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col15" class="data row2 col15" >0.34</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col16" class="data row2 col16" >0.016</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col17" class="data row2 col17" >0.049</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col18" class="data row2 col18" >-0.051</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row2_col19" class="data row2 col19" >0.2</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row3" class="row_heading level0 row3" >307</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col0" class="data row3 col0" >0.3</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col1" class="data row3 col1" >0.36</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col2" class="data row3 col2" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col3" class="data row3 col3" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col4" class="data row3 col4" >0.36</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col5" class="data row3 col5" >0.3</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col6" class="data row3 col6" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col7" class="data row3 col7" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col8" class="data row3 col8" >0.22</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col9" class="data row3 col9" >0.3</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col10" class="data row3 col10" >0.094</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col11" class="data row3 col11" >0.24</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col12" class="data row3 col12" >0.39</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col13" class="data row3 col13" >0.35</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col14" class="data row3 col14" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col15" class="data row3 col15" >0.27</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col16" class="data row3 col16" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col17" class="data row3 col17" >0.28</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col18" class="data row3 col18" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row3_col19" class="data row3 col19" >0.15</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row4" class="row_heading level0 row4" >322</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col0" class="data row4 col0" >0.51</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col1" class="data row4 col1" >0.5</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col2" class="data row4 col2" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col3" class="data row4 col3" >0.36</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col4" class="data row4 col4" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col5" class="data row4 col5" >0.44</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col6" class="data row4 col6" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col7" class="data row4 col7" >0.11</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col8" class="data row4 col8" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col9" class="data row4 col9" >0.43</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col10" class="data row4 col10" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col11" class="data row4 col11" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col12" class="data row4 col12" >0.4</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col13" class="data row4 col13" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col14" class="data row4 col14" >0.096</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col15" class="data row4 col15" >0.44</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col16" class="data row4 col16" >0.43</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col17" class="data row4 col17" >0.26</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col18" class="data row4 col18" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row4_col19" class="data row4 col19" >0.089</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row5" class="row_heading level0 row5" >332</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col0" class="data row5 col0" >0.44</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col1" class="data row5 col1" >0.27</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col2" class="data row5 col2" >0.38</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col3" class="data row5 col3" >0.3</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col4" class="data row5 col4" >0.44</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col5" class="data row5 col5" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col6" class="data row5 col6" >0.19</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col7" class="data row5 col7" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col8" class="data row5 col8" >0.036</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col9" class="data row5 col9" >0.39</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col10" class="data row5 col10" >0.086</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col11" class="data row5 col11" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col12" class="data row5 col12" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col13" class="data row5 col13" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col14" class="data row5 col14" >0.043</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col15" class="data row5 col15" >0.46</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col16" class="data row5 col16" >0.3</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col17" class="data row5 col17" >0.12</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col18" class="data row5 col18" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row5_col19" class="data row5 col19" >-0.017</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row6" class="row_heading level0 row6" >400</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col0" class="data row6 col0" >0.27</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col1" class="data row6 col1" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col2" class="data row6 col2" >0.064</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col3" class="data row6 col3" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col4" class="data row6 col4" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col5" class="data row6 col5" >0.19</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col6" class="data row6 col6" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col7" class="data row6 col7" >0.32</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col8" class="data row6 col8" >0.12</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col9" class="data row6 col9" >0.19</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col10" class="data row6 col10" >0.064</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col11" class="data row6 col11" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col12" class="data row6 col12" >0.22</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col13" class="data row6 col13" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col14" class="data row6 col14" >0.073</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col15" class="data row6 col15" >0.23</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col16" class="data row6 col16" >0.26</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col17" class="data row6 col17" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col18" class="data row6 col18" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row6_col19" class="data row6 col19" >0.16</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row7" class="row_heading level0 row7" >401</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col0" class="data row7 col0" >0.15</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col1" class="data row7 col1" >0.084</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col2" class="data row7 col2" >0.093</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col3" class="data row7 col3" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col4" class="data row7 col4" >0.11</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col5" class="data row7 col5" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col6" class="data row7 col6" >0.32</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col7" class="data row7 col7" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col8" class="data row7 col8" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col9" class="data row7 col9" >0.05</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col10" class="data row7 col10" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col11" class="data row7 col11" >0.075</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col12" class="data row7 col12" >0.062</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col13" class="data row7 col13" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col14" class="data row7 col14" >0.087</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col15" class="data row7 col15" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col16" class="data row7 col16" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col17" class="data row7 col17" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col18" class="data row7 col18" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row7_col19" class="data row7 col19" >0.17</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row8" class="row_heading level0 row8" >422</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col0" class="data row8 col0" >0.31</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col1" class="data row8 col1" >0.23</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col2" class="data row8 col2" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col3" class="data row8 col3" >0.22</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col4" class="data row8 col4" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col5" class="data row8 col5" >0.036</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col6" class="data row8 col6" >0.12</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col7" class="data row8 col7" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col8" class="data row8 col8" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col9" class="data row8 col9" >0.0053</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col10" class="data row8 col10" >0.28</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col11" class="data row8 col11" >0.077</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col12" class="data row8 col12" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col13" class="data row8 col13" >0.23</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col14" class="data row8 col14" >0.05</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col15" class="data row8 col15" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col16" class="data row8 col16" >0.22</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col17" class="data row8 col17" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col18" class="data row8 col18" >0.075</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row8_col19" class="data row8 col19" >0.21</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row9" class="row_heading level0 row9" >426</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col0" class="data row9 col0" >0.41</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col1" class="data row9 col1" >0.47</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col2" class="data row9 col2" >0.028</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col3" class="data row9 col3" >0.3</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col4" class="data row9 col4" >0.43</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col5" class="data row9 col5" >0.39</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col6" class="data row9 col6" >0.19</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col7" class="data row9 col7" >0.05</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col8" class="data row9 col8" >0.0053</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col9" class="data row9 col9" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col10" class="data row9 col10" >0.058</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col11" class="data row9 col11" >0.38</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col12" class="data row9 col12" >0.41</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col13" class="data row9 col13" >0.26</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col14" class="data row9 col14" >0.11</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col15" class="data row9 col15" >0.43</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col16" class="data row9 col16" >0.55</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col17" class="data row9 col17" >0.19</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col18" class="data row9 col18" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row9_col19" class="data row9 col19" >-0.022</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row10" class="row_heading level0 row10" >431</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col0" class="data row10 col0" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col1" class="data row10 col1" >0.12</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col2" class="data row10 col2" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col3" class="data row10 col3" >0.094</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col4" class="data row10 col4" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col5" class="data row10 col5" >0.086</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col6" class="data row10 col6" >0.064</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col7" class="data row10 col7" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col8" class="data row10 col8" >0.28</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col9" class="data row10 col9" >0.058</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col10" class="data row10 col10" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col11" class="data row10 col11" >0.061</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col12" class="data row10 col12" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col13" class="data row10 col13" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col14" class="data row10 col14" >0.32</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col15" class="data row10 col15" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col16" class="data row10 col16" >0.17</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col17" class="data row10 col17" >0.19</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col18" class="data row10 col18" >0.072</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row10_col19" class="data row10 col19" >0.17</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row11" class="row_heading level0 row11" >440</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col0" class="data row11 col0" >0.38</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col1" class="data row11 col1" >0.36</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col2" class="data row11 col2" >-0.066</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col3" class="data row11 col3" >0.24</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col4" class="data row11 col4" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col5" class="data row11 col5" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col6" class="data row11 col6" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col7" class="data row11 col7" >0.075</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col8" class="data row11 col8" >0.077</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col9" class="data row11 col9" >0.38</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col10" class="data row11 col10" >0.061</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col11" class="data row11 col11" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col12" class="data row11 col12" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col13" class="data row11 col13" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col14" class="data row11 col14" >0.028</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col15" class="data row11 col15" >0.37</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col16" class="data row11 col16" >0.4</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col17" class="data row11 col17" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col18" class="data row11 col18" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row11_col19" class="data row11 col19" >0.063</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row12" class="row_heading level0 row12" >441</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col0" class="data row12 col0" >0.4</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col1" class="data row12 col1" >0.49</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col2" class="data row12 col2" >-0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col3" class="data row12 col3" >0.39</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col4" class="data row12 col4" >0.4</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col5" class="data row12 col5" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col6" class="data row12 col6" >0.22</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col7" class="data row12 col7" >0.062</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col8" class="data row12 col8" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col9" class="data row12 col9" >0.41</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col10" class="data row12 col10" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col11" class="data row12 col11" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col12" class="data row12 col12" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col13" class="data row12 col13" >0.51</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col14" class="data row12 col14" >0.23</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col15" class="data row12 col15" >0.27</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col16" class="data row12 col16" >0.43</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col17" class="data row12 col17" >0.24</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col18" class="data row12 col18" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row12_col19" class="data row12 col19" >0.099</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row13" class="row_heading level0 row13" >442</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col0" class="data row13 col0" >0.36</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col1" class="data row13 col1" >0.35</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col2" class="data row13 col2" >-0.086</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col3" class="data row13 col3" >0.35</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col4" class="data row13 col4" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col5" class="data row13 col5" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col6" class="data row13 col6" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col7" class="data row13 col7" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col8" class="data row13 col8" >0.23</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col9" class="data row13 col9" >0.26</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col10" class="data row13 col10" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col11" class="data row13 col11" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col12" class="data row13 col12" >0.51</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col13" class="data row13 col13" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col14" class="data row13 col14" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col15" class="data row13 col15" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col16" class="data row13 col16" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col17" class="data row13 col17" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col18" class="data row13 col18" >0.1</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row13_col19" class="data row13 col19" >0.098</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row14" class="row_heading level0 row14" >445</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col0" class="data row14 col0" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col1" class="data row14 col1" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col2" class="data row14 col2" >0.089</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col3" class="data row14 col3" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col4" class="data row14 col4" >0.096</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col5" class="data row14 col5" >0.043</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col6" class="data row14 col6" >0.073</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col7" class="data row14 col7" >0.087</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col8" class="data row14 col8" >0.05</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col9" class="data row14 col9" >0.11</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col10" class="data row14 col10" >0.32</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col11" class="data row14 col11" >0.028</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col12" class="data row14 col12" >0.23</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col13" class="data row14 col13" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col14" class="data row14 col14" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col15" class="data row14 col15" >0.046</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col16" class="data row14 col16" >0.15</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col17" class="data row14 col17" >0.17</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col18" class="data row14 col18" >0.011</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row14_col19" class="data row14 col19" >0.15</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row15" class="row_heading level0 row15" >460</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col0" class="data row15 col0" >0.44</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col1" class="data row15 col1" >0.29</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col2" class="data row15 col2" >0.34</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col3" class="data row15 col3" >0.27</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col4" class="data row15 col4" >0.44</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col5" class="data row15 col5" >0.46</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col6" class="data row15 col6" >0.23</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col7" class="data row15 col7" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col8" class="data row15 col8" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col9" class="data row15 col9" >0.43</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col10" class="data row15 col10" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col11" class="data row15 col11" >0.37</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col12" class="data row15 col12" >0.27</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col13" class="data row15 col13" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col14" class="data row15 col14" >0.046</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col15" class="data row15 col15" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col16" class="data row15 col16" >0.41</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col17" class="data row15 col17" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col18" class="data row15 col18" >0.19</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row15_col19" class="data row15 col19" >0.047</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row16" class="row_heading level0 row16" >472</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col0" class="data row16 col0" >0.41</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col1" class="data row16 col1" >0.43</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col2" class="data row16 col2" >0.016</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col3" class="data row16 col3" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col4" class="data row16 col4" >0.43</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col5" class="data row16 col5" >0.3</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col6" class="data row16 col6" >0.26</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col7" class="data row16 col7" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col8" class="data row16 col8" >0.22</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col9" class="data row16 col9" >0.55</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col10" class="data row16 col10" >0.17</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col11" class="data row16 col11" >0.4</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col12" class="data row16 col12" >0.43</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col13" class="data row16 col13" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col14" class="data row16 col14" >0.15</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col15" class="data row16 col15" >0.41</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col16" class="data row16 col16" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col17" class="data row16 col17" >0.38</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col18" class="data row16 col18" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row16_col19" class="data row16 col19" >0.22</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row17" class="row_heading level0 row17" >481</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col0" class="data row17 col0" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col1" class="data row17 col1" >0.35</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col2" class="data row17 col2" >0.049</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col3" class="data row17 col3" >0.28</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col4" class="data row17 col4" >0.26</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col5" class="data row17 col5" >0.12</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col6" class="data row17 col6" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col7" class="data row17 col7" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col8" class="data row17 col8" >0.33</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col9" class="data row17 col9" >0.19</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col10" class="data row17 col10" >0.19</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col11" class="data row17 col11" >0.25</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col12" class="data row17 col12" >0.24</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col13" class="data row17 col13" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col14" class="data row17 col14" >0.17</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col15" class="data row17 col15" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col16" class="data row17 col16" >0.38</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col17" class="data row17 col17" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col18" class="data row17 col18" >0.057</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row17_col19" class="data row17 col19" >0.3</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row18" class="row_heading level0 row18" >540</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col0" class="data row18 col0" >0.22</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col1" class="data row18 col1" >0.17</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col2" class="data row18 col2" >-0.051</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col3" class="data row18 col3" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col4" class="data row18 col4" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col5" class="data row18 col5" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col6" class="data row18 col6" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col7" class="data row18 col7" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col8" class="data row18 col8" >0.075</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col9" class="data row18 col9" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col10" class="data row18 col10" >0.072</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col11" class="data row18 col11" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col12" class="data row18 col12" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col13" class="data row18 col13" >0.1</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col14" class="data row18 col14" >0.011</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col15" class="data row18 col15" >0.19</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col16" class="data row18 col16" >0.18</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col17" class="data row18 col17" >0.057</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col18" class="data row18 col18" >1.0</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row18_col19" class="data row18 col19" >0.12</td> 
    </tr>    <tr> 
        <th id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62level0_row19" class="row_heading level0 row19" >598</th> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col0" class="data row19 col0" >0.13</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col1" class="data row19 col1" >0.14</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col2" class="data row19 col2" >0.2</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col3" class="data row19 col3" >0.15</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col4" class="data row19 col4" >0.089</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col5" class="data row19 col5" >-0.017</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col6" class="data row19 col6" >0.16</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col7" class="data row19 col7" >0.17</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col8" class="data row19 col8" >0.21</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col9" class="data row19 col9" >-0.022</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col10" class="data row19 col10" >0.17</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col11" class="data row19 col11" >0.063</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col12" class="data row19 col12" >0.099</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col13" class="data row19 col13" >0.098</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col14" class="data row19 col14" >0.15</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col15" class="data row19 col15" >0.047</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col16" class="data row19 col16" >0.22</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col17" class="data row19 col17" >0.3</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col18" class="data row19 col18" >0.12</td> 
        <td id="T_2ef0ed76_7ea9_11e7_a851_1866da0c9c62row19_col19" class="data row19 col19" >1.0</td> 
    </tr></tbody> 
</table> 



Still, it's not a bad idea to test PCA to see if there would be a benefit to reducing dimensionality. As can be seen in the output, a significant number of principal components would be needed to cover a majority of the variance in the data (e.g. 10 components to cover 80% of variance). This is not seen as a beneficial step to take, especially given that we would lose all interpretability with PCA.


```python
# Principal Component Analysis
pca = PCA(n_components=None)
grades_pca = pca.fit_transform(grades)
print('Explained Variance')
for pc in range(len(pca.explained_variance_ratio_)):
    print('PC{0}: {1:5.2f}%'.format(pc+1, pca.explained_variance_ratio_[pc]*100))
# Cumulative variance
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
# Plot it
plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,len(pca.explained_variance_ratio_)+1), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.show()
```

    Explained Variance
    PC1: 33.50%
    PC2: 10.78%
    PC3:  7.70%
    PC4:  5.65%
    PC5:  4.62%
    PC6:  4.36%
    PC7:  4.14%
    PC8:  3.80%
    PC9:  3.27%
    PC10:  3.20%
    PC11:  3.03%
    PC12:  2.71%
    PC13:  2.33%
    PC14:  2.23%
    PC15:  2.10%
    PC16:  1.94%
    PC17:  1.52%
    PC18:  1.29%
    PC19:  0.97%
    PC20:  0.84%



![png](output_30_1.png)


## Inner Join of Data
We want to be sure that we are not retaining student data in one dataframe that does not exist in the others. So, we will perform an inner join of student IDs, and retain data only for those IDs moving forward.


```python
# Multi-part operation:
# 1) Chain two merge statements together to produce an inner join on three dataframes
# 2) Keep only the "Student ID" field
# 3) Drop true duplicates
# 4) Reset index from 0
student_ids = nclex.merge(grades, left_index=True, right_index=True).merge(pa, left_index=True, right_on='Student ID')['Student ID'].drop_duplicates().reset_index(drop=True)
```


```python
# Keep only the student IDs that are represented across all dataframes
grades = grades.ix[student_ids]
nclex = nclex.ix[student_ids]
pa = pa[pa['Student ID'].isin(student_ids)]
```

## Preparing the Predictive Assessment Data
The predictive testing product contained multiple categories of assessments (e.g. Medical/Surgical, Pediatrics, etc.) as well as multiple versions within each category containing different questions. Assessment and Assessment ID identify the assessment, and Booklet ID is a unique identifier for the test-taker. Date Taken identifies the date and time that the exam was begun.

For each assessment, several numerical scores are provided:
* Score (raw score)
* National Mean
* National Percentile
* Program Mean
* Program Percentile
* Proficiency Level (meant to indicate level of mastery; is based on a proprietary algorithm).

Each assessment is further broken down by section, where each section indicates a theoretical grouping of questions intended to provide an indication of the level of mastery of specific sub-topics. Numerical scores are provided both for the overall assessment and for each individual section.


```python
pa.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Assessment</th>
      <th>Assessment ID</th>
      <th>Booklet ID</th>
      <th>Date Taken</th>
      <th>National Mean</th>
      <th>National Percentile</th>
      <th>Proficiency Level</th>
      <th>Program Mean</th>
      <th>Program Percentile</th>
      <th>Score</th>
      <th>Section</th>
      <th>Student ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>145314</th>
      <td>RN Community Health 2010 Form A</td>
      <td>50478</td>
      <td>49403969</td>
      <td>Mar  6 2013  8:27AM</td>
      <td>65.46</td>
      <td>99.0</td>
      <td>Level 3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>RN Community Health 2010 Form A</td>
      <td>1359</td>
    </tr>
    <tr>
      <th>145315</th>
      <td>RN Community Health 2010 Form A</td>
      <td>50478</td>
      <td>49403969</td>
      <td>Mar  6 2013  8:27AM</td>
      <td>67.75</td>
      <td>99.0</td>
      <td>Below Level 1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>90.0</td>
      <td>Management of Care</td>
      <td>1359</td>
    </tr>
    <tr>
      <th>145316</th>
      <td>RN Community Health 2010 Form A</td>
      <td>50478</td>
      <td>49403969</td>
      <td>Mar  6 2013  8:27AM</td>
      <td>28.13</td>
      <td>99.0</td>
      <td>Below Level 1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>Advocacy</td>
      <td>1359</td>
    </tr>
    <tr>
      <th>145317</th>
      <td>RN Community Health 2010 Form A</td>
      <td>50478</td>
      <td>49403969</td>
      <td>Mar  6 2013  8:27AM</td>
      <td>79.38</td>
      <td>99.0</td>
      <td>Below Level 1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>Case Management</td>
      <td>1359</td>
    </tr>
    <tr>
      <th>145318</th>
      <td>RN Community Health 2010 Form A</td>
      <td>50478</td>
      <td>49403969</td>
      <td>Mar  6 2013  8:27AM</td>
      <td>81.25</td>
      <td>99.0</td>
      <td>Below Level 1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>Client Rights</td>
      <td>1359</td>
    </tr>
  </tbody>
</table>
</div>



### Cleaning the Data
There are a few things we need to do to clean up the data so that it is easier to work with:
* Convert dates to datetime format so that we can perform calculations on them.
* Rename overall scores in each assessment to "Overall" rather than the same name as the assessment.
* Gather assessments into a few basic categories. Because the predictive testing vendor released new versions of the assessments every few years, in order to compare scores over time we need to know what these basic categories are.
* Represent Proficiency Level as an ordinal integer value.


```python
# Convert string representation of 'Date Taken' to a python DateTime
pa['Date Taken'] = pd.to_datetime(pa['Date Taken'])
```


```python
# Rename section headings that are 'overall' scores using a quick lambda function
def overall(Assessment, Section):
    if Section == Assessment:
        return 'Overall'
    else:
        return Section
pa['Section'] = pa.apply(lambda x: overall(x['Assessment'], x['Section']), axis=1)
```


```python
# Create categories for all Assessments for comparison purposes using a dictionary and a lambda function.
# Students may take multiple assessments for each category.
def Categorize (string):
    # If the dictionary key appears in the Assessment Name, apply the dictionary value as the Category.
    kw_dict = {
               'Mental Health': 'Psych',
               'Comprehensive': 'Comprehensive',
               'Medical-Surgical': 'MedSurg',
               'Medical Surgical': 'MedSurg',
               'Fundamentals': 'Fundamentals',
               'Essential Academic Skills': 'Fundamentals',
               'Pharmacology': 'Pharm',
               'Maternal': 'OB',
               'Community Health': 'Community',
               'Leadership': 'Leadership',
               'Care of Children': 'Peds',
               'Critical Thinking': 'Critical Thinking'
               }
    for key in kw_dict:
        if key in string:
            return kw_dict[key]
pa['Category'] = pa.apply(lambda x: Categorize(x['Assessment']), axis=1)
```


```python
# Convert Proficiency Levels to integer values using a dictionary mapping.
proficiency_map = {'Below Level 1': 0,
                   'Level 1': 1,
                   'Level 2': 2,
                   'Level 3': 3                   
                   }
pa['Proficiency Level'] = pa['Proficiency Level'].map(proficiency_map)
```

### Removing Duplicates
Before we can run any kind of analysis, we need to remove duplicates from our data. Some duplicates are true duplicates (all fields duplicated), so those are dropped without any loss of data. There also appears to be some malformed data, where everything is duplicated but all of the numeric scores are lost (value = 'Nan'). Again, these are dropped without any loss of data.


```python
# Drop true duplicates
pa = pa.drop_duplicates()
```


```python
# Drop NaN in National Percentile and/or National Mean (only accompanies malformed duplicates)
pa = pa.dropna(axis=0,how='any',subset=['National Percentile'])
pa = pa.dropna(axis=0,how='any',subset=['National Mean'])
```

### Missing Values
We will need a plan for dealing with any missing values before we can run any analysis. First, let's take a look at where our missing values reside.


```python
# Which columns contain NaNs?
pa.isnull().sum()
```




    Assessment                 0
    Assessment ID              0
    Booklet ID                 0
    Date Taken                 0
    National Mean              0
    National Percentile        0
    Proficiency Level      28919
    Program Mean            6885
    Program Percentile      6885
    Score                      0
    Section                    0
    Student ID                 0
    Category                   0
    dtype: int64



#### Impute Missing Values for Program Mean
Missing values in Program Mean are clustered in two specific assessments. Because we have almost 7,000 NaNs, and we don't want to have to drop this data when modeling, we will impute the missing values. Program Mean should be very simple to impute, however, since it should just be a mean value of all the scores for a particular section of a particular assessment. In order to get a little more granular than just replacing NaNs with the mean of the entire column (which wouldn't account for differences in sections or assessments), we will iterate over each section to impute mean values.


```python
# First, let's look at the data.
# We make a copy of the df to avoid making unwanted changes
Prog_Nans = pa.copy(deep=True)
Prog_Nans = Prog_Nans[Prog_Nans['Program Mean'].isnull()]
Prog_Nans.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Assessment</th>
      <th>Assessment ID</th>
      <th>Booklet ID</th>
      <th>Date Taken</th>
      <th>National Mean</th>
      <th>National Percentile</th>
      <th>Proficiency Level</th>
      <th>Program Mean</th>
      <th>Program Percentile</th>
      <th>Score</th>
      <th>Section</th>
      <th>Student ID</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>145314</th>
      <td>RN Community Health 2010 Form A</td>
      <td>50478</td>
      <td>49403969</td>
      <td>2013-03-06 08:27:00</td>
      <td>65.46</td>
      <td>99.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>Overall</td>
      <td>1359</td>
      <td>Community</td>
    </tr>
    <tr>
      <th>145315</th>
      <td>RN Community Health 2010 Form A</td>
      <td>50478</td>
      <td>49403969</td>
      <td>2013-03-06 08:27:00</td>
      <td>67.75</td>
      <td>99.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>90.0</td>
      <td>Management of Care</td>
      <td>1359</td>
      <td>Community</td>
    </tr>
    <tr>
      <th>145316</th>
      <td>RN Community Health 2010 Form A</td>
      <td>50478</td>
      <td>49403969</td>
      <td>2013-03-06 08:27:00</td>
      <td>28.13</td>
      <td>99.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>Advocacy</td>
      <td>1359</td>
      <td>Community</td>
    </tr>
    <tr>
      <th>145317</th>
      <td>RN Community Health 2010 Form A</td>
      <td>50478</td>
      <td>49403969</td>
      <td>2013-03-06 08:27:00</td>
      <td>79.38</td>
      <td>99.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>Case Management</td>
      <td>1359</td>
      <td>Community</td>
    </tr>
    <tr>
      <th>145318</th>
      <td>RN Community Health 2010 Form A</td>
      <td>50478</td>
      <td>49403969</td>
      <td>2013-03-06 08:27:00</td>
      <td>81.25</td>
      <td>99.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>Client Rights</td>
      <td>1359</td>
      <td>Community</td>
    </tr>
  </tbody>
</table>
</div>




```python
# These NaNs are clustered in the Community Health assessments
Prog_Nans['Assessment'].unique()
```




    array(['RN Community Health 2010 Form A'], dtype=object)




```python
# Closer look at missing values
for assessment in Prog_Nans['Assessment'].unique():
    NaNs = len(pa[(pa['Assessment'] == assessment) & (pa['Program Mean'].isnull())])
    Total = len(pa[(pa['Assessment'] == assessment)])
    print('Assessment: {}'.format(assessment))
    print('{0:5d} NaNs / {1:5d} Total = {2:6.2f}% Missing'.format(NaNs, Total, (NaNs/Total)*100))
```

    Assessment: RN Community Health 2010 Form A
     6885 NaNs /  6885 Total = 100.00% Missing



```python
# Initialize the mean_imputer (axis=0 means it will impute mean of column)
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
```


```python
# Iterate through each section of each assessment (for localized means)
for assessment in Prog_Nans['Assessment'].unique():
    for section in Prog_Nans[Prog_Nans['Assessment'] == assessment]['Section'].unique():
        # Copy df and remove unnecessary features
        impute_df = Prog_Nans.copy(deep=True)
        impute_df = impute_df[(impute_df['Assessment'] == assessment) & (impute_df['Section'] == section)]
        impute_df = impute_df[['Program Mean', 'Score']]
        # Fit and transform imputer
        impute_df['Program Mean'] = mean_imputer.fit_transform(impute_df)
        # Add imputed values to df using fillna
        pa['Program Mean'].fillna(value=impute_df['Program Mean'], inplace=True)
```

#### Impute Missing Values for Program Percentile
Missing values for Program Percentile should be easy to compute now that we have all of the Program Means. We'll use scipy's percentileofscore function to generate our percentiles. Again, we want to calculate percentiles for each section of each assessment. We want to apply a function over each row of a Pandas dataframe, but we need to be careful to pass both the full range of scores and a single score to each call of the percentile function.


```python
# Copy df where Program Percentile is NaN
Percentile_Nans = pa.copy(deep=True)
Percentile_Nans = Percentile_Nans[Percentile_Nans['Program Percentile'].isnull()]
```


```python
# Note that in this function we need to take the score for each observation and determine the percentile against
# the overall range of scores. So the first argument in percentileofscore is not taken row by row.
def percentile(row):
    'A simple function to calculate percentiles'
    return percentileofscore(impute_df['Score'], row['Score'])
```


```python
# Iterate through each section of each assessment (localized percentiles)
for assessment in Percentile_Nans['Assessment'].unique():
    for section in Percentile_Nans[Percentile_Nans['Assessment'] == assessment]['Section'].unique():
        # Copy df and remove unnecessary features
        impute_df = Percentile_Nans.copy(deep=True)
        impute_df = impute_df[(impute_df['Assessment'] == assessment) & (impute_df['Section'] == section)]
        impute_df = impute_df[['Program Percentile', 'Score']]
        # Apply percentile function
        impute_df['Program Percentile'] = impute_df.apply(percentile, axis=1)
        # Add imputed values
        pa['Program Percentile'].fillna(value=impute_df['Program Percentile'], inplace=True)
```


```python
# Which columns still contain NaNs?
pa.isnull().sum()
```




    Assessment                 0
    Assessment ID              0
    Booklet ID                 0
    Date Taken                 0
    National Mean              0
    National Percentile        0
    Proficiency Level      28919
    Program Mean               0
    Program Percentile         0
    Score                      0
    Section                    0
    Student ID                 0
    Category                   0
    dtype: int64



#### Impute Missing Values for Proficiency Level
Proficiency level is not based simply on a cutoff value of the raw score for a test, but is a proprietary algorithm. Proficiency levels change over time and are different for different assessments. Proficiency levels for similar performances on similar exams, however, should be the similar. For this reason, we will implement K-Nearest Neighbors to impute the missing values.


```python
# Percent of data missing for Proficiency Level
NaNs = pa['Proficiency Level'].isnull().sum()
Total = pa.shape[0]
print('Proficiency Level: {0:5d} NaNs / {1:6d} Total = {2:5.2f}% Missing'.format(NaNs, Total, (NaNs/Total)*100))
```

    Proficiency Level: 28919 NaNs / 134533 Total = 21.50% Missing



```python
# Create a subset of data that contains class labels for Proficiency level
x = pa[(pa['Proficiency Level'].notnull())]
# Drop non-numeric data and any other data which can't be used in the KNN classifier
x = x.drop(['Date Taken', 'Student ID', 'Assessment', 'Section', 'Category', 'Assessment ID', 'Booklet ID'], axis=1)
# Drop any lingering NaNs that might be in the data
x = x.dropna()
```


```python
# Separate the target variable
y = x['Proficiency Level'].as_matrix().astype(int)
x = x.drop(['Proficiency Level'], axis=1)
```


```python
#Use train test split for cross-validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```


```python
# Pre-scale the data (scale separately to avoid "cheating")
x_train_scaled = scale(x_train)
x_test_scaled = scale(x_test)
```


```python
# Use a grid search to determine the best parameter (number of neighbors)
k = list(range(3,21,3))
params = {'n_neighbors': k,
          }
```


```python
# Initialize K-Nearest Neighbors
knn_clf = KNeighborsClassifier()
```


```python
# Hyperparameter tuning using GridSearchCV
# Use f1 scoring to balance precision and recall. Use weighted because we have large class imbalance.
knn_gs = GridSearchCV(knn_clf, param_grid=params, scoring='f1_weighted', cv=5)
knn_gs.fit(x_train_scaled, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform'),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'n_neighbors': [3, 6, 9, 12, 15, 18]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring='f1_weighted', verbose=0)




```python
# Best score (F1-weighted)
knn_gs.best_score_
```




    0.99747056817624169




```python
# Best estimator
knn_gs.best_estimator_
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=3, p=2,
               weights='uniform')




```python
# Save the best estimator to a function
knn_best = knn_gs.best_estimator_
```

As we can see from the confusion matrix and classification report below, our KNN model is very accurate at predicting each of the four classifications for Proficiency level. We achieve both high precision and high recall, giving us confidence that KNN will be a good method to impute missing values. KNN should be much more accurate than simply taking the mode of the data or the most recent value.


```python
# View Confusion Matrix and Classification Report
knn_best.fit(x_train, y_train)
y_pred = knn_best.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    [[30445     3     2     7]
     [   14   377     3     0]
     [   22     5   602     2]
     [    9     0     5   189]]
                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00     30457
              1       0.98      0.96      0.97       394
              2       0.98      0.95      0.97       631
              3       0.95      0.93      0.94       203
    
    avg / total       1.00      1.00      1.00     31685
    



```python
#Impute NaNs for Proficiency Level
x_impute = pa[(pa['Proficiency Level'].isnull())]
x_impute = x_impute.drop(['Date Taken', 'Student ID', 'Assessment', 'Section', 'Category', 'Assessment ID', 'Booklet ID'], axis=1)
x_impute['Proficiency Level'] = knn_best.predict(x_impute[['National Mean','National Percentile','Program Mean','Program Percentile','Score']])
```


```python
#Add imputed values to df using fillna
pa['Proficiency Level'].fillna(x_impute['Proficiency Level'], inplace=True)
```


```python
# Which columns still contain NaNs?
pa.isnull().sum()
```




    Assessment             0
    Assessment ID          0
    Booklet ID             0
    Date Taken             0
    National Mean          0
    National Percentile    0
    Proficiency Level      0
    Program Mean           0
    Program Percentile     0
    Score                  0
    Section                0
    Student ID             0
    Category               0
    dtype: int64



### Create New Features
Intuitively, we can expect that some of our numeric features will be highly correlated with one another. We will check this for ourselves in a moment, but first let's see if combining some features might help to capture more variance in fewer features. Here, we'll calculate a simple distance to to the National Mean and Program Mean.

Again, intuitively, we might expect these features, especially Distance to the National Mean, to match very well with our overall prediction of NCLEX success. To understand why, we need to understand more about the NCLEX.


```python
# Calculate distance to National Mean
pa['Dist to National Mean'] = pa.apply(lambda x: x['Score'] - x['National Mean'],axis=1)

# Calculate distance to Program Mean
pa['Dist to Program Mean'] = pa.apply(lambda x: x['Score'] - x['Program Mean'],axis=1)
```

#### A Brief Detour into the NCLEX
The NCLEX uses Computerized Adaptive Testing (CAT), which displays questions to candidates in a way that attempts to understand each candidate's ability while using as few questions as possible. A standard exam requires all candidates to answer the same questions, while CAT attempts to always display questions that a candidate should find challenging (based on his or her ability). 

But how can the NCLEX know how challenging each question is? The answer is that questions must be tested ahead of time across a wide sample of respondents. This is the first instance where something similar to a "National Mean" plays a role in determining pass rate.

The second instance is in the actual decision rules of whether a candidate passes or fails. The passing standard is based, at least partially, on past NCLEX results, sample test-takers, and education readiness of high-school graduates interested in nursing. All of these are related to a National Mean.

### Standardize Numeric Values
For most machine learning algorithms, we want to pass standardized numeric values so that we do not give too much emphasis to features with very large maximum values or very wide ranges.

Given the type of data that we have (no negative values, and both a min and a max that have real meaning), using SciKit Learn's StandardScaler would be the wrong choice. By forcing the data to center at 0 with a standard deviation of 1, we would create negative values and lose any inherent meaning in "0" as a test score or a percentile. The MinMaxScaler, on the other hand, will preserve both of these features.


```python
# Scale numeric data using the MinMaxScaler
min_max_scaler = MinMaxScaler()
pa[['National Mean', 'National Percentile', 'Program Mean', 'Program Percentile', 'Score', 'Dist to National Mean', 'Dist to Program Mean']] = min_max_scaler.fit_transform(pa[['National Mean', 'National Percentile', 'Program Mean', 'Program Percentile', 'Score', 'Dist to National Mean', 'Dist to Program Mean']])
```


```python
# Proof that our scaling worked
pa.describe(include=[np.float])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>National Mean</th>
      <th>National Percentile</th>
      <th>Proficiency Level</th>
      <th>Program Mean</th>
      <th>Program Percentile</th>
      <th>Score</th>
      <th>Dist to National Mean</th>
      <th>Dist to Program Mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>134533.000000</td>
      <td>134533.000000</td>
      <td>134533.000000</td>
      <td>134533.000000</td>
      <td>134533.000000</td>
      <td>134533.000000</td>
      <td>134533.000000</td>
      <td>134533.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.626741</td>
      <td>0.703299</td>
      <td>0.089532</td>
      <td>0.672169</td>
      <td>0.682016</td>
      <td>0.693176</td>
      <td>0.550292</td>
      <td>0.546523</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.150885</td>
      <td>0.322488</td>
      <td>0.423430</td>
      <td>0.140552</td>
      <td>0.318684</td>
      <td>0.328152</td>
      <td>0.177796</td>
      <td>0.173748</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.544790</td>
      <td>0.428571</td>
      <td>0.000000</td>
      <td>0.597900</td>
      <td>0.417717</td>
      <td>0.500000</td>
      <td>0.459742</td>
      <td>0.464138</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.625945</td>
      <td>0.816327</td>
      <td>0.000000</td>
      <td>0.671100</td>
      <td>0.769094</td>
      <td>0.750000</td>
      <td>0.583919</td>
      <td>0.571531</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.719613</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.752000</td>
      <td>0.989961</td>
      <td>1.000000</td>
      <td>0.676498</td>
      <td>0.667329</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Recast Data
Several of our columns contain numerals, but should not be treated as either floats or integers. For example, we will never want to perform arithmetic on Student ID. For these columns, it would be much better to recast as type numpy.object.


```python
# Recast data
pa = pa.astype({'Assessment ID': np.object, 'Booklet ID': np.object, 'Student ID': np.object, 'Proficiency Level': np.int})
```


```python
# Review resulting data types
pa.dtypes
```




    Assessment                       object
    Assessment ID                    object
    Booklet ID                       object
    Date Taken               datetime64[ns]
    National Mean                   float64
    National Percentile             float64
    Proficiency Level                 int32
    Program Mean                    float64
    Program Percentile              float64
    Score                           float64
    Section                          object
    Student ID                       object
    Category                         object
    Dist to National Mean           float64
    Dist to Program Mean            float64
    dtype: object



### Was an Assessment Proctored?
The predictive testing vendor provides multiple versions of each assessment. Typically, only one of these assessments will be treated as a proctored exam requiring all students to test at the same time. The proctored exam is likely to be the most important for our purposes, as it will be the only assessment for which we can be sure of the seriousness of the testing environment. 

Students are able to take other versions of the assessment as practice. While this might provide us with important data, it would be helpful to be able to identify the difference between a proctored and a non-proctored exam.

Finally, there are also situations in which students are required to take a proctored assessment at a time other than when the rest of their cohort took the exam. This could be a re-scheduling situation, or it could be required remediation. In the former case, we would want to group the exam with the regular proctored assessment. In the latter case, a re-test in which the student improves their score might be an important indication.

Because there is no true indicator for whether or not an assessment was proctored, we will make the following assumptions:
* If a student only took one assessment in a category, it was a proctored exam.
* If a student took more than one assessment in a category, the one with the most concurrent candidates was a proctored exam.
* For all other assessments, if 3 or more students took the assessment on the same day, it was a proctored retake. The rest are non-proctored practice.


```python
# We create a subset of the full dataset to make storing values quicker
overall_df = pa.copy(deep=True)
overall_df = overall_df.drop_duplicates(subset=['Assessment ID', 'Booklet ID'])
```


```python
# We will start by making a dictionary to store counts of assessments taken on a given day.
Assessments_by_Date = {}
# Create a list of all possible assessments and iterate through that list
assessments = overall_df['Assessment'].unique().tolist()
for assessment in assessments:
    # List of all dates on which that assessment was taken
    dates = overall_df[overall_df['Assessment'] == assessment]['Date Taken'].dt.date.unique().tolist()
    for date in dates:
        # Create a tuple as the dictionary key
        key = (assessment, date)
        # The dictionary value will be the number of unique test takers
        Assessments_by_Date[key] = len(overall_df[(overall_df['Assessment'] == assessment) & (overall_df['Date Taken'].dt.date == date)]['Student ID'].unique())
```


```python
# Next we will make a dictionary to store the number of assessments each student took in each category
Assessments_by_Category = {}
# Create a list of all students and iterate through that list
students = overall_df['Student ID'].unique().tolist()
for student in students:
    # List all categories
    categories = overall_df[overall_df['Student ID'] == student]['Category'].unique().tolist()
    for category in categories:
        # Create a tuple as the dictionary key
        key = (student, category)
        # The dictionary value will be a tuple of (Assessment, Date Taken)
        Assessments_by_Category[key] = tuple(zip(overall_df[(overall_df['Student ID'] == student) & (overall_df['Category'] == category)]['Assessment'],
                                                 overall_df[(overall_df['Student ID'] == student) & (overall_df['Category'] == category)]['Date Taken'].dt.date.unique().tolist()))
```


```python
# A simple function to look up dictionary values and return a result
def Modality (row):
    # If there is only one assessment in a category, mark as the Proctored Exam
    if len(Assessments_by_Category[(row['Student ID'], row['Category'])]) == 1:
        return 'Proctored Exam'
    else:
        # Store the (Assessment, Date Taken) tuples in a category
        all_dates = Assessments_by_Category[(row['Student ID'], row['Category'])]
        # Sort tuples by Date Taken
        all_dates = sorted(all_dates, key=lambda x: x[1])
        # Create a list of the number of candidates who took each assessment on each date
        all_candidates = [Assessments_by_Date[x] for x in all_dates]
        # Find the index of the assessment with the max number of candidates
        max_candidates_index = np.argmax(all_candidates)
        # The one with the most candidates will be the Proctored Exam
        if (row['Assessment'], row['Date Taken'].date()) == all_dates[max_candidates_index]:
            return 'Proctored Exam'
        else:
            # If there were 3 or more candidates, mark as Proctored Retake
            if Assessments_by_Date[(row['Assessment'], row['Date Taken'].date())] >= 3:
                return 'Proctored Retake'
            # Else, it's practice
            else:
                return 'Non-proctored Practice'
```


```python
# Define a new dataset feature
pa['Modality'] = pa.apply(Modality, axis=1)
```


```python
# Show the value counts to see a quick distribution
pa['Modality'].value_counts()
```




    Proctored Exam            112369
    Proctored Retake           19693
    Non-proctored Practice      2471
    Name: Modality, dtype: int64



### Drop Features No Longer Needed
* We will drop both National Mean and Program Mean, as these are not student-specific values, but aggregates. As such, we can't include in any model building.
* We also drop Booklet ID and the integer version of our Assessment ID.


```python
# Drop features
pa = pa.drop(['National Mean', 'Program Mean', 'Assessment ID', 'Booklet ID'], axis=1)
```

### Reshape Data from Long to Wide Format
At this point, we will only continue working with proctored exam data and the overall scores. Future work could explore the usefulness of practice exams or remediation scores, as well as the individual section scores.


```python
# Take only the Proctored Exams
pa = pa[pa['Modality'] == 'Proctored Exam'].copy(deep=True)
```

As it turns out, we still have a small amount of duplicated data. If we take a closer look, we can see that these are true duplicates except for having a slightly different time value in the Date Taken column. This appears to be a simple inconsistency in the data, likely introduced when we dropped an unneeded variable such as "Class." We can remove one of the values with no real loss of data.


```python
# Find duplicates and keep all duplicates so that we can review
df_wide_dupe = pa.duplicated(subset=['Student ID', 'Category', 'Section'], keep=False)
```


```python
# 'True' indicates a duplicate
df_wide_dupe.value_counts()
```




    False    112241
    True        128
    dtype: int64




```python
# Drop duplicates
pa = pa.drop_duplicates(subset=['Student ID', 'Category', 'Section'], keep='first')
```


```python
# For now, keep only the overall section scores
pa = pa[pa['Section'] == 'Overall']
```


```python
# Unstack the data by Category
pa = pa.set_index(['Student ID', 'Category']).unstack('Category')
```


```python
pa.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="9" halign="left">Assessment</th>
      <th>Date Taken</th>
      <th>...</th>
      <th>Dist to Program Mean</th>
      <th colspan="9" halign="left">Modality</th>
    </tr>
    <tr>
      <th>Category</th>
      <th>Community</th>
      <th>Comprehensive</th>
      <th>Critical Thinking</th>
      <th>Fundamentals</th>
      <th>MedSurg</th>
      <th>OB</th>
      <th>Peds</th>
      <th>Pharm</th>
      <th>Psych</th>
      <th>Community</th>
      <th>...</th>
      <th>Psych</th>
      <th>Community</th>
      <th>Comprehensive</th>
      <th>Critical Thinking</th>
      <th>Fundamentals</th>
      <th>MedSurg</th>
      <th>OB</th>
      <th>Peds</th>
      <th>Pharm</th>
      <th>Psych</th>
    </tr>
    <tr>
      <th>Student ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>RN Community Health 2010 Form A</td>
      <td>RN Comprehensive Predictor 2013 Form B</td>
      <td>Critical Thinking Assessment: Entrance</td>
      <td>RN Fundamentals 2010 Form B</td>
      <td>RN Adult Medical Surgical 2010 Form B</td>
      <td>RN Maternal Newborn 2010 Form B</td>
      <td>RN Nursing Care of Children 2010 Form B</td>
      <td>RN Pharmacology 2013 Form B</td>
      <td>RN Mental Health 2010 Form B</td>
      <td>2014-02-26 11:36:00</td>
      <td>...</td>
      <td>0.553200</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RN Community Health 2010 Form A</td>
      <td>RN Comprehensive Predictor 2013 Form B</td>
      <td>Critical Thinking Assessment: Entrance</td>
      <td>RN Fundamentals 2010 Form B</td>
      <td>RN Adult Medical Surgical 2010 Form B</td>
      <td>RN Maternal Newborn 2010 Form B</td>
      <td>RN Nursing Care of Children 2010 Form B</td>
      <td>RN Pharmacology 2013 Form B</td>
      <td>RN Mental Health 2010 Form B</td>
      <td>2014-05-21 14:36:00</td>
      <td>...</td>
      <td>0.525592</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RN Community Health 2010 Form A</td>
      <td>RN Comprehensive Predictor 2013 Form B</td>
      <td>Critical Thinking Assessment: Entrance</td>
      <td>RN Fundamentals 2010 Form B</td>
      <td>RN Adult Medical Surgical 2010 Form B</td>
      <td>RN Maternal Newborn 2010 Form B</td>
      <td>RN Nursing Care of Children 2010 Form B</td>
      <td>RN Pharmacology 2013 Form B</td>
      <td>RN Mental Health 2010 Form B</td>
      <td>2014-02-26 11:37:00</td>
      <td>...</td>
      <td>0.562421</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RN Community Health 2013</td>
      <td>RN Comprehensive Predictor 2016</td>
      <td>Critical Thinking Assessment: Entrance</td>
      <td>RN Fundamentals 2013</td>
      <td>None</td>
      <td>RN Maternal Newborn 2013</td>
      <td>RN Nursing Care of Children 2013</td>
      <td>RN Pharmacology 2013</td>
      <td>RN Mental Health 2013</td>
      <td>2016-03-02 15:58:00</td>
      <td>...</td>
      <td>0.613715</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>None</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
    </tr>
    <tr>
      <th>15</th>
      <td>RN Community Health 2010 Form A</td>
      <td>RN Comprehensive Predictor 2010 Form B</td>
      <td>Critical Thinking Assessment: Entrance</td>
      <td>RN Fundamentals 2010 Form B</td>
      <td>RN Adult Medical Surgical 2010 Form B</td>
      <td>RN Maternal Newborn 2010 Form B</td>
      <td>RN Nursing Care of Children 2010 Form B</td>
      <td>RN Pharmacology 2010 Form B</td>
      <td>RN Mental Health 2010 Form B</td>
      <td>2013-05-29 13:49:00</td>
      <td>...</td>
      <td>0.599194</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
      <td>Proctored Exam</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 90 columns</p>
</div>



### Impute Missing Values for Numeric Data
Now that we have data in a wide format, we can easily see that there are some missing values. Not every student took an assessment in every category. Some of these exams were used for only a short period of time, while others have close to 100% data. We need to impute data for any features that we expect to use in model building.


```python
# A count of missing values for each assessment category
pa['Assessment'].isnull().sum()
```




    Category
    Community              0
    Comprehensive         55
    Critical Thinking      4
    Fundamentals           1
    MedSurg              109
    OB                     0
    Peds                   0
    Pharm                 36
    Psych                  1
    dtype: int64




```python
# Lists of categories and numeric columns
categories = list(pa.columns.get_level_values(1).unique())
numeric_columns = ['National Percentile', 'Program Percentile', 'Score', 'Dist to National Mean', 'Dist to Program Mean']
```

We have to jump through a few hoops to use the fillna method on a single column in a multi-index, but it is important that we do not use the same imputing strategy for numeric and categorical data. We will first impute the values for Proficiency Level using the mode for each assessment category. Notice that this case is different from before - in this case we have no data upon which to base a Proficiency Level, whereas before we had a full set of data with only the Proficiency Level missing. We use the update method because inplace does not work with .loc, and here we need to be able to access a particular subset of the data.


```python
# Iterate through categories to impute missing values for Proficiency Level
for category in categories:
    pa.update(pa.loc[:, [('Proficiency Level', category)]].fillna(pa.loc[:, [('Proficiency Level', category)]].mode().values[0][0]))
```


```python
# Ensure there are no more NaNs in Proficiency Level
pa['Proficiency Level'].isnull().sum()
```




    Category
    Community            0
    Comprehensive        0
    Critical Thinking    0
    Fundamentals         0
    MedSurg              0
    OB                   0
    Peds                 0
    Pharm                0
    Psych                0
    dtype: int64




```python
# Fill NaNs in numeric columns with the mean of the column for numeric data
pa.fillna(pa.mean(), inplace=True);
```


```python
# Check to see that there are no more missing values
pa[numeric_columns].isnull().sum()
```




                           Category         
    National Percentile    Community            0
                           Comprehensive        0
                           Critical Thinking    0
                           Fundamentals         0
                           MedSurg              0
                           OB                   0
                           Peds                 0
                           Pharm                0
                           Psych                0
    Program Percentile     Community            0
                           Comprehensive        0
                           Critical Thinking    0
                           Fundamentals         0
                           MedSurg              0
                           OB                   0
                           Peds                 0
                           Pharm                0
                           Psych                0
    Score                  Community            0
                           Comprehensive        0
                           Critical Thinking    0
                           Fundamentals         0
                           MedSurg              0
                           OB                   0
                           Peds                 0
                           Pharm                0
                           Psych                0
    Dist to National Mean  Community            0
                           Comprehensive        0
                           Critical Thinking    0
                           Fundamentals         0
                           MedSurg              0
                           OB                   0
                           Peds                 0
                           Pharm                0
                           Psych                0
    Dist to Program Mean   Community            0
                           Comprehensive        0
                           Critical Thinking    0
                           Fundamentals         0
                           MedSurg              0
                           OB                   0
                           Peds                 0
                           Pharm                0
                           Psych                0
    dtype: int64



## Reviewing for Correlations and Covariance
For each student / assessment category pairing, we have the following six data points:

* Raw Score
* National Percentile
* Program Percentile
* Proficiency Level
* Distance from National Mean
* Distance from Program Mean

As can be shown, many of these data points for a specific subject area are highly correlated to one another. This makes intuitive sense, as the raw score would be used to calculate all other data points.

In the correlation matrix below, the noticeable diagonal bands indicate the high correlation between the numeric data points for each subject.


```python
# Calculate correlation matrix
corr = pa.corr()
# Create a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up plot
f, ax = plt.subplots(figsize=(11,9))
# Generate a colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x218c87f8240>




![png](output_110_1.png)


If we take a closer look at some of the actual correlation values, we can see that the numeric data is close to being perfectly correlated with all values above 0.90. Given the high amount of correlation, it will be necessary to remove features from our dataset before we can engage in model building. Otherwise, we risk major issues due to covariance. Removing features should also help with computation time and with interpretability.


```python
# Separate out the data for Comprehensive assessments
comprehensive = pa.loc[:, pa.columns.get_level_values(1)=='Comprehensive']
# Calculate correlation matrix
corr = comprehensive.corr()

# Chaining multiple styles together
corr_styled = corr.style.\
    applymap(high_corr_red).\
    format("{:.2}")
# Call to output
corr_styled
```




<style  type="text/css" >
    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col0 {
            color:  black;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col1 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col2 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col3 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col4 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col5 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col0 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col1 {
            color:  black;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col2 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col3 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col4 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col5 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col0 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col1 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col2 {
            color:  black;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col3 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col4 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col5 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col0 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col1 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col2 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col3 {
            color:  black;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col4 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col5 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col0 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col1 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col2 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col3 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col4 {
            color:  black;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col5 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col0 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col1 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col2 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col3 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col4 {
            color:  red;
        }    #T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col5 {
            color:  black;
        }</style>  
<table id="T_7834b694_7ea9_11e7_986f_1866da0c9c62" > 
<thead>    <tr> 
        <th class="blank" ></th> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >National Percentile</th> 
        <th class="col_heading level0 col1" >Proficiency Level</th> 
        <th class="col_heading level0 col2" >Program Percentile</th> 
        <th class="col_heading level0 col3" >Score</th> 
        <th class="col_heading level0 col4" >Dist to National Mean</th> 
        <th class="col_heading level0 col5" >Dist to Program Mean</th> 
    </tr>    <tr> 
        <th class="blank" ></th> 
        <th class="index_name level1" >Category</th> 
        <th class="col_heading level1 col0" >Comprehensive</th> 
        <th class="col_heading level1 col1" >Comprehensive</th> 
        <th class="col_heading level1 col2" >Comprehensive</th> 
        <th class="col_heading level1 col3" >Comprehensive</th> 
        <th class="col_heading level1 col4" >Comprehensive</th> 
        <th class="col_heading level1 col5" >Comprehensive</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="index_name level1" >Category</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level0_row0" class="row_heading level0 row0" >National Percentile</th> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level1_row0" class="row_heading level1 row0" >Comprehensive</th> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col0" class="data row0 col0" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col1" class="data row0 col1" >0.72</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col2" class="data row0 col2" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col3" class="data row0 col3" >0.97</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col4" class="data row0 col4" >0.98</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row0_col5" class="data row0 col5" >0.98</td> 
    </tr>    <tr> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level0_row1" class="row_heading level0 row1" >Proficiency Level</th> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level1_row1" class="row_heading level1 row1" >Comprehensive</th> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col0" class="data row1 col0" >0.72</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col1" class="data row1 col1" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col2" class="data row1 col2" >0.72</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col3" class="data row1 col3" >0.72</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col4" class="data row1 col4" >0.73</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row1_col5" class="data row1 col5" >0.73</td> 
    </tr>    <tr> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level0_row2" class="row_heading level0 row2" >Program Percentile</th> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level1_row2" class="row_heading level1 row2" >Comprehensive</th> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col0" class="data row2 col0" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col1" class="data row2 col1" >0.72</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col2" class="data row2 col2" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col3" class="data row2 col3" >0.98</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col4" class="data row2 col4" >0.98</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row2_col5" class="data row2 col5" >0.98</td> 
    </tr>    <tr> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level0_row3" class="row_heading level0 row3" >Score</th> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level1_row3" class="row_heading level1 row3" >Comprehensive</th> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col0" class="data row3 col0" >0.97</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col1" class="data row3 col1" >0.72</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col2" class="data row3 col2" >0.98</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col3" class="data row3 col3" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col4" class="data row3 col4" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row3_col5" class="data row3 col5" >1.0</td> 
    </tr>    <tr> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level0_row4" class="row_heading level0 row4" >Dist to National Mean</th> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level1_row4" class="row_heading level1 row4" >Comprehensive</th> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col0" class="data row4 col0" >0.98</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col1" class="data row4 col1" >0.73</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col2" class="data row4 col2" >0.98</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col3" class="data row4 col3" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col4" class="data row4 col4" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row4_col5" class="data row4 col5" >1.0</td> 
    </tr>    <tr> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level0_row5" class="row_heading level0 row5" >Dist to Program Mean</th> 
        <th id="T_7834b694_7ea9_11e7_986f_1866da0c9c62level1_row5" class="row_heading level1 row5" >Comprehensive</th> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col0" class="data row5 col0" >0.98</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col1" class="data row5 col1" >0.73</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col2" class="data row5 col2" >0.98</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col3" class="data row5 col3" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col4" class="data row5 col4" >1.0</td> 
        <td id="T_7834b694_7ea9_11e7_986f_1866da0c9c62row5_col5" class="data row5 col5" >1.0</td> 
    </tr></tbody> 
</table> 



## Examining Feature Importance
Due to high covariance in the numeric features, we would like to see if it is possible to identify what the most important features are for model building. We will start by using random forests. 

If we cycle through one category at a time, we can see that feature importance varies by category. Distance to National Mean is a consistently strong performer, and this feature make intuitive sense when we consider that the NCLEX is a nationally standardized exam. 

Simply choosing one feature may not capture as much variance as we would want, though.


```python
# Convert numeric_columns list to numpy array for indexing purposes
feat_list = np.array(numeric_columns)
# Iterate through categories
for category in categories:
    # Grab numeric data in the particular category
    df_features = pa.loc[:, pa.columns.get_level_values(1)==category][feat_list]
    # Fit forest
    forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(df_features, nclex['Result'])
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Print importances
    print('Category: {}'.format(category))
    for f in range(df_features.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_list[indices[f]], importances[indices[f]]))
```

    Category: Community
     1) Program Percentile             0.339158
     2) Score                          0.237573
     3) National Percentile            0.188584
     4) Dist to National Mean          0.157730
     5) Dist to Program Mean           0.076955
    Category: Comprehensive
     1) Dist to National Mean          0.244719
     2) Dist to Program Mean           0.231406
     3) Program Percentile             0.207649
     4) National Percentile            0.177371
     5) Score                          0.138855
    Category: Critical Thinking
     1) Score                          0.205512
     2) National Percentile            0.202068
     3) Dist to National Mean          0.201892
     4) Program Percentile             0.201539
     5) Dist to Program Mean           0.188988
    Category: Fundamentals
     1) Dist to National Mean          0.260777
     2) Score                          0.255704
     3) Program Percentile             0.184239
     4) National Percentile            0.153670
     5) Dist to Program Mean           0.145610
    Category: MedSurg
     1) Dist to Program Mean           0.244040
     2) Score                          0.211423
     3) Dist to National Mean          0.197186
     4) National Percentile            0.176364
     5) Program Percentile             0.170988
    Category: OB
     1) Dist to National Mean          0.243809
     2) National Percentile            0.219076
     3) Dist to Program Mean           0.211143
     4) Program Percentile             0.209495
     5) Score                          0.116476
    Category: Peds
     1) Dist to Program Mean           0.241762
     2) Dist to National Mean          0.234856
     3) National Percentile            0.227147
     4) Program Percentile             0.181952
     5) Score                          0.114283
    Category: Pharm
     1) Dist to Program Mean           0.218001
     2) Dist to National Mean          0.217421
     3) Score                          0.194257
     4) Program Percentile             0.189952
     5) National Percentile            0.180369
    Category: Psych
     1) Dist to National Mean          0.279836
     2) Dist to Program Mean           0.233820
     3) National Percentile            0.189690
     4) Program Percentile             0.187121
     5) Score                          0.109533


## Reducing Dimensionality
We can use Principal Component Analysis to reduce dimensionality while maintaining an acceptable threshold of variance. As we can see in the example below, which looks at numeric data in the Comprehensive assessment, a major portion of the overall variance can be captured in a single principal component. This is not surprising given our correlations.

While Principal Component Analysis is sure to capture more variance than manually selecting one feature, we would be sacrificing interpretability.


```python
# Principal Component Analysis
pca = PCA(n_components=None)
pa_pca = pca.fit_transform(df_features)
print('Explained Variance')
for pc in range(len(pca.explained_variance_ratio_)):
    print('PC{0}: {1:5.2f}%'.format(pc+1, pca.explained_variance_ratio_[pc]*100))
# Cumulative variance
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
# Plot it
plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,len(pca.explained_variance_ratio_)+1), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.show()
```

    Explained Variance
    PC1: 99.42%
    PC2:  0.48%
    PC3:  0.08%
    PC4:  0.01%
    PC5:  0.00%



![png](output_116_1.png)


## Performing Manual Feature Selection


```python
pa_saved = pa.copy(deep=True)
```


```python
pa = pa['Dist to National Mean']
```

# Preparing Our Data for Final Modeling
At this point, we will identify a general ordering for the data, based on the curriculum of the nursing program, where each numerical value indicates the academic term in which the student would be expected to produce the data.

## Mapping Courses and Categories to a Term


```python
# Initialize dictionary to store term lists
term_map = {'courses': {}, 'categories': {}}

# Courses
term_map['courses']['fall_2011'] = {1:['301', '322', '332', '431'],
                                    2:['302', '400', '422'],
                                    3:['303', '401', '481'],
                                    4:['307', '460'],
                                    5:['440', '441', '540'],
                                    6:['442', '445', '472'],
                                    7:['426', '598']}
term_map['courses']['fall_2015'] = {1:['322', '332', '431'],
                                    2:['301', '400', '422'],
                                    3:['302', '481', '540'],
                                    4:['303', '460'],
                                    5:['307', '401', '445'],
                                    6:['440', '441'],
                                    7:['442', '472'],
                                    8:['426', '598']}

# Predictive Assessment Categories
term_map['categories']['fall_2011'] = {1:['Critical Thinking', 'Fundamentals'],
                                       2:[],
                                       3:['MedSurg'],
                                       4:['Psych'],
                                       5:['OB', 'Peds'],
                                       6:['Community'],
                                       7:['Pharm', 'Comprehensive']}
term_map['categories']['fall_2015'] = {1:['Critical Thinking'],
                                       2:['Fundamentals'],
                                       3:[],
                                       4:['MedSurg'],
                                       5:['Psych'],
                                       6:['OB', 'Peds'],
                                       7:['Community'],
                                       8:['Pharm', 'Comprehensive']}
```

## Pulling Data by Term
To cut down on the amount of variable names (and the resulting confusion), we will create a nested dictionary where the first level is the academic term. As we progress through each term, exam data will be added for each relevant term. On the second level of the dictionary, we will store:
* 'data': all feature data
* 'x_train': training split for feature data
* 'x_test': testing split for feature data
* 'y_train': training split for target data
* 'y_test': testing split for target data

As we begin model building, we will also store the results from models for easy access.


```python
# Set Desired Curriculum Pattern
curriculum_pattern = 'fall_2011'
# Determine number of terms
number_of_terms = len(term_map['courses'][curriculum_pattern].keys())
```


```python
# Initialize dictionary
df_dict = {}
for term in range(1, number_of_terms+1):
    df_dict[term] = {}   
    # Combine Grades and Predictive Assessments by term
    temp_df = grades[term_map['courses'][curriculum_pattern][term]].merge(pa[term_map['categories'][curriculum_pattern][term]], left_index=True, right_index=True)
    # In first term, assign to dictionary
    if term == 1:
        df_dict[term]['data'] = temp_df.copy(deep=True)
    # For all other terms, merge new data with previous data
    else:
        df_dict[term]['data'] = df_dict[term-1]['data'].merge(temp_df, left_index=True, right_index=True)
    # Train / Test Split
    x = df_dict[term]['data']
    y = nclex['Result']
    df_dict[term]['x_train'], df_dict[term]['x_test'], df_dict[term]['y_train'], df_dict[term]['y_test'] = train_test_split(x, y, test_size=0.3, random_state=25, stratify=y)
```


```python
# Pickle the data for future use
with open('df_dict.pickle', 'wb') as f:
    pickle.dump(df_dict, f, pickle.HIGHEST_PROTOCOL)
```
