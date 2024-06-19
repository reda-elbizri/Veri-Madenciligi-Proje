.. code:: ipython3

    # Import the neccessary modules
    import pandas as pd # manage dataframes
    import numpy as np # 
    import seaborn as sb

.. code:: ipython3

    # Read the dataset into a dataframe
    df = pd.read_csv(r'C:\Users\User\OneDrive\Desktop\Data Mining Project\titanic.csv', sep='\t', engine='python')
    df.head(10)




.. raw:: html

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
          <th>PassengerId</th>
          <th>Survived</th>
          <th>Pclass</th>
          <th>Name</th>
          <th>Sex</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Ticket</th>
          <th>Fare</th>
          <th>Cabin</th>
          <th>Embarked</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>3</td>
          <td>Braund, Mr. Owen Harris</td>
          <td>male</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>A/5 21171</td>
          <td>7.2500</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
          <td>female</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>PC 17599</td>
          <td>71.2833</td>
          <td>C85</td>
          <td>C</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>1</td>
          <td>3</td>
          <td>Heikkinen, Miss. Laina</td>
          <td>female</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>STON/O2. 3101282</td>
          <td>7.9250</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>1</td>
          <td>1</td>
          <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
          <td>female</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>113803</td>
          <td>53.1000</td>
          <td>C123</td>
          <td>S</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>0</td>
          <td>3</td>
          <td>Allen, Mr. William Henry</td>
          <td>male</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>373450</td>
          <td>8.0500</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>5</th>
          <td>6</td>
          <td>0</td>
          <td>3</td>
          <td>Moran, Mr. James</td>
          <td>male</td>
          <td>NaN</td>
          <td>0</td>
          <td>0</td>
          <td>330877</td>
          <td>8.4583</td>
          <td>NaN</td>
          <td>Q</td>
        </tr>
        <tr>
          <th>6</th>
          <td>7</td>
          <td>0</td>
          <td>1</td>
          <td>McCarthy, Mr. Timothy J</td>
          <td>male</td>
          <td>54.0</td>
          <td>0</td>
          <td>0</td>
          <td>17463</td>
          <td>51.8625</td>
          <td>E46</td>
          <td>S</td>
        </tr>
        <tr>
          <th>7</th>
          <td>8</td>
          <td>0</td>
          <td>3</td>
          <td>Palsson, Master. Gosta Leonard</td>
          <td>male</td>
          <td>2.0</td>
          <td>3</td>
          <td>1</td>
          <td>349909</td>
          <td>21.0750</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>8</th>
          <td>9</td>
          <td>1</td>
          <td>3</td>
          <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
          <td>female</td>
          <td>27.0</td>
          <td>0</td>
          <td>2</td>
          <td>347742</td>
          <td>11.1333</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>9</th>
          <td>10</td>
          <td>1</td>
          <td>2</td>
          <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
          <td>female</td>
          <td>14.0</td>
          <td>1</td>
          <td>0</td>
          <td>237736</td>
          <td>30.0708</td>
          <td>NaN</td>
          <td>C</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Drop some columns which is not relevant to the analysis (they are not numeric)
    cols_to_drop = ['Name', 'Ticket', 'Cabin']
    df = df.drop(cols_to_drop, axis=1)

.. code:: ipython3

    df.head(3)




.. raw:: html

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
          <th>PassengerId</th>
          <th>Survived</th>
          <th>Pclass</th>
          <th>Sex</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Fare</th>
          <th>Embarked</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>3</td>
          <td>male</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>7.2500</td>
          <td>S</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>female</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>71.2833</td>
          <td>C</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>1</td>
          <td>3</td>
          <td>female</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>7.9250</td>
          <td>S</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    df.info()
    sb.heatmap(df.isnull())


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 156 entries, 0 to 155
    Data columns (total 9 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  156 non-null    int64  
     1   Survived     156 non-null    int64  
     2   Pclass       156 non-null    int64  
     3   Sex          156 non-null    object 
     4   Age          126 non-null    float64
     5   SibSp        156 non-null    int64  
     6   Parch        156 non-null    int64  
     7   Fare         156 non-null    float64
     8   Embarked     155 non-null    object 
    dtypes: float64(2), int64(5), object(2)
    memory usage: 11.1+ KB
    



.. parsed-literal::

    <Axes: >




.. image:: output_4_2.png


.. code:: ipython3

    # To replace missing values with interpolated values, for example Age
    df['Age'] = df['Age'].interpolate()

.. code:: ipython3

    sb.heatmap(df.isnull())




.. parsed-literal::

    <Axes: >




.. image:: output_6_1.png


.. code:: ipython3

    # Drop all rows with missin data
    df = df.dropna() # drop not avaialable

.. code:: ipython3

    df.head()




.. raw:: html

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
          <th>PassengerId</th>
          <th>Survived</th>
          <th>Pclass</th>
          <th>Sex</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Fare</th>
          <th>Embarked</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>3</td>
          <td>male</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>7.2500</td>
          <td>S</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>female</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>71.2833</td>
          <td>C</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>1</td>
          <td>3</td>
          <td>female</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>7.9250</td>
          <td>S</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>1</td>
          <td>1</td>
          <td>female</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>53.1000</td>
          <td>S</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>0</td>
          <td>3</td>
          <td>male</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>8.0500</td>
          <td>S</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # First, create dummy columns from the Embarked and Sex columns
    EmbarkedColumnDummy = pd.get_dummies(df['Embarked'])
    SexColumnDummy = pd.get_dummies(df['Sex'])

.. code:: ipython3

    df = pd.concat((df, EmbarkedColumnDummy, SexColumnDummy), axis=1)

.. code:: ipython3

    df.head()




.. raw:: html

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
          <th>PassengerId</th>
          <th>Survived</th>
          <th>Pclass</th>
          <th>Sex</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Fare</th>
          <th>Embarked</th>
          <th>C</th>
          <th>Q</th>
          <th>S</th>
          <th>female</th>
          <th>male</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>3</td>
          <td>male</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>7.2500</td>
          <td>S</td>
          <td>False</td>
          <td>False</td>
          <td>True</td>
          <td>False</td>
          <td>True</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>female</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>71.2833</td>
          <td>C</td>
          <td>True</td>
          <td>False</td>
          <td>False</td>
          <td>True</td>
          <td>False</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>1</td>
          <td>3</td>
          <td>female</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>7.9250</td>
          <td>S</td>
          <td>False</td>
          <td>False</td>
          <td>True</td>
          <td>True</td>
          <td>False</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>1</td>
          <td>1</td>
          <td>female</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>53.1000</td>
          <td>S</td>
          <td>False</td>
          <td>False</td>
          <td>True</td>
          <td>True</td>
          <td>False</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>0</td>
          <td>3</td>
          <td>male</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>8.0500</td>
          <td>S</td>
          <td>False</td>
          <td>False</td>
          <td>True</td>
          <td>False</td>
          <td>True</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Drop the redundant columns thus converted
    df = df.drop(['Sex','Embarked'],axis=1)

.. code:: ipython3

    # Seperate the dataframe into X and y data
    X = df.values
    y = df['Survived'].values
    
    # Delete the Survived column from X
    X = np.delete(X,1,axis=1)

.. code:: ipython3

    # Split the dataset into 70% Training and 30% Test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

.. code:: ipython3

    # Using simple Decision Tree classifier
    from sklearn import tree
    dt_clf = tree.DecisionTreeClassifier(max_depth=5)
    dt_clf.fit(X_train, y_train)
    dt_clf.score(X_test, y_test)




.. parsed-literal::

    0.7872340425531915



.. code:: ipython3

    # Using Naive Bayes classifier
    from sklearn.naive_bayes import GaussianNB
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    nb_clf.score(X_test, y_test)




.. parsed-literal::

    0.7659574468085106



.. code:: ipython3

    # Using KNN classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train, y_train)
    knn_clf.score(X_test, y_test)




.. parsed-literal::

    0.574468085106383



.. code:: ipython3

    # Using KNN classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train, y_train)
    knn_clf.score(X_test, y_test)




.. parsed-literal::

    0.574468085106383



.. code:: ipython3

    # Using KNN classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train, y_train)
    knn_clf.score(X_test, y_test)




.. parsed-literal::

    0.574468085106383


