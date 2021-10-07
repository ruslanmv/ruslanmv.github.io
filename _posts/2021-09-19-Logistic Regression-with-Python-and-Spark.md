---
title: "Logistic Regression with Python and Spark"
excerpt: "Logistic Regression with Python and Spark"

header:
  image: "../assets/images/posts/2021-09-19-Logistic%20Regression-with-Python-and-Spark/pic4.jpg"
  teaser: "../assets/images/posts/2021-09-19-Logistic%20Regression-with-Python-and-Spark/pic4.jpg"
  caption: "“Is artificial intelligence less than our intelligence?” —Spike Jonze"
  
---

### Customer churn with Logistic Regression

A marketing agency  is concerned about the number of customers stop buying their service. They need to understand who is leaving. Imagine that you are an analyst at this company and you have to find out the customers most at risk to churn an account manager.    

<h4>Table of contents</h4>
<div class="alert alert-block alert-info" style="margin-top: 20px">
    <ol>
        <li><a href="#ref1">Logistic Regression with Python</a></li>
        <li><a href="#ref2">Logistic Regression with Pyspark</a></li>
    </ol>
</div>
<br>

The installation of **Python** and **Pyspark**  and the introduction of the Logistic Regression is given [here.](./Machine-Learning-with-Python-and-Spark)

<a id="ref1"></a>

### Logistic Regression with Python

Lets first import required libraries:


```python
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
%matplotlib inline 
import matplotlib.pyplot as plt
```



<h3 id="understanding_data">Understanding the Data</h3>


```python
churn_df = pd.read_csv("customer_churn.csv")
churn_df.head(1)
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
      <th>Names</th>
      <th>Age</th>
      <th>Total_Purchase</th>
      <th>Account_Manager</th>
      <th>Years</th>
      <th>Num_Sites</th>
      <th>Onboard_date</th>
      <th>Location</th>
      <th>Company</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cameron Williams</td>
      <td>42.0</td>
      <td>11066.8</td>
      <td>0</td>
      <td>7.22</td>
      <td>8.0</td>
      <td>2013-08-30 07:00:40</td>
      <td>10265 Elizabeth Mission Barkerburgh, AK 89518</td>
      <td>Harvey LLC</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>



The data is saved as customer_churn.csv. Here are the fields and their definitions:

    Name : Name of the latest contact at Company
    Age: Customer Age
    Total_Purchase: Total Ads Purchased
    Account_Manager: Binary 0=No manager, 1= Account manager assigned
    Years: Totaly Years as a customer
    Num_sites: Number of websites that use the service.
    Onboard_date: Date that the name of the latest contact was onboarded
    Location: Client HQ Address
    Company: Name of Client Company

Lets select some features for the modeling. Also we change the target data type to be integer, as it is a requirement  by the skitlearn algorithm:


```python
churn_df.columns
```




    Index(['Names', 'Age', 'Total_Purchase', 'Account_Manager', 'Years',
           'Num_Sites', 'Onboard_date', 'Location', 'Company', 'Churn'],
          dtype='object')




```python
inputCols=['Age',
 'Total_Purchase',
 'Account_Manager',
 'Years',
 'Num_Sites','Churn']
```


```python
churn_df = churn_df[inputCols]
```


```python
churn_df['Churn'] = churn_df['Churn'].astype('int')
churn_df.head()
```







<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Total_Purchase</th>
      <th>Account_Manager</th>
      <th>Years</th>
      <th>Num_Sites</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42.0</td>
      <td>11066.80</td>
      <td>0</td>
      <td>7.22</td>
      <td>8.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41.0</td>
      <td>11916.22</td>
      <td>0</td>
      <td>6.50</td>
      <td>11.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>12884.75</td>
      <td>0</td>
      <td>6.67</td>
      <td>12.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42.0</td>
      <td>8010.76</td>
      <td>0</td>
      <td>6.71</td>
      <td>10.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37.0</td>
      <td>9191.58</td>
      <td>0</td>
      <td>5.56</td>
      <td>9.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>



Lets define X, and y for our dataset:


```python
X = np.asarray(churn_df[['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']])X[0:5]
```




    array([[4.200000e+01, 1.106680e+04, 0.000000e+00, 7.220000e+00,        8.000000e+00],       [4.100000e+01, 1.191622e+04, 0.000000e+00, 6.500000e+00,        1.100000e+01],       [3.800000e+01, 1.288475e+04, 0.000000e+00, 6.670000e+00,        1.200000e+01],       [4.200000e+01, 8.010760e+03, 0.000000e+00, 6.710000e+00,        1.000000e+01],       [3.700000e+01, 9.191580e+03, 0.000000e+00, 5.560000e+00,        9.000000e+00]])




```python
y = np.asarray(churn_df['Churn'])y [0:5]
```




    array([1, 1, 1, 1, 1])



Also, we normalize the dataset:


```python
from sklearn import preprocessingX = preprocessing.StandardScaler().fit(X).transform(X)X[0:5]
```




    array([[ 0.0299361 ,  0.41705373, -0.96290958,  1.52844634, -0.33323478],       [-0.13335172,  0.76990459, -0.96290958,  0.96318219,  1.36758544],       [-0.6232152 ,  1.172234  , -0.96290958,  1.09664734,  1.93452551],       [ 0.0299361 , -0.85243173, -0.96290958,  1.1280509 ,  0.80064537],       [-0.78650303, -0.36191661, -0.96290958,  0.22519844,  0.2337053 ]])




```python
## Train/Test dataset
```

We split our dataset into train and test set:


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
```

    Train set: (720, 5) (720,)Test set: (180, 5) (180,)


Lets build our model using __LogisticRegression__ from Scikit-learn package. The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting problem in machine learning models.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR
```




    LogisticRegression(C=0.01, solver='liblinear')



Now we can predict using our test set:


```python
yhat = LR.predict(X_test)
yhat
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,       0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,       0, 0, 0, 0])



__predict_proba__  returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):


```python
yhat_prob = LR.predict_proba(X_test)
yhat_prob
```




    array([[0.55115319, 0.44884681],       [0.50775355, 0.49224645],       [0.78905855, 0.21094145],       [0.82586384, 0.17413616],       [0.75824089, 0.24175911],       [0.75057247, 0.24942753],       [0.77136927, 0.22863073],       [0.82929872, 0.17070128],       [0.69618372, 0.30381628],       [0.85843925, 0.14156075],       [0.7115565 , 0.2884435 ],       [0.84799908, 0.15200092],       [0.43464548, 0.56535452],       [0.79007774, 0.20992226],       [0.56321142, 0.43678858],       [0.87795661, 0.12204339],       [0.78809214, 0.21190786],       [0.44728951, 0.55271049],       [0.67567028, 0.32432972],       [0.77654205, 0.22345795],       [0.54008405, 0.45991595],       [0.82353426, 0.17646574],       [0.89258452, 0.10741548],       [0.61936631, 0.38063369],       [0.66148835, 0.33851165],       [0.79621017, 0.20378983],       [0.82950855, 0.17049145],       [0.80546686, 0.19453314],       [0.81026516, 0.18973484],       [0.78321259, 0.21678741],       [0.51444245, 0.48555755],       [0.66593741, 0.33406259],       [0.65339164, 0.34660836],       [0.68943534, 0.31056466],       [0.47926904, 0.52073096],       [0.83670879, 0.16329121],       [0.78999775, 0.21000225],       [0.7047045 , 0.2952955 ],       [0.83574241, 0.16425759],       [0.83025175, 0.16974825],       [0.56893188, 0.43106812],       [0.84985654, 0.15014346],       [0.77395151, 0.22604849],       [0.90676545, 0.09323455],       [0.7787319 , 0.2212681 ],       [0.68414442, 0.31585558],       [0.77666739, 0.22333261],       [0.72131692, 0.27868308],       [0.54245497, 0.45754503],       [0.81406035, 0.18593965],       [0.52157222, 0.47842778],       [0.73727144, 0.26272856],       [0.47840111, 0.52159889],       [0.88380047, 0.11619953],       [0.89172479, 0.10827521],       [0.38354502, 0.61645498],       [0.73814911, 0.26185089],       [0.78056001, 0.21943999],       [0.63239947, 0.36760053],       [0.53121787, 0.46878213],       [0.54814705, 0.45185295],       [0.52312722, 0.47687278],       [0.76739279, 0.23260721],       [0.80332797, 0.19667203],       [0.79036892, 0.20963108],       [0.67214005, 0.32785995],       [0.48671992, 0.51328008],       [0.76484354, 0.23515646],       [0.8155299 , 0.1844701 ],       [0.67160484, 0.32839516],       [0.82541353, 0.17458647],       [0.68755774, 0.31244226],       [0.85531287, 0.14468713],       [0.56196058, 0.43803942],       [0.85455398, 0.14544602],       [0.77256622, 0.22743378],       [0.74017055, 0.25982945],       [0.84707244, 0.15292756],       [0.71965029, 0.28034971],       [0.63627897, 0.36372103],       [0.8486821 , 0.1513179 ],       [0.4956892 , 0.5043108 ],       [0.58358899, 0.41641101],       [0.52703884, 0.47296116],       [0.80328002, 0.19671998],       [0.77292201, 0.22707799],       [0.70175886, 0.29824114],       [0.42400746, 0.57599254],       [0.80819026, 0.19180974],       [0.69313651, 0.30686349],       [0.89661622, 0.10338378],       [0.65098842, 0.34901158],       [0.81565175, 0.18434825],       [0.61950808, 0.38049192],       [0.71250051, 0.28749949],       [0.73726468, 0.26273532],       [0.71850564, 0.28149436],       [0.82503922, 0.17496078],       [0.73086073, 0.26913927],       [0.79658028, 0.20341972],       [0.68822995, 0.31177005],       [0.79657285, 0.20342715],       [0.64092767, 0.35907233],       [0.59180605, 0.40819395],       [0.52464209, 0.47535791],       [0.84599247, 0.15400753],       [0.52275865, 0.47724135],       [0.54657704, 0.45342296],       [0.71247344, 0.28752656],       [0.77436313, 0.22563687],       [0.70850185, 0.29149815],       [0.87238837, 0.12761163],       [0.5295663 , 0.4704337 ],       [0.83494035, 0.16505965],       [0.81620117, 0.18379883],       [0.73823045, 0.26176955],       [0.84690546, 0.15309454],       [0.47586463, 0.52413537],       [0.78822224, 0.21177776],       [0.87548492, 0.12451508],       [0.65573165, 0.34426835],       [0.77089876, 0.22910124],       [0.64617429, 0.35382571],       [0.6907823 , 0.3092177 ],       [0.69681897, 0.30318103],       [0.45951092, 0.54048908],       [0.62067857, 0.37932143],       [0.74160583, 0.25839417],       [0.53013499, 0.46986501],       [0.28850165, 0.71149835],       [0.49512568, 0.50487432],       [0.72593514, 0.27406486],       [0.80394446, 0.19605554],       [0.65956213, 0.34043787],       [0.48001116, 0.51998884],       [0.69646787, 0.30353213],       [0.81443981, 0.18556019],       [0.4580436 , 0.5419564 ],       [0.69271673, 0.30728327],       [0.75114546, 0.24885454],       [0.56717161, 0.43282839],       [0.7252306 , 0.2747694 ],       [0.59836389, 0.40163611],       [0.78848626, 0.21151374],       [0.69708225, 0.30291775],       [0.50678377, 0.49321623],       [0.82565035, 0.17434965],       [0.79616745, 0.20383255],       [0.55807373, 0.44192627],       [0.80805658, 0.19194342],       [0.74371419, 0.25628581],       [0.74945198, 0.25054802],       [0.73397128, 0.26602872],       [0.77815725, 0.22184275],       [0.90242618, 0.09757382],       [0.38981756, 0.61018244],       [0.7513245 , 0.2486755 ],       [0.7519866 , 0.2480134 ],       [0.79901699, 0.20098301],       [0.84877662, 0.15122338],       [0.49459397, 0.50540603],       [0.82638406, 0.17361594],       [0.4522431 , 0.5477569 ],       [0.77862228, 0.22137772],       [0.51452033, 0.48547967],       [0.85666661, 0.14333339],       [0.59774733, 0.40225267],       [0.66366638, 0.33633362],       [0.79063591, 0.20936409],       [0.34793486, 0.65206514],       [0.66749101, 0.33250899],       [0.57026695, 0.42973305],       [0.43376394, 0.56623606],       [0.47592774, 0.52407226],       [0.65214083, 0.34785917],       [0.82552994, 0.17447006],       [0.61757605, 0.38242395],       [0.56300855, 0.43699145],       [0.81953379, 0.18046621],       [0.72932443, 0.27067557]])



### confusion matrix

Another way of looking at accuracy of classifier is to look at __confusion matrix__.


```python
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))
```

    [[ 14  16] [  6 144]]



```python
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
```

    Confusion matrix, without normalization[[ 14  16] [  6 144]]




![png](../assets/images/posts/2021-09-19-Logistic%20Regression-with-Python-and-Spark/Logistic-Regression_30_1.png)
    



```python
print (classification_report(y_test, yhat))
```

                  precision    recall  f1-score   support           0       0.90      0.96      0.93       150           1       0.70      0.47      0.56        30    accuracy                           0.88       180   macro avg       0.80      0.71      0.74       180weighted avg       0.87      0.88      0.87       180


​    

The average accuracy for this classifier is the average of the F1-score for both labels, which is 0.87 in our case.

Based on the count of each section, we can calculate precision and recall of each label:


- __Precision__ is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)

- __Recall__ is true positive rate. It is defined as: Recall =  TP / (TP + FN)

    

we can calculate precision and recall of each class.

__F1 score:__
Now we are in the position to calculate the F1 scores for each label based on the precision and recall of that label. 

The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.



<a id="ref2"></a>

### Logistic Regression with Pyspark


```python
import findspark
```


```python
findspark.init()
```


```python
from pyspark.sql import SparkSession
```


```python
spark = SparkSession.builder.appName('logregconsult').getOrCreate()
```


```python
data = spark.read.csv('customer_churn.csv',inferSchema=True,                     header=True)
```


```python
data.printSchema()
```

    root |-- Names: string (nullable = true) |-- Age: double (nullable = true) |-- Total_Purchase: double (nullable = true) |-- Account_Manager: integer (nullable = true) |-- Years: double (nullable = true) |-- Num_Sites: double (nullable = true) |-- Onboard_date: string (nullable = true) |-- Location: string (nullable = true) |-- Company: string (nullable = true) |-- Churn: integer (nullable = true)


​    

### Check out the data


```python
data.describe().show(1)
```

    +-------+-----+---+--------------+---------------+-----+---------+------------+--------+-------+-----+|summary|Names|Age|Total_Purchase|Account_Manager|Years|Num_Sites|Onboard_date|Location|Company|Churn|+-------+-----+---+--------------+---------------+-----+---------+------------+--------+-------+-----+|  count|  900|900|           900|            900|  900|      900|         900|     900|    900|  900|+-------+-----+---+--------------+---------------+-----+---------+------------+--------+-------+-----+only showing top 1 row


​    


```python

```

### Format for MLlib

We'll ues the numerical columns. We'll include Account Manager because its easy enough, but keep in mind it probably won't be any sort of a signal because the agency mentioned its randomly assigned!


```python
from pyspark.ml.feature import VectorAssembler
```


```python
assembler = VectorAssembler(inputCols=['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites'],outputCol='features')
```


```python
output = assembler.transform(data)
```


```python
final_data = output.select('features','churn')
```

### Test Train Split


```python
train_churn,test_churn = final_data.randomSplit([0.7,0.3])
```

### Fit the model


```python
from pyspark.ml.classification import LogisticRegression
```


```python
lr_churn = LogisticRegression(labelCol='churn')
```


```python
fitted_churn_model = lr_churn.fit(train_churn)
```


```python
training_sum = fitted_churn_model.summary
```


```python
training_sum.predictions.describe().show()
```

    +-------+-------------------+-------------------+|summary|              churn|         prediction|+-------+-------------------+-------------------+|  count|                613|                613||   mean|0.17781402936378465|0.12887438825448613|| stddev|  0.382668372099746|   0.33533449140226||    min|                0.0|                0.0||    max|                1.0|                1.0|+-------+-------------------+-------------------+


​    

### Evaluate results

Let's evaluate the results on the data set we were given (using the test data)


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```


```python
pred_and_labels = fitted_churn_model.evaluate(test_churn)
```


```python
pred_and_labels.predictions.show()
```

    +--------------------+-----+--------------------+--------------------+----------+|            features|churn|       rawPrediction|         probability|prediction|+--------------------+-----+--------------------+--------------------+----------+|[22.0,11254.38,1....|    0|[4.05430795643892...|[0.98294832241403...|       0.0||[26.0,8939.61,0.0...|    0|[5.77405037803505...|[0.99690247758406...|       0.0||[28.0,8670.98,0.0...|    0|[7.23826268284963...|[0.99928195691507...|       0.0||[28.0,9090.43,1.0...|    0|[0.94821948475399...|[0.72075696108881...|       0.0||[29.0,9617.59,0.0...|    0|[3.91324626692407...|[0.98041565826455...|       0.0||[29.0,10203.18,1....|    0|[3.26598082266084...|[0.96324313418667...|       0.0||[29.0,11274.46,1....|    0|[4.17445848712145...|[0.98484954751144...|       0.0||[29.0,13255.05,1....|    0|[3.93587227636939...|[0.98084540599653...|       0.0||[30.0,8403.78,1.0...|    0|[5.44208726208492...|[0.99568823673280...|       0.0||[30.0,8677.28,1.0...|    0|[3.43610521361442...|[0.96881405579738...|       0.0||[30.0,10183.98,1....|    0|[2.48028976858033...|[0.92274845618277...|       0.0||[31.0,5304.6,0.0,...|    0|[2.67870443526006...|[0.93575828504198...|       0.0||[31.0,5387.75,0.0...|    0|[1.70594375589089...|[0.84630943026440...|       0.0||[31.0,10058.87,1....|    0|[3.92946303722257...|[0.98072461932954...|       0.0||[31.0,10182.6,1.0...|    0|[4.50159146584151...|[0.98903033715124...|       0.0||[31.0,11297.57,1....|    1|[0.60412216064385...|[0.64659882437760...|       0.0||[31.0,12264.68,1....|    0|[3.22109124776813...|[0.96162030913584...|       0.0||[32.0,5756.12,0.0...|    0|[3.45850939238844...|[0.96948389815194...|       0.0||[32.0,6367.22,1.0...|    0|[2.58844173988655...|[0.93011399503796...|       0.0||[32.0,7896.65,0.0...|    0|[2.74301964391580...|[0.93951791284489...|       0.0|+--------------------+-----+--------------------+--------------------+----------+only showing top 20 rows


​    

### Using AUC


```python
churn_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',                                           labelCol='churn')
```


```python
auc = churn_eval.evaluate(pred_and_labels.predictions)
```


```python
auc
```




    0.7926829268292683



### Predict on brand new unlabeled data



```python
final_lr_model = lr_churn.fit(final_data)
```


```python
new_customers = spark.read.csv('new_customers.csv',inferSchema=True,                              header=True)
```


```python
new_customers.printSchema()
```

    root |-- Names: string (nullable = true) |-- Age: double (nullable = true) |-- Total_Purchase: double (nullable = true) |-- Account_Manager: integer (nullable = true) |-- Years: double (nullable = true) |-- Num_Sites: double (nullable = true) |-- Onboard_date: string (nullable = true) |-- Location: string (nullable = true) |-- Company: string (nullable = true)


​    


```python
test_new_customers = assembler.transform(new_customers)
```


```python
test_new_customers.printSchema()
```

    root |-- Names: string (nullable = true) |-- Age: double (nullable = true) |-- Total_Purchase: double (nullable = true) |-- Account_Manager: integer (nullable = true) |-- Years: double (nullable = true) |-- Num_Sites: double (nullable = true) |-- Onboard_date: string (nullable = true) |-- Location: string (nullable = true) |-- Company: string (nullable = true) |-- features: vector (nullable = true)


​    


```python
final_results = final_lr_model.transform(test_new_customers)
```


```python
final_results.select('Company','prediction').show()
```

    +----------------+----------+|         Company|prediction|+----------------+----------+|        King Ltd|       0.0||   Cannon-Benson|       1.0||Barron-Robertson|       1.0||   Sexton-Golden|       1.0||        Wood LLC|       0.0||   Parks-Robbins|       1.0|+----------------+----------+


​    

Now we know that we should assign Acocunt Managers to Cannon-Benson,Barron-Robertson,Sexton-GOlden, and Parks-Robbins!

You can download the notebook [here](https://github.com/ruslanmv/Machine-Learning-with-Python-and-Spark/blob/master/Logistic-Regression/Logistic-Regression.ipynb)

**Congratulations!** We have practiced Logistic Regression with Python and Spark.

