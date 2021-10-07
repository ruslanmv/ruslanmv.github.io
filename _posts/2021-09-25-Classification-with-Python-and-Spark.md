---
title: "Decision Tree classification with Python and Spark."
excerpt: "Decision Tree classification"

header:
  image: "../assets/images/posts/2021-09-25-Classification-with-Python-and-Spark/tree.jpg"
  teaser: "../assets/images/posts/2021-09-25-Classification-with-Python-and-Spark/tree.jpg"
  caption: "It's going to be interesting to see how society deals with artificial intelligence, but it will definitely be cool. —Colin Angle"
  
---



We will use Decision Tree classification algorithm to build a model from historical data of patients, and their response to different medications.

Then we will use the trained decision tree to predict the class of a unknown patient, or to find a proper drug for a new patient.

We have data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and y. 


We want to build a model to find out which drug might be appropriate for a future patient with the same illness.


<h4>Table of contents</h4>
<div class="alert alert-block alert-info" style="margin-top: 20px">
    <ol>
        <li><a href="#ref1">Decision Tree with Python</a></li>
        <li><a href="#ref2">Decision Tree with Pyspark</a></li>
    </ol>
</div>
<br>

The installation of **Python** and **Pyspark**  and the introduction of the Decision Tree classification is given [here.](./Machine-Learning-with-Python-and-Spark)



<a id="ref1"></a>

### Decision Tree classification  with Python


```python
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
```


```python
my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]
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
      <th>Age</th>
      <th>Sex</th>
      <th>BP</th>
      <th>Cholesterol</th>
      <th>Na_to_K</th>
      <th>Drug</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>F</td>
      <td>HIGH</td>
      <td>HIGH</td>
      <td>25.355</td>
      <td>drugY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47</td>
      <td>M</td>
      <td>LOW</td>
      <td>HIGH</td>
      <td>13.093</td>
      <td>drugC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47</td>
      <td>M</td>
      <td>LOW</td>
      <td>HIGH</td>
      <td>10.114</td>
      <td>drugC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>F</td>
      <td>NORMAL</td>
      <td>HIGH</td>
      <td>7.798</td>
      <td>drugX</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61</td>
      <td>F</td>
      <td>LOW</td>
      <td>HIGH</td>
      <td>18.043</td>
      <td>drugY</td>
    </tr>
  </tbody>
</table>

</div>



Using **drug200.csv** data read by pandas, declare the following variables: <br>

<ul>
    <li> <b> X </b> as the <b> Feature Matrix </b> (data of my_data) </li>
    <li> <b> y </b> as the <b> response vector (target) </b> </li>
</ul>



```python
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]
```




    array([[23, 'F', 'HIGH', 'HIGH', 25.355],
           [47, 'M', 'LOW', 'HIGH', 13.093],
           [47, 'M', 'LOW', 'HIGH', 10.113999999999999],
           [28, 'F', 'NORMAL', 'HIGH', 7.797999999999999],
           [61, 'F', 'LOW', 'HIGH', 18.043]], dtype=object)




```python
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

```




    array([[23, 0, 0, 0, 25.355],
           [47, 1, 1, 0, 13.093],
           [47, 1, 1, 0, 10.113999999999999],
           [28, 0, 2, 0, 7.797999999999999],
           [61, 0, 1, 0, 18.043]], dtype=object)




```python
y = my_data["Drug"]
y[0:5]
```




    0    drugY1    drugC2    drugC3    drugX4    drugYName: Drug, dtype: object



    We will be using train/test split  on our decision tree Let's import train_test_split from sklearn.cross_validation


```python
from sklearn.model_selection import train_test_split
```


```python
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
```

 We will first create an instance of the DecisionTreeClassifier called drugTree.
    Inside of the classifier, specify  criterion="entropy" so we can see the information gain of each node.


```python
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
```




    DecisionTreeClassifier(criterion='entropy', max_depth=4)




```python
drugTree.fit(X_trainset,y_trainset)
```




    DecisionTreeClassifier(criterion='entropy', max_depth=4)





<div id="prediction">
    <h2>Prediction</h2>
    Let's make some <b>predictions</b> on the testing dataset and store it into a variable called <b>predTree</b>.
</div>



```python
predTree = drugTree.predict(X_testset)
```


```python
print (predTree [0:5])
print (y_testset [0:5])
```

    ['drugY' 'drugX' 'drugX' 'drugX' 'drugX']40     drugY51     drugX139    drugX197    drugX170    drugXName: Drug, dtype: object


<hr>


<div id="evaluation">
    <h2>Evaluation</h2>
    Next, let's import <b>metrics</b> from sklearn and check the accuracy of our model.
</div>



```python
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
```

    DecisionTrees's Accuracy:  0.9833333333333333


__Accuracy classification score__ computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.  

In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

<a id="ref2"></a>

### Decision Tree classification  with Pyspark


```python
import findspark
```


```python
findspark.init()
```



```
#Tree methods Example
from pyspark.sql import SparkSessionspark = SparkSession.builder.appName('treecode').getOrCreate()
```

### Understanding the Data

```python
data = spark.read.csv('drug200.csv',inferSchema=True,header=True)
```




```python
data.printSchema()
```

    root |-- Age: integer (nullable = true) |-- Sex: string (nullable = true) |-- BP: string (nullable = true) |-- Cholesterol: string (nullable = true) |-- Na_to_K: double (nullable = true) |-- Drug: string (nullable = true)


​    

The feature sets of this dataset are Age, Sex, Blood Pressure, and Cholesterol of patients, and the target is the drug that each patient responded to.


It is a sample of binary classifier, we will use the training part of the dataset  to build a decision tree, and then use it to predict the class of a unknown patient, or to prescribe it to a new patient.


```python
data.head()
```




    Row(Age=23, Sex='F', BP='HIGH', Cholesterol='HIGH', Na_to_K=25.355, Drug='drugY')



### Spark Formatting of Data


```python
# A few things we need to do before Spark can accept the data!
# It needs to be in the form of two columns
# ("label","features")
# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
```


```python
data.columns
```




    ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']




```python
data.show()
```

    +---+---+------+-----------+-------+-----+
    |Age|Sex|    BP|Cholesterol|Na_to_K| Drug|
    +---+---+------+-----------+-------+-----+
    | 23|  F|  HIGH|       HIGH| 25.355|drugY|
    | 47|  M|   LOW|       HIGH| 13.093|drugC|
    | 47|  M|   LOW|       HIGH| 10.114|drugC|
    | 28|  F|NORMAL|       HIGH|  7.798|drugX|
    | 61|  F|   LOW|       HIGH| 18.043|drugY|
    | 22|  F|NORMAL|       HIGH|  8.607|drugX|
    | 49|  F|NORMAL|       HIGH| 16.275|drugY|
    | 41|  M|   LOW|       HIGH| 11.037|drugC|
    | 60|  M|NORMAL|       HIGH| 15.171|drugY|
    | 43|  M|   LOW|     NORMAL| 19.368|drugY|
    | 47|  F|   LOW|       HIGH| 11.767|drugC|
    | 34|  F|  HIGH|     NORMAL| 19.199|drugY|
    | 43|  M|   LOW|       HIGH| 15.376|drugY|
    | 74|  F|   LOW|       HIGH| 20.942|drugY|
    | 50|  F|NORMAL|       HIGH| 12.703|drugX|
    | 16|  F|  HIGH|     NORMAL| 15.516|drugY|
    | 69|  M|   LOW|     NORMAL| 11.455|drugX|
    | 43|  M|  HIGH|       HIGH| 13.972|drugA|
    | 23|  M|   LOW|       HIGH|  7.298|drugC|
    | 32|  F|  HIGH|     NORMAL| 25.974|drugY|
    +---+---+------+-----------+-------+-----+
    only showing top 20 rows


​    


```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer
```


```python
data.show()
```

    +---+---+------+-----------+-------+-----+
    |Age|Sex|    BP|Cholesterol|Na_to_K| Drug|
    +---+---+------+-----------+-------+-----+
    | 23|  F|  HIGH|       HIGH| 25.355|drugY|
    | 47|  M|   LOW|       HIGH| 13.093|drugC|
    | 47|  M|   LOW|       HIGH| 10.114|drugC|
    | 28|  F|NORMAL|       HIGH|  7.798|drugX|
    | 61|  F|   LOW|       HIGH| 18.043|drugY|
    | 22|  F|NORMAL|       HIGH|  8.607|drugX|
    | 49|  F|NORMAL|       HIGH| 16.275|drugY|
    | 41|  M|   LOW|       HIGH| 11.037|drugC|
    | 60|  M|NORMAL|       HIGH| 15.171|drugY|
    | 43|  M|   LOW|     NORMAL| 19.368|drugY|
    | 47|  F|   LOW|       HIGH| 11.767|drugC|
    | 34|  F|  HIGH|     NORMAL| 19.199|drugY|
    | 43|  M|   LOW|       HIGH| 15.376|drugY|
    | 74|  F|   LOW|       HIGH| 20.942|drugY|
    | 50|  F|NORMAL|       HIGH| 12.703|drugX|
    | 16|  F|  HIGH|     NORMAL| 15.516|drugY|
    | 69|  M|   LOW|     NORMAL| 11.455|drugX|
    | 43|  M|  HIGH|       HIGH| 13.972|drugA|
    | 23|  M|   LOW|       HIGH|  7.298|drugC|
    | 32|  F|  HIGH|     NORMAL| 25.974|drugY|
    +---+---+------+-----------+-------+-----+
    only showing top 20 rows


​    

As you may figure out, some features in this dataset are categorical such as __Sex__ or __BP__. 

Decision Trees do not handle categorical variables. But still we can convert these features to numerical values. 


```python
data.columns
```




    ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']



We can apply StringIndexer to several columns in a PySpark Dataframe


```python
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(data) for column in list(set(data.columns)-set(['Drug','Na_to_K','Age'])) ]
```


```python
pipeline = Pipeline(stages=indexers)
```


```python
df_r = pipeline.fit(data).transform(data)
```


```python
df_r.show()
```

    +---+---+------+-----------+-------+-----+--------+-----------------+---------+
    |Age|Sex|    BP|Cholesterol|Na_to_K| Drug|BP_index|Cholesterol_index|Sex_index|
    +---+---+------+-----------+-------+-----+--------+-----------------+---------+
    | 23|  F|  HIGH|       HIGH| 25.355|drugY|     0.0|              0.0|      1.0|
    | 47|  M|   LOW|       HIGH| 13.093|drugC|     1.0|              0.0|      0.0|
    | 47|  M|   LOW|       HIGH| 10.114|drugC|     1.0|              0.0|      0.0|
    | 28|  F|NORMAL|       HIGH|  7.798|drugX|     2.0|              0.0|      1.0|
    | 61|  F|   LOW|       HIGH| 18.043|drugY|     1.0|              0.0|      1.0|
    | 22|  F|NORMAL|       HIGH|  8.607|drugX|     2.0|              0.0|      1.0|
    | 49|  F|NORMAL|       HIGH| 16.275|drugY|     2.0|              0.0|      1.0|
    | 41|  M|   LOW|       HIGH| 11.037|drugC|     1.0|              0.0|      0.0|
    | 60|  M|NORMAL|       HIGH| 15.171|drugY|     2.0|              0.0|      0.0|
    | 43|  M|   LOW|     NORMAL| 19.368|drugY|     1.0|              1.0|      0.0|
    | 47|  F|   LOW|       HIGH| 11.767|drugC|     1.0|              0.0|      1.0|
    | 34|  F|  HIGH|     NORMAL| 19.199|drugY|     0.0|              1.0|      1.0|
    | 43|  M|   LOW|       HIGH| 15.376|drugY|     1.0|              0.0|      0.0|
    | 74|  F|   LOW|       HIGH| 20.942|drugY|     1.0|              0.0|      1.0|
    | 50|  F|NORMAL|       HIGH| 12.703|drugX|     2.0|              0.0|      1.0|
    | 16|  F|  HIGH|     NORMAL| 15.516|drugY|     0.0|              1.0|      1.0|
    | 69|  M|   LOW|     NORMAL| 11.455|drugX|     1.0|              1.0|      0.0|
    | 43|  M|  HIGH|       HIGH| 13.972|drugA|     0.0|              0.0|      0.0|
    | 23|  M|   LOW|       HIGH|  7.298|drugC|     1.0|              0.0|      0.0|
    | 32|  F|  HIGH|     NORMAL| 25.974|drugY|     0.0|              1.0|      1.0|
    +---+---+------+-----------+-------+-----+--------+-----------------+---------+
    only showing top 20 rows


​    


```python
assembler = VectorAssembler(
  inputCols=['Age',
             'Sex_index', 
             'BP_index', 
             'Cholesterol_index', 
             'Na_to_K'],
              outputCol="features")
```


```python
output = assembler.transform(df_r)
```

Now we can fill the target variable Drug, 

Deal with type of Drug


```python
from pyspark.ml.feature import StringIndexer
```


```python
indexer = StringIndexer(inputCol="Drug", outputCol="DrugIndex")
output_fixed = indexer.fit(output).transform(output)
```


```python
final_data = output_fixed.select("features",'DrugIndex')
```


```python
train_data,test_data = final_data.randomSplit([0.7,0.3])
```


```python
train_data.show()
```

    +--------------------+---------+
    |            features|DrugIndex|
    +--------------------+---------+
    |(5,[0,4],[29.0,12...|      2.0|
    |(5,[0,4],[31.0,30...|      0.0|
    |(5,[0,4],[34.0,18...|      0.0|
    |(5,[0,4],[39.0,9....|      2.0|
    |(5,[0,4],[40.0,27...|      0.0|
    |(5,[0,4],[47.0,10...|      2.0|
    |(5,[0,4],[50.0,7....|      2.0|
    |(5,[0,4],[58.0,18...|      0.0|
    |(5,[0,4],[60.0,13...|      3.0|
    |(5,[0,4],[66.0,16...|      0.0|
    |(5,[0,4],[68.0,11...|      3.0|
    |(5,[0,4],[70.0,9....|      3.0|
    |(5,[0,4],[70.0,13...|      3.0|
    |(5,[0,4],[74.0,9....|      3.0|
    |[15.0,0.0,0.0,1.0...|      0.0|
    |[15.0,0.0,2.0,0.0...|      1.0|
    |[15.0,1.0,0.0,1.0...|      0.0|
    |[16.0,0.0,0.0,1.0...|      0.0|
    |[16.0,1.0,0.0,1.0...|      0.0|
    |[18.0,1.0,0.0,1.0...|      0.0|
    +--------------------+---------+
    only showing top 20 rows




### The Classifiers


```python
from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifierfrom pyspark.ml import Pipeline
```

Create two models:


* A single decision tree

* A random forest

    

We will be using a college dataset to try to classify colleges as Private or Public based off these features


```python
# Use mostly defaults to make this comparison "fair"

dtc = DecisionTreeClassifier(labelCol='DrugIndex',featuresCol='features')
rfc = RandomForestClassifier(labelCol='DrugIndex',featuresCol='features')
```

Train  models:


```python
# Train the models (its three models, so it might take some time)
dtc_model = dtc.fit(train_data)
```


```python
rfc_model = rfc.fit(train_data)
```



## Model Comparison

Let's compare each of these models!


```python
dtc_predictions = dtc_model.transform(test_data)rfc_predictions = rfc_model.transform(test_data)#gbt_predictions = 
```

**Evaluation Metrics:**


```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
```


```python
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="DrugIndex", predictionCol="prediction", metricName="accuracy")
```


```python
dtc_acc = acc_evaluator.evaluate(dtc_predictions)
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
```


```python
print("Here are the results!")
print('-'*80)
print('A single decision tree had an accuracy of: {0:2.2f}%'.format(dtc_acc*100))
print('-'*80)
print('A random forest ensemble had an accuracy of: {0:2.2f}%'.format(rfc_acc*100))

```

    Here are the results!--------------------------------------------------------------------------------A single decision tree had an accuracy of: 92.86%--------------------------------------------------------------------------------A random forest ensemble had an accuracy of: 89.29%



You can download the notebook [here](https://github.com/ruslanmv/Machine-Learning-with-Python-and-Spark/blob/master/Decision-Trees/Decision-Tree-classification.ipynb)

**Congratulations!** We have practiced Decision Tree classification with Python and Spark.

