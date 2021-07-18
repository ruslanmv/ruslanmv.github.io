---
title: "Regression with Neural Networks"
excerpt: "Regression with Neural Networks"

header:
  image: "../assets/images/posts/2020-09-17-Regression-with-Neural-Networks/image.jpg"
  teaser: "../assets/images/posts/2020-09-17-Regression-with-Neural-Networks/image.jpg"
  caption: "Laptop"
  
---

For this project, we are going to work on evaluating price of houses given the following features:

1. Year of sale of the house
2. The age of the house at the time of sale
3. Distance from city center
4. Number of stores in the locality
5. The latitude
6. The longitude



Note: This notebook uses `python 3` and these packages: `tensorflow`, `pandas`, `matplotlib`, `scikit-learn`.

## Importing Libraries & Helper Functions

First of all, we will need to import some libraries and helper functions. This includes TensorFlow and some utility functions that I've written to save time.


```python
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

```


```python
%matplotlib inline
print('Libraries imported.')
```

#  Importing the Data

You can download the data from [here](https://github.com/ruslanmv/Regression-with-Neural-Networks/blob/main/data.csv).

The dataset is saved in a `data.csv` file. We will use `pandas` to take a look at some of the rows.


```python
df = pd.read_csv('data.csv', names = column_names)
df.head()
```

## Check Missing Data

It's a good practice to check if the data has any missing values. In real world data, this is quite common and must be taken care of before any data pre-processing or model training.


```python
df.isna()
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
      <th>serial</th>
      <th>date</th>
      <th>age</th>
      <th>distance</th>
      <th>stores</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 8 columns</p>
</div>




```python
df.isna().sum()
```




    serial       0
    date         0
    age          0
    distance     0
    stores       0
    latitude     0
    longitude    0
    price        0
    dtype: int64



We can see that there are no nun values

# Data Normalization



We can make it easier for optimization algorithms to find minimas by normalizing the data before training a model.


```python
df.head()
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
      <th>serial</th>
      <th>date</th>
      <th>age</th>
      <th>distance</th>
      <th>stores</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2009</td>
      <td>21</td>
      <td>9</td>
      <td>6</td>
      <td>84</td>
      <td>121</td>
      <td>14264</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2007</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>86</td>
      <td>121</td>
      <td>12032</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2016</td>
      <td>18</td>
      <td>3</td>
      <td>7</td>
      <td>90</td>
      <td>120</td>
      <td>13560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2002</td>
      <td>13</td>
      <td>2</td>
      <td>2</td>
      <td>80</td>
      <td>128</td>
      <td>12029</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2014</td>
      <td>25</td>
      <td>5</td>
      <td>8</td>
      <td>81</td>
      <td>122</td>
      <td>14157</td>
    </tr>
  </tbody>
</table>
</div>



We will skip the first column serial, there are several ways to do, one is 


```python
df.iloc[:,1:].head()
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
      <th>date</th>
      <th>age</th>
      <th>distance</th>
      <th>stores</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009</td>
      <td>21</td>
      <td>9</td>
      <td>6</td>
      <td>84</td>
      <td>121</td>
      <td>14264</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>86</td>
      <td>121</td>
      <td>12032</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>18</td>
      <td>3</td>
      <td>7</td>
      <td>90</td>
      <td>120</td>
      <td>13560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2002</td>
      <td>13</td>
      <td>2</td>
      <td>2</td>
      <td>80</td>
      <td>128</td>
      <td>12029</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>25</td>
      <td>5</td>
      <td>8</td>
      <td>81</td>
      <td>122</td>
      <td>14157</td>
    </tr>
  </tbody>
</table>
</div>



and we normalize


```python
df = df.iloc[:,1:]
df_norm = (df - df.mean())/df.std()
df_norm.head()
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
      <th>date</th>
      <th>age</th>
      <th>distance</th>
      <th>stores</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.015978</td>
      <td>0.181384</td>
      <td>1.257002</td>
      <td>0.345224</td>
      <td>-0.307212</td>
      <td>-1.260799</td>
      <td>0.350088</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.350485</td>
      <td>-1.319118</td>
      <td>-0.930610</td>
      <td>-0.609312</td>
      <td>0.325301</td>
      <td>-1.260799</td>
      <td>-1.836486</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.298598</td>
      <td>-0.083410</td>
      <td>-0.618094</td>
      <td>0.663402</td>
      <td>1.590328</td>
      <td>-1.576456</td>
      <td>-0.339584</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.266643</td>
      <td>-0.524735</td>
      <td>-0.930610</td>
      <td>-0.927491</td>
      <td>-1.572238</td>
      <td>0.948803</td>
      <td>-1.839425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.932135</td>
      <td>0.534444</td>
      <td>0.006938</td>
      <td>0.981581</td>
      <td>-1.255981</td>
      <td>-0.945141</td>
      <td>0.245266</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

##  Convert Label Value

Because we are using normalized values for the labels, we will get the predictions back from a trained model in the same distribution. So, we need to convert the predicted values back to the original distribution if we want predicted prices.


```python
y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)


print(convert_label_value(-1.836486), " that corresponds to the price 12032")
```

    12031  that corresponds to the price 12032


# Create Training and Test Sets

## Select Features

Make sure to remove the column __price__ from the list of features as it is the label and should not be used as a feature.


```python
x = df_norm.iloc[:,:6]
x.head()
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
      <th>date</th>
      <th>age</th>
      <th>distance</th>
      <th>stores</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.015978</td>
      <td>0.181384</td>
      <td>1.257002</td>
      <td>0.345224</td>
      <td>-0.307212</td>
      <td>-1.260799</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.350485</td>
      <td>-1.319118</td>
      <td>-0.930610</td>
      <td>-0.609312</td>
      <td>0.325301</td>
      <td>-1.260799</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.298598</td>
      <td>-0.083410</td>
      <td>-0.618094</td>
      <td>0.663402</td>
      <td>1.590328</td>
      <td>-1.576456</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.266643</td>
      <td>-0.524735</td>
      <td>-0.930610</td>
      <td>-0.927491</td>
      <td>-1.572238</td>
      <td>0.948803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.932135</td>
      <td>0.534444</td>
      <td>0.006938</td>
      <td>0.981581</td>
      <td>-1.255981</td>
      <td>-0.945141</td>
    </tr>
  </tbody>
</table>
</div>



##  Select Labels

We select the prices


```python
y = df_norm.iloc[:,-1:]
y.head()
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
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.350088</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.836486</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.339584</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.839425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.245266</td>
    </tr>
  </tbody>
</table>
</div>



##  Feature and Label Values

We will need to extract just the numeric values for the features and labels as the TensorFlow model will expect just numeric values as input.


```python
x_arr = x.values
y_arr = y.values

print('features array shape', x_arr.shape)
print('labels array shape', y_arr.shape)
```

    features array shape (5000, 6)
    labels array shape (5000, 1)


## Train and Test Split

We will keep some part of the data aside as a __test__ set. The model will not use this set during training and it will be used only for checking the performance of the model in trained and un-trained states. This way, we can make sure that we are going in the right direction with our model training.


```python
x_train, x_test , y_train, y_test = train_test_split(x_arr,
                                                     y_arr, 
                                                     test_size = 0.05, 
                                                     random_state =0)
print( 'Training set', x_train.shape, y_train.shape)
print( 'Test set', x_test.shape, y_test.shape)
```

    Training set (4750, 6) (4750, 1)
    Test set (250, 6) (250, 1)


Let's write a function that returns an untrained model
of a certain architecture.
We're using a simple neural network architecture with just
three hidden layers.
We're going to use the rail you activation function on all
the layers except for the output layer.

#  Create the Model

We will use the sequential class from Keras.
And the cool thing about this class is you can just pass
on a list of layers to create your model architecture.

The first layer is a dense layer with just 10 nodes. 

We know that input shape is simply a list
of 6 values because we just have 6 features.

The activation is going to be really or rectified linear unit,
and the next layer will be again a fully connected or dense
layer. 

And this time, let's use 20 nodes and again, 
the same activation function relu and one more hidden layer with 5.
nodes and finally, the output layer with just 1 node.
so we have essentially, we have three hidden
layers.





## Create the Model

Let's write a function that returns an untrained model of a certain architecture.


```python
def get_model():
    model = Sequential([
        Dense(10, input_shape = (6,), activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(5, activation = 'relu'),
        Dense(1) 
        
        
    ])
    
    model.compile(
         loss = 'mse',
         optimizer = 'adam'    
            )
    return model

```

This is the input and then we have three hidden layers.
but 10, 20 and five nodes respectively.
All the layers have activation function set to value
except for the output layer.
Since this is a regression problem, we just need the linear
output without any activation function here.

These are all fully connected layers, and the number
of parameters obviously correspond to the number of nodes
that we have.

We need to specify a loss function in this case means great
error, and we need to specify an optimizer.

In this case we are using Adam.

mse is a pretty common loss function used for regression problems.
Remember This is the loss function that the optimization
algorithm tries to minimize.
And we are using a variant of stochastic gradient descent
called Adam 

An optimization algorithm is for and that is to minimize
the loss of function.



```python

```


```python
get_model().summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 10)                70        
    _________________________________________________________________
    dense_1 (Dense)              (None, 20)                220       
    _________________________________________________________________
    dense_2 (Dense)              (None, 5)                 105       
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 6         
    =================================================================
    Total params: 401
    Trainable params: 401
    Non-trainable params: 0
    _________________________________________________________________


The first dense layer that you see here is our first
hidden there, which has 10 nodes.

The next one has 20 nodes.
Next one has 5 nodes.
And the final one node, the output there has one, and you can see
that we have trainable parameters count 401.

Because these are dense layers, they are fully
connected layers and if you want to understand how
these parameters,count is arrived at, you can simply
multiply the nodes in your output layer in in any
one of these layers with the notes in the proceeding here.

So if you if you take a look at dense to for example,
we have 5 nodes and in the preceding layer that
is connected to we have 20 nodes, so each no disconnected.


Each node of dense one is connected to each note of dense two,
which means we have total 100 connections.
But you see that you have 105  parameters
for the Slayer.
Why is that?
That's simply because even though we have 100 weights, we
also have a bias or intercept,  in every layer.

Now that is just one interceptor connected to all the nodes
of the layer that you calculating these parameters for so
520 gives you 100 and five into one gives you 5,
and the total is 105 and you can do this exercise for all
the layers and arrive at the same number of trainable
parameters.




#  Model Training



We can use an `EarlyStopping` callback from Keras to stop the model training if the validation loss stops decreasing for a few epochs.


```python
es_cb = EarlyStopping(monitor='val_loss',patience=5)
model = get_model()
preds_on_untrained = model.predict(x_test)
history = model.fit(
        x_train,y_train,
        validation_data =(x_test,y_test),
        epochs = 100,
        callbacks = [es_cb]

)
```

    Epoch 1/100
    149/149 [==============================] - 1s 3ms/step - loss: 0.6905 - val_loss: 0.2966
    Epoch 2/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.2348 - val_loss: 0.1922
    Epoch 3/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1783 - val_loss: 0.1677
    Epoch 4/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1676 - val_loss: 0.1666
    Epoch 5/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1631 - val_loss: 0.1633
    Epoch 6/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1602 - val_loss: 0.1612
    Epoch 7/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1589 - val_loss: 0.1616
    Epoch 8/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1562 - val_loss: 0.1602
    Epoch 9/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1558 - val_loss: 0.1608
    Epoch 10/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1561 - val_loss: 0.1623
    Epoch 11/100
    149/149 [==============================] - 0s 2ms/step - loss: 0.1550 - val_loss: 0.1604
    Epoch 12/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1540 - val_loss: 0.1605
    Epoch 13/100
    149/149 [==============================] - 0s 2ms/step - loss: 0.1539 - val_loss: 0.1573
    Epoch 14/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1530 - val_loss: 0.1537
    Epoch 15/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1526 - val_loss: 0.1600
    Epoch 16/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1523 - val_loss: 0.1544
    Epoch 17/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1515 - val_loss: 0.1567
    Epoch 18/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1516 - val_loss: 0.1601
    Epoch 19/100
    149/149 [==============================] - 0s 1ms/step - loss: 0.1510 - val_loss: 0.1593


##  Plot Training and Validation Loss

Let's use the `plot_loss` helper function to take a look training and validation loss.


```python
plot_loss(history)
```


​    
![png](../assets/images/posts/2020-09-17-Regression-with-Neural-Networks/Regression-with-Neural-Networks_45_0.png)
​    


The training and validation loss.
Values decreased as the training then don, and that's great.
But we don't know yet if the train model actually makes
reasonably accurate predictions.
So let's take a look at that.
Now. Remember that we had some predictions on the untrained
model. Similarly, we will make some predictions on the train
model on the same data set, of course, which is X test.

#  Predictions

##  Plot Raw Predictions

Let's use the `compare_predictions` helper function to compare predictions from the model when it was untrained and when it was trained.


```python
preds_on_trained = model.predict(x_test)
compare_predictions(preds_on_untrained,preds_on_trained,y_test)
```


​    
![png](../assets/images/posts/2020-09-17-Regression-with-Neural-Networks/Regression-with-Neural-Networks_49_0.png)
​    


You can see the pattern.
It's pretty much a linear, uh, plot for the train model.
Predictions now it's making mistakes, of course, but the
extent of those mistakes is, ah drastically less compared to
what's happening on the N train model.
So, in an ideal situation, our model should make predictions
same as labels.
And that means this dotted blue line, right?
But our green, uh, train model predictions are sort of
following this started blue line, which means that our model
training did actually work.

## Plot Price Predictions

The plot for price predictions and raw predictions will look the same with just one difference: The x and y axis scale is changed.


```python
price_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_trained = [convert_label_value(y) for y in preds_on_trained]
price_test= [convert_label_value(y) for y in y_test]
```


```python
compare_predictions(preds_on_untrained,preds_on_trained,y_test)
```


​    
![png](../assets/images/posts/2020-09-17-Regression-with-Neural-Networks/Regression-with-Neural-Networks_53_0.png)
​    

We pretty much get the same graph, but the ranges are now
different. You can see the ranges from 12,000 to 16,000 or something for
both predictions and labels You can see that the train model is a lot more
aligned in its predictions to ground truth compared to the
untrained model.

You can download the notebook [here](https://github.com/ruslanmv/Regression-with-Neural-Networks/blob/main/Regression-with-Neural-Networks.ipynb)



**Congratulations!** we have created a neural network that performs regression.
