---
title: "How to create a List, Array and a Matrix in Python"
excerpt: "Creation of a lists, arrays and matrices"

header:
  image: "../assets/images/posts/2020-07-27-How-to-create-a-List-Array-and-Matrix-in-Python/pc.jpg"
  teaser: "../assets/images/posts/2020-07-27-How-to-create-a-List-Array-and-Matrix-in-Python/pc.jpg"
  caption: "Programming isn't about what you know; it's about what you can figure out. - Chris Pine"
  
---

Hello everyone, today I would like to explain how to use a **list**, **arrays** and **matrices**. As you know manage very well those concepts are a key to succeed in you career in you **IT profession**.

Moreover when you are doing several **coding Interviews**, usually you have few minutes to solve the problems and if you don't remember the main concepts, will be difficult finish the test at time.

# Creation of  a List

Let’s start by constructing the simplest list, that is the **empty list** by

```python
mylist=[]
```

The **empty  list** is very **powerful** because you can do several things with list, you can allocate things in memory,  the empty list is not empty, indeed contains a list without elements . You can think that the empty list is like a empty bag that you will put objects inside.

Now let us add something to this empty list.

For example the string 

```python
'video_id'
```

There three methods for adding elements:

- append() - appends a single element to the list.
- extend() - appends elements of an iterable to the list.
- insert() - inserts a single item at a given position of the list.

If we want to add a **single element** in a list  we use append

```python
mylist.append('video_id')p
```

so you will get

```python
['video_id']
```

if you want to add two elements together,

`'video_id'` and `'language'`

 you should think that two things are iterable, so you should write it them in a proper way,  that is  adding list to a list

```python
mylist=[]
mylist.extend(['video_id','language'])
```

you will get

```python
['video_id', 'language']
```

moreover also you can use tuples because is iterable, 

```python
mylist=[]
mytuple=('video_id','language')
mylist.extend(mytuple)
```

```python
['video_id', 'language']
```

and finally if you want add  `'language_code'` to the first position

```python
mylist=[]
mylist.extend(['video_id','language'])
mylist.insert(0, 'language_code')
```

you will get

```python
['language_code', 'video_id', 'language']
```

remember that in  Python the first index begin with 0  from left to right, but also you can say

that an index start from -1 right to left,

```python
mylist=[]
mylist.extend(['video_id','language'])
mylist.insert(-3, 'language_code')
```

you will get the same results

```python
['language_code', 'video_id', 'language']
```

# List comprehension

One of the amazing things that Python has that I have never seen in other programming languages is the list comprehension.  

Is something like perform several loops for  appending elements to the empty list.

For that reason, I have started to discuss first the empty list. This allow us understand better the list comprehension.

List comprehension is a useful way to construct new lists based on the values in other lists. The structure of list comprehension generally looks like the following:

```python
mylist = [expression for element in iterable]
```

An expression can simply be the element in the iterable itself or some transformation of the element, such as checking the truth value of a condition or even more complex like perform an evaluation of your element, 

for example :

Given a function f(x)  defined as

```python
def f(x):
    y=2*x
    return y
```

in the domain 

```python
lista=[1,2,3,4]
```

the list comprehension Z

```python
[f(x) for x in lista ]
```

gives

```python
[2, 4, 6, 8]
```

In ordering to understand how to manage a **list compression**, let us discuss the following example:

### Use case : Odd numbers in a list

> Write a function that returns the elements on odd positions  in a list

Let us assume that you have the following input :

```python
input = [0,1,2,3,4,5]
```

First,  we create a list of all positions of the input,

```python
a=list(range(0,len(input)))
```

Second, we need take all elements of a list a, here is were can use a **list compression**,

```python
num for num in a
```

Third,  we need to identify if a number is odd, this is possible to know if the residue of 2 is not 0 otherwise the number should be even, 

```python
num % 2 != 0
```

Fourth, we select the elements of the input by the index,

```python
input[i] for i in b
```

and again, we have used  **list compression**.

Now we can put together into a new ffunction

```python
def solution(input):
    a=list(range(0,len(input)))
    b=[num for num in a if num % 2 != 0 ]
    c = [ input[i] for i in b]
    return c
```

You can check that is valid this solution

```python
assert solution([0,1,2,3,4,5]) == [1,3,5]
```

that is something that we expected.

The `sort()` method accepts a `reverse` parameter as an optional argument.

Setting `reverse = True` sorts the list in the descending order.

```
list.sort(reverse=True)
```

Alternatively for `sorted()`, you can use the following code.

```
sorted(list, reverse=True)
```

## Summary:

Let summarize what we have cover in few words:

> `l = []` "Define an empty list"
>
> `l[i]` "Return value at index i in list"
>
> `len(l)` "Return length of list"
>
> `l.append(x)` "Add value x to the end of list"
>
> `l.sort()` "Sort values in list - in place sort, returns None"
>
> `sorted(l)` "Return sorted copy of list"
>
> `x in l`: "Evaluate True if x is contained in the list"

# Matrices from a list

The next important element that we have to know is the  construction of matrices.

If you want to create an empty matrix with a list you can define a matrix as, 

```python
matrix = [[]]
```

which is   a list of lists, that can be obtained also 

```python
matrix = []
matrix.append([])
matrix.append([])
```

## Define a two-dimensional matrix 

Let assume that you want to build a matrix with 0

```python
# Creates a list containing 2 lists, each of 3 items, all set to 0
w, h = 3, 2
matrix = [[0 for x in range(w)] for y in range(h)] 
```

your output is 

```python
[[0, 0, 0], [0, 0, 0]]
```

As you see again, the  **list compression** appears again,  I know that maybe is you are coming from **Fortran**  or **C** like me, tis a bit non common,, where  you we have used Procedural Programming .

### Use case : Add two matrices using list comprehension

> Write a function that returns a matrix from the operation addition of  two matrices

```python
A = [[1,2,3],
    [4,5,6],
    [7,8,9]]
    
B = [[5,8,1],
    [6,7,3],
    [4,5,9]]

result = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

for r in result:
   print(r)
```

# Arrays

When you are dealing with matrices it s conveniently use a dictionary instead lists. 

You can define an empty matrix as follows

```py
matrix = {}
```

then you can populate it

```
Matrix[0,0] = 13
```

This works because `0,0` is a tuple, and you're using it as a key to index the dictionary.

## NumPy Array 

NumPy is a Python package useful for generating arrays, which have many distinctions from Python lists. 

The biggest difference is NumPy arrays use fewer resources than Python lists, which becomes important when storing a large amount of data.

To proceed, let’s import the NumPy package:

```python
import numpy as np 
```

then let us create a matrix from list inputs into 

```
lista=[1,2,3,4,5,6]
```

```
matrix = np.array(lista)
```

if you print the previous array, you got

```
array([1, 2, 3, 4, 5, 6])
```

now if you wants to  give a shape of matrix you can do the follow, define the shape of the matrix, you want, for example

2X3, so you says 

```python
shape = (2,3)
matrix=matrix.reshape(shape)
```

and the output

```python
array([[1, 2, 3],
       [4, 5, 6]])
```

This is matrix in a numpy array. with the form matrix[m,n]

The first row is corresponds to `matrix[0]`  a the second row  `matrix[1]`  

```python
matrix[1]
#array([4, 5, 6])
```

you can choose one element like

```python
matrix[0,0]
```

which is 1. 

## Numpy matrix to List 

Moreover we can return back to our  matrix in terms of list

```python
matrix_list=matrix.tolist()
```

```python
[[1, 2, 3], [4, 5, 6]]
```

## Convert Matrix to dictionary

Having the previous matrix in terms of list we can do

```python
# using dictionary comprehension for iteration
matrix_dic = {idx + 1 : matrix_list[idx] for idx in range(len(matrix_list))}
```

then

```python
matrix_dic 
```

gives

```python
{1: [1, 2, 3], 2: [4, 5, 6]}
```

by taking 

```python
matrix_dic[1]
```

you will obtain

```python
[1, 2, 3]
```

Another possibility to get the  dictionary is by  use **enumerate** and again use  **comprehension list**

```python
# enumerate used to perform assigning row number
matrix_dic = {idx: val for idx, val in enumerate(matrix_dic, start = 1)}
```

It is possible get your keys by using

```python
matrix_dic.keys()
```

you get

```python
dict_keys([1, 2])
```

and values

```python
matrix_dic.values()
```

```python
dict_values([[1, 2, 3], [4, 5, 6]])
```

## Summary

> `d = {}` "Define an empty Dictionary"
>
> `d[x]` "Return value for key x"
>
> `d[x] = 1` "Set value for key x to 1"
>
> `d.keys()` "Return list of keys"
>
> `d.values()` "Return list of values"

**Congratulation!** We have created   Lists, Arrays and a Matrices in Python 