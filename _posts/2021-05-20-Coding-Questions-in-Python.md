---

title: " Top Coding Exercises in Python"
excerpt: "Top coding Exercises in Python that we should know "

header:
  image: "../assets/images/posts/2021-05-20-Coding-Questions-in-Python/mac.jpg"
  teaser: "../assets/images/posts/2021-05-20-Coding-Questions-in-Python/mac.jpg"
  caption: "Laptop displaying codes"

---

Hello I have collected a set of coding exercises in Python that we should know during any interview for Data Scientist or Data Engineer.

You should be familiar with Loops, Lists (ArrayLists), and Dictionaries (HashMaps). 

```python
"""
Example Python Syntax 
"""
#Loops
for x in l: "Iterate on x for each value in list"
for i in range(0,5): "Iterate on i from value 0 to 4"
for k, v in d.items(): "Iterate on each key, value pair in dict"
#Lists (Array)
l = [] "Define an empty list"
l[i] "Return value at index i in list"
len(l) "Return length of list"
l.append(x) "Add value x to the end of list"
l.sort() "Sort values in list - in place sort, returns None"
sorted(l) "Return sorted copy of list"
x in l: "Evaluate True if x is contained in the list"
#Dictionary (HashMap)
d = {} "Define an empty Dictionary"
d[x] "Return value for key x"
d[x] = 1 "Set value for key x to 1"
d.keys() "Return list of keys"
d.values() "Return list of values"
#Tuple
tup = ()
tup = (1,2) + tup
#Other functions
reversed(n) "reverse a list"
random.random() "random number between 0 and 1"
random.randrange(start, stop) "Return a randomly selected element
from range(start, stop)"
isinstance(x, list) "returns True if x is instance of list"
split() "returns a list of all the words in the string"
ceil() "returns the smallest integer value greater than or equal to x"
```



Let us practice your Python skills , in the next sections there are some exercises that you can see the solution ,  for each solution there is a small python editor where you can write the solution and run it .  Just remove the `return None`with the correct solution.



### 1. Fibonacci Program

Extract the Fibonacci numbers from a list from 1 to 100

```python
#The fibonaci sequence is given by
#the following recursion
#F_n=F_{n-1}-F_{n-2}
# F_0=0 and F_1=1

lista=list(range(1,10))

def fibonacci(n):
    if n < 0 :
        print("The n should be positive")
    elif n == 0 :
        return 0
    elif n == 1 :
        return 1
    else:
        return fibonacci(n-1)+fibonacci(n-2)
        
print(list(map(lambda x: fibonacci(x),lista)))
```

**Practice yourself**

<iframe src="https://trinket.io/embed/python/701f59c1f2" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 2. Occurrence characters

```python
'''
Count how  many times is repeated a character in a string
certain character
'''
def solution(string,substring):
     string=list(string)
     #string=[char for char in string]
     count = string.count(substring)
     return count
print(solution('missisipi','s'))
```

**Practice yourself**

<iframe src="https://trinket.io/embed/python/3dc3185e4f" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 3. Replace None values of list

```python
# replace the None value with previous number of the followig list
list = [1,None, 3 ,4,None,5]

def solution(list):
    count=0
    for n in list :
        if n == None:
           list[count]=list[count-1]
        count=count+1
    return list

print(solution(list))
```



**Practice yourself:**

<iframe src="https://trinket.io/embed/python/bef350191b" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

### 4. Count the occurrences of each word in a given sentence



```python
def solution(string):
    counts = {}
    split=string.split()
    for word in split:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] =  1
    return counts
    
    
print(solution('Sono Annapaola a Annapaola piace la pizza perche Annapaola adora la mozarella'))
```

**Practice yourself:**

<iframe src="https://trinket.io/embed/python/74669246cd" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

### 5. Odd position of list 

```python
"""
Write a function that returns the elements on odd positions (0
based) in a list
"""
#Solution:
#First we create a list of all positions of the input
#       a=list(range(0,len(input)))
#Second we need take all elements of a list a
#       num for num in a
#Third need identify if a number is odd
#       This is possible with  num % 2 != 0
# Fourth we select the elements of the input by the index
#        input[i] for i in b

def solution(input):
    a=list(range(0,len(input)))
    b=[num for num in a if num % 2 != 0 ]
    c = [ input[i] for i in b]
    return c
print(solution([0,1,2,3,4,5]))
print(solution([1,-1,2,-2]))
assert solution([0,1,2,3,4,5]) == [1,3,5]
assert solution([1,-1,2,-2]) == [-1,-2]
```

**Practice yourself:**

<iframe src="https://trinket.io/embed/python/62512b3494" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

### 6. List of its digits



```python
"""
Write a function that takes a number and returns a list of its
digits
"""
def solution(input):
    string=str(input)
    lista=list(string)
    new_list = [int(i) for i in lista]
    return new_list
print(solution(400)) 
assert solution(123) == [1,2,3]
assert solution(400) == [4,0,0]
```



**Practice yourself:**

<iframe src="https://trinket.io/embed/python/5e138ee3ef" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 7. Custom Average List

```python
"""
From: http://codingbat.com/prob/p126968
Return the "centered" average of an array of ints, which we'll 
say is the mean average of the values, except ignoring the largest and 
smallest values in the array. If there are multiple copies of the 
smallest value, ignore just one copy, and likewise for the 
largest  value. Use int division to produce the final average. You may 
assume  that the array is length 3 or more.
"""

def solution(numbers):
    sum=0
    count=0
    numbers.sort()
#    numbers = list(dict.fromkeys(numbers))

    size=len(numbers)
    print(numbers)
    for n in range(1, size-1):
        sum=sum+numbers[n]
        count = count +1
    return int(sum/count)

#print(solution([1, 2, 3, 4, 100]))
print(solution([1, 1, 5, 5, 10, 8, 7]))
assert solution([1, 2, 3, 4, 100]) == 3
assert solution([1, 1, 5, 5, 10, 8, 7]) == 5
assert solution([-10, -4, -2, -4, -2, 0]) == -3
```



**Practice yourself:**

<iframe src="https://trinket.io/embed/python/45146e48eb" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 8. For a given sentence, return the average word length



```python
#For a given setence, return the average word lenght
# Remove the punctuation first

sentence1= " Hi all, my name is Tom... I am from Australia"
sentence2= " I need work hard to learn more Python!"

def countword(sentence):
    for p in "!?',;.":
        sentence = sentence.replace(p,'')
    words = sentence.split()
    print(words)
    return round(sum(len(word) for word in words)/len(words),2)

print(countword(sentence1))
print(countword(sentence2))
```

**Practice yourself:**

<iframe src="https://trinket.io/embed/python/a762c13b74" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>





### 9. Summation of numbers in a strings

```python
#Given two non negative integers num1 and num2
#represneted as string
#return the sum of num1 and num2

# num1 and num2 contains digits 0-9
# num1 and num2 does not contain any leading zero

num1='364'
num2='1836'

def sumation(num1,num2):
    return str(int(num1)+int(num2))
print(sumation(num1,num2))
```

**Practice yourself**

<iframe src="https://trinket.io/embed/python/2423f98330" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 10. Filter only the vowels

Write a program to filter only the vowels from a list  letters = ['a', 'b', 'd', 'e', 'i', 'j', 'o']

```python
# list of letters
letters = ['a', 'b', 'd', 'e', 'i', 'j', 'o']

# function that filters vowels
def filter_vowels(letter):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if(letter in vowels):
        return True
    else:
        return False
filtered_vowels = filter(filter_vowels, letters)

print('The filtered vowels are:')
for vowel in filtered_vowels:
    print(vowel)
```

**Practice yourself**

<iframe src="https://trinket.io/embed/python/c446cee1b4" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 11. Lambda function



Write a lambda function to add one number to 2

```python
solution = lambda x: x + 1
print(solution(2))
```

**Practice yourself**

<iframe src="https://trinket.io/embed/python/5d078074d9" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 12. Reversed digits

```python
#The integer could be either positive or negative
#Given and integer, return the integer with reversed digits

def reversedigits(x):
    string = str(x)
    if string[0] == '-':
        return int('-'+string[:0:-1])
    else:
        return int(string[::-1])

print(reversedigits(-231))
print(reversedigits(345))
```

**Practice yourself**

<iframe src="https://trinket.io/embed/python/9ffd08899b" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 13. Even and odd number

```python
# Python program to print even and odd Numbers in a List
# list of numbers
list1 = list(range(2,20))
# using list comprehension
even_nos = [num for num in list1 if num % 2 == 0]

print("Even numbers in the list: ", even_nos)

odd_nos = [num for num in list1 if num % 2 != 0]

print("ODDn numbers in the list: ", odd_nos)
```

**Practice yourself**

<iframe src="https://trinket.io/embed/python/29e04ff574" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 14. Extract repeated word

Write a Python program to extract repeated word in both sentences and no repeated

```python
sentence1='We are really pleased to mmet you in our city, Ruslan'
sentence2='The city was hit by a really heavy storm, Ruslan'
def solution(sentence1, sentence2):
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())
    return sorted(list(set1^set2)),sorted(list(set1&set2))
print(solution(sentence1, sentence2))
```

**Practice yourself**

<iframe src="https://trinket.io/embed/python/f9f2569cef" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 15. Prime numbers

```python
#print prime numbers
lista=range(2,100)

prime =[  x for x in lista if not

        [t for t in range(2,x) if not x%t]

]
print(prime)
```

**Practice yourself**

<iframe src="https://trinket.io/embed/python/73ca0127ee" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>





###  16. Sort numbers

 Write a sorting algorithm for a numerical dataset in Python.

```python
list = ["1", "4", "0", "6", "9"]
def solution(list):
    lista = [int(i) for i in list]
    lista.sort()
    return lista

print(solution(list))
```

**Practice yourself**

<iframe src="https://trinket.io/embed/python/e176aa736c" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



### 17. Identify numbers in a list

Given a list of numbers list =[1,22,34,12,11,43,8] Check if there is a number where all numbers to the right are greater and to the left are smaller .Give me the position of the list where is located.

```python
list = [1,22,34,12,11,43,8]
for item in list:
            resa=[number for number in list if item > number]
            size1=len(resa)
            size2=len(list)-1
            if (size1 == size2):
                print("This is bigger than all",item,resa)
            resa_small=[number for number in list if item < number]
            size3=len(resa_small)
            if (size2 == size3):
                print("This is smaller than all",item,resa_small)    
            print("For the number",
                   item,
                   "The smaller numbers are",
                   resa,
                   "The bigger numbers are",
                    resa_small
            )
```

**Practice yourself**



<iframe src="https://trinket.io/embed/python/90a99fcfbf" width="100%" height="300" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>



If you liked the playground in python there are tools that allow you to embed code into your HTML website:

- [Trinket.io](https://trinket.io/)
- [Repl.it](https://repl.it/)
- [Brython](http://www.brython.info/tests/console.html)
- [Skulpt](http://skulpt.org/)
- [PyPy Sandbox](http://doc.pypy.org/en/latest/sandbox.html)
- [CreateWithCode](https://create.withcode.uk/)

**Congratulations** you have practiced some essential coding exercises.

