---

title: "Top Python Coding Exercises and Interview Questions"
description: "A curated set of top Python coding exercises and interview questions, with solutions to practice for data science and software engineering roles."
last_modified_at: 2026-06-02
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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=I1RoZSBmaWJvbmFjaSBzZXF1ZW5jZSBpcyBnaXZlbiBieQojdGhlIGZvbGxvd2luZyByZWN1cnNpb24KI0Zfbj1GX3tuLTF9LUZfe24tMn0KIyBGXzA9MCBhbmQgRl8xPTEKCmxpc3RhPWxpc3QocmFuZ2UoMSwxMCkpCgpkZWYgZmlib25hY2NpKG4pOgogICAgcmV0dXJuIE5vbmUKCnByaW50KGxpc3QobWFwKGxhbWJkYSB4OiBmaWJvbmFjY2koeCksbGlzdGEpKSkK&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=JycnCkNvdW50IGhvdyAgbWFueSB0aW1lcyBpcyByZXBlYXRlZCBhIGNoYXJhY3RlciBpbiBhIHN0cmluZwpjZXJ0YWluIGNoYXJhY3RlcgonJycKZGVmIHNvbHV0aW9uKHN0cmluZyxzdWJzdHJpbmcpOgoKICAgICByZXR1cm4gTm9uZQogICAgIAogICAgIApwcmludChzb2x1dGlvbignbWlzc2lzaXBpJywncycpKQo&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=IyByZXBsYWNlIHRoZSBOb25lIHZhbHVlIHdpdGggcHJldmlvdXMgbnVtYmVyIG9mIHRoZSBmb2xsb3dpZyBsaXN0Cmxpc3QgPSBbMSxOb25lLCAzICw0LE5vbmUsNV0KCmRlZiBzb2x1dGlvbihsaXN0KToKCiAgICByZXR1cm4gTm9uZQoKcHJpbnQoc29sdXRpb24obGlzdCkpCg&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>

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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=IyBDb3VudCB0aGUgb2NjdXJyZW5jZXMgb2YgZWFjaCB3b3JkIGluIGEgZ2l2ZW4gc2VudGVuY2UKCmRlZiBzb2x1dGlvbihzdHJpbmcpOgoKICAgIHJldHVybiBOb25lCgoKcHJpbnQoc29sdXRpb24oJ1Nvbm8gQW5uYXBhb2xhIGEgQW5uYXBhb2xhIHBpYWNlIGxhIHBpenphIHBlcmNoZSBBbm5hcGFvbGEgYWRvcmEgbGEgbW96YXJlbGxhJykpCg&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>

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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=IiIiCldyaXRlIGEgZnVuY3Rpb24gdGhhdCByZXR1cm5zIHRoZSBlbGVtZW50cyBvbiBvZGQgcG9zaXRpb25zICgwCmJhc2VkKSBpbiBhIGxpc3QKIiIiCgoKZGVmIHNvbHV0aW9uKGlucHV0KToKCiAgICByZXR1cm4gTm9uZQogICAgCiAgICAKcHJpbnQoc29sdXRpb24oWzAsMSwyLDMsNCw1XSkpCnByaW50KHNvbHV0aW9uKFsxLC0xLDIsLTJdKSkKYXNzZXJ0IHNvbHV0aW9uKFswLDEsMiwzLDQsNV0pID09IFsxLDMsNV0KYXNzZXJ0IHNvbHV0aW9uKFsxLC0xLDIsLTJdKSA9PSBbLTEsLTJdCg&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>

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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=IiIiCldyaXRlIGEgZnVuY3Rpb24gdGhhdCB0YWtlcyBhIG51bWJlciBhbmQgcmV0dXJucyBhIGxpc3Qgb2YgaXRzCmRpZ2l0cwoiIiIKZGVmIHNvbHV0aW9uKGlucHV0KToKCiAgICByZXR1cm4gTm9uZQoKCnByaW50KHNvbHV0aW9uKDQwMCkpCmFzc2VydCBzb2x1dGlvbigxMjMpID09IFsxLDIsM10KYXNzZXJ0IHNvbHV0aW9uKDQwMCkgPT0gWzQsMCwwXQo&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=IiIiCkZyb206IGh0dHA6Ly9jb2RpbmdiYXQuY29tL3Byb2IvcDEyNjk2OApSZXR1cm4gdGhlICJjZW50ZXJlZCIgYXZlcmFnZSBvZiBhbiBhcnJheSBvZiBpbnRzLCB3aGljaCB3ZSdsbApzYXkgaXMgdGhlIG1lYW4gYXZlcmFnZSBvZiB0aGUgdmFsdWVzLCBleGNlcHQgaWdub3JpbmcgdGhlIGxhcmdlc3QgYW5kCnNtYWxsZXN0IHZhbHVlcyBpbiB0aGUgYXJyYXkuIElmIHRoZXJlIGFyZSBtdWx0aXBsZSBjb3BpZXMgb2YgdGhlCnNtYWxsZXN0IHZhbHVlLCBpZ25vcmUganVzdCBvbmUgY29weSwgYW5kIGxpa2V3aXNlIGZvciB0aGUKbGFyZ2VzdCAgdmFsdWUuIFVzZSBpbnQgZGl2aXNpb24gdG8gcHJvZHVjZSB0aGUgZmluYWwgYXZlcmFnZS4gWW91IG1heQphc3N1bWUgIHRoYXQgdGhlIGFycmF5IGlzIGxlbmd0aCAzIG9yIG1vcmUuCiIiIgoKZGVmIHNvbHV0aW9uKG51bWJlcnMpOgoKICAgIHJldHVybiBOb25lCgoKcHJpbnQoc29sdXRpb24oWzEsIDEsIDUsIDUsIDEwLCA4LCA3XSkpCmFzc2VydCBzb2x1dGlvbihbMSwgMiwgMywgNCwgMTAwXSkgPT0gMwphc3NlcnQgc29sdXRpb24oWzEsIDEsIDUsIDUsIDEwLCA4LCA3XSkgPT0gNQphc3NlcnQgc29sdXRpb24oWy0xMCwgLTQsIC0yLCAtNCwgLTIsIDBdKSA9PSAtMwo&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=I0ZvciBhIGdpdmVuIHNldGVuY2UsIHJldHVybiB0aGUgYXZlcmFnZSB3b3JkIGxlbmdodAojIFJlbW92ZSB0aGUgcHVuY3R1YXRpb24gZmlyc3QKCnNlbnRlbmNlMT0gIiBIaSBhbGwsIG15IG5hbWUgaXMgVG9tLi4uIEkgYW0gZnJvbSBBdXN0cmFsaWEiCnNlbnRlbmNlMj0gIiBJIG5lZWQgd29yayBoYXJkIHRvIGxlYXJuIG1vcmUgUHl0aG9uISIKCmRlZiBjb3VudHdvcmQoc2VudGVuY2UpOgoKICAgIHJldHVybiBOb25lCgoKcHJpbnQoY291bnR3b3JkKHNlbnRlbmNlMSkpCnByaW50KGNvdW50d29yZChzZW50ZW5jZTIpKQo&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>





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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=I0dpdmVuIHR3byBub24gbmVnYXRpdmUgaW50ZWdlcnMgbnVtMSBhbmQgbnVtMgojcmVwcmVzbmV0ZWQgYXMgc3RyaW5nCiNyZXR1cm4gdGhlIHN1bSBvZiBudW0xIGFuZCBudW0yCgojIG51bTEgYW5kIG51bTIgY29udGFpbnMgZGlnaXRzIDAtOQojIG51bTEgYW5kIG51bTIgZG9lcyBub3QgY29udGFpbiBhbnkgbGVhZGluZyB6ZXJvCgpudW0xPSczNjQnCm51bTI9JzE4MzYnCgpkZWYgc3VtYXRpb24obnVtMSxudW0yKToKICAgIHJldHVybiBOb25lCnByaW50KHN1bWF0aW9uKG51bTEsbnVtMikpCg&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=IyBsaXN0IG9mIGxldHRlcnMKbGV0dGVycyA9IFsnYScsICdiJywgJ2QnLCAnZScsICdpJywgJ2onLCAnbyddCgojIGZ1bmN0aW9uIHRoYXQgZmlsdGVycyB2b3dlbHMKZGVmIGZpbHRlcl92b3dlbHMobGV0dGVyKToKICAgIHJldHVybiBOb25lCiAgICAgICAgCiAgICAgICAgCmZpbHRlcmVkX3Zvd2VscyA9IGZpbHRlcihmaWx0ZXJfdm93ZWxzLCBsZXR0ZXJzKQoKcHJpbnQoJ1RoZSBmaWx0ZXJlZCB2b3dlbHMgYXJlOicpCmZvciB2b3dlbCBpbiBmaWx0ZXJlZF92b3dlbHM6CiAgICBwcmludCh2b3dlbCkK&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



### 11. Lambda function



Write a lambda function to add one number to 2

```python
solution = lambda x: x + 1
print(solution(2))
```

**Practice yourself**

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=CnNvbHV0aW9uID0gTm9uZQoKcHJpbnQoc29sdXRpb24oMikpCg&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=I1RoZSBpbnRlZ2VyIGNvdWxkIGJlIGVpdGhlciBwb3NpdGl2ZSBvciBuZWdhdGl2ZQojR2l2ZW4gYW5kIGludGVnZXIsIHJldHVybiB0aGUgaW50ZWdlciB3aXRoIHJldmVyc2VkIGRpZ2l0cwoKZGVmIHJldmVyc2VkaWdpdHMoeCk6CiAgICByZXR1cm4gTm9uZQoKcHJpbnQocmV2ZXJzZWRpZ2l0cygtMjMxKSkKcHJpbnQocmV2ZXJzZWRpZ2l0cygzNDUpKQo&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=IyBQeXRob24gcHJvZ3JhbSB0byBwcmludCBldmVuIGFuZCBvZGQgTnVtYmVycyBpbiBhIExpc3QKIyBsaXN0IG9mIG51bWJlcnMKbGlzdDEgPSBsaXN0KHJhbmdlKDIsMjApKQojIHVzaW5nIGxpc3QgY29tcHJlaGVuc2lvbgpldmVuX25vcyA9IE5vbmUKCnByaW50KCJFdmVuIG51bWJlcnMgaW4gdGhlIGxpc3Q6ICIsIGV2ZW5fbm9zKQoKb2RkX25vcyA9IE5vbmUKCnByaW50KCJPRERuIG51bWJlcnMgaW4gdGhlIGxpc3Q6ICIsIG9kZF9ub3MpCg&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=c2VudGVuY2UxPSdXZSBhcmUgcmVhbGx5IHBsZWFzZWQgdG8gbW1ldCB5b3UgaW4gb3VyIGNpdHksIFJ1c2xhbicKc2VudGVuY2UyPSdUaGUgY2l0eSB3YXMgaGl0IGJ5IGEgcmVhbGx5IGhlYXZ5IHN0b3JtLCBSdXNsYW4nCmRlZiBzb2x1dGlvbihzZW50ZW5jZTEsIHNlbnRlbmNlMik6CiAKICAgIHJldHVybiBOb25lCnByaW50KHNvbHV0aW9uKHNlbnRlbmNlMSwgc2VudGVuY2UyKSkK&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=I3ByaW50IHByaW1lIG51bWJlcnMKbGlzdGE9cmFuZ2UoMiwxMDApCgpwcmltZSA9Tm9uZQoKcHJpbnQocHJpbWUpCg&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>





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

<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=bGlzdCA9IFsiMSIsICI0IiwgIjAiLCAiNiIsICI5Il0KCmRlZiBzb2x1dGlvbihsaXN0KToKCiAgICByZXR1cm4gTm9uZQoKCnByaW50KHNvbHV0aW9uKGxpc3QpKQo&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



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



<figure style="margin:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden"><iframe src="https://coddy.tech/embed-editor?lang=python&theme=light&layout=stacked&code=bGlzdCA9IFsxLDIyLDM0LDEyLDExLDQzLDhdCgoKCgoKICAgICAgICAgICAgCiAgICAgICAgICAgIAogICAgICAKICAgICAgICAgICAgICAgIHByaW50KCJUaGlzIGlzIGJpZ2dlciB0aGFuIGFsbCIsaXRlbSxyZXNhKQogICAgIAogICAgICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgICAgIAoKICAgICAgICAgICAgCiAgICAgICAgICAgIAogICAgICAgICAgCiAgICAgICAgICAgICAgICBwcmludCgiVGhpcyBpcyBzbWFsbGVyIHRoYW4gYWxsIixpdGVtLHJlc2Ffc21hbGwpICAgCiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgcHJpbnQoIkZvciB0aGUgbnVtYmVyIiwKICAgICAgICAgICAgICAgICAgIGl0ZW0sCiAgICAgICAgICAgICAgICAgICAiVGhlIHNtYWxsZXIgbnVtYmVycyBhcmUiLAogICAgICAgICAgICAgICAgICAgcmVzYSwKICAgICAgICAgICAgICAgICAgICJUaGUgYmlnZ2VyIG51bWJlcnMgYXJlIiwKICAgICAgICAgICAgICAgICAgICByZXNhX3NtYWxsCiAgICAgICAgICAgICAgICAgICAgKQo&credit=0" width="100%" height="420" style="display:block;border:0;width:100%" title="Python code editor by Coddy" loading="lazy"></iframe><figcaption style="box-sizing:border-box;padding:5px 12px;font:11px/1.4 system-ui,-apple-system,sans-serif;color:#64748b;background:#f1f5f9;border-top:1px solid #e2e8f0;text-align:right" dir="ltr">Powered by <a href="https://coddy.tech/embed" target="_blank" rel="noopener" style="color:#0077ff;font-weight:600;text-decoration:none;">Coddy</a></figcaption></figure>



If you liked the playground in python there are tools that allow you to embed code into your HTML website:

- [Trinket.io](https://trinket.io/)
- [Repl.it](https://repl.it/)
- [Brython](http://www.brython.info/tests/console.html)
- [Skulpt](http://skulpt.org/)
- [PyPy Sandbox](http://doc.pypy.org/en/latest/sandbox.html)
- [CreateWithCode](https://create.withcode.uk/)

**Congratulations** you have practiced some essential coding exercises.


## Related tutorials

- [What a data scientist should know](/blog/What-all-should-a-data-scientist-know)
- [Top Big Tech data science questions](/blog/Top-Big-Tech-Data-Science-Questions)
- [How to use Jupyter Notebook](/blog/How-to-use-Jupyter-Notebook)

## FAQ

**Are these Python questions good for interviews?**

Yes — they cover the patterns commonly asked in data science and software interviews, with worked solutions.

**Where can I run these exercises?**

In any Python setup; a quick option is [Jupyter Notebook](/blog/How-to-use-Jupyter-Notebook).
