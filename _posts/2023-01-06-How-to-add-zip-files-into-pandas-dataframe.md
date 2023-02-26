---
title: "How to add zip files into Pandas Dataframe"
excerpt: "How to add a zip file into a  Dataframe with Python"

header:
  image: "../assets/images/posts/2023-01-06-How-to-add-zip-files-into-pandas-dataframe/green.jpg"
  teaser: "../assets/images/posts/2023-01-06-How-to-add-zip-files-into-pandas-dataframe/green.jpg"
  caption: "Most hackers are young because young people tend to be adaptable. As long as you remain adaptable, you can always be a good hacker. Emmanuel Goldstein"
  
---

Hello everyone, today I am interested to show an interesting trick to include a zip file into a column pandas dataframe.

Sometimes when you are creating a unstructured database where you require include photos, videos, word documents, excel files or simple binary files.

There are plenty of methods to do this.  The method that I will persue is zip all the files that you want to storage then convert into pandas  dataframe.

# Getting files of a library

Let us assume that we want to insert some python libraries into a zip file.
For example I want to get the findspark application


```python
# Import the os module
import os
# Path
home = os.getcwd()
# Print the current working directory
print("Current working directory: {0}".format(home))
```

    Current working directory: /home/wsuser/work

```python
!mkdir folder
# Join various path components
target=os.path.join(home, "folder")
print(target)
# Change the current working directory
os.chdir(target)
# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))
```

    /home/wsuser/work/folder
    Current working directory: /home/wsuser/work/folder


### Here we put the files that we want to convert to dataframe


```python
!pip download findspark 
```

    Collecting findspark
      Downloading findspark-2.0.1-py2.py3-none-any.whl (4.4 kB)
    Saved ./findspark-2.0.1-py2.py3-none-any.whl
    Successfully downloaded findspark


Printing the files to be inserted into the dataframe


```python
!dir
```

    findspark-2.0.1-py2.py3-none-any.whl


```python
# Change the current working directory
os.chdir(home)
```

We are interested to zip the folder by using python

# Zip files python

Here we want to create a program that creates  a zip file of all the files contained in a certain folder.


```python
#importing the os module
import glob
import os
#to get the current working directory
current_dir=os.getcwd()
# Join various path components
folder_to_zip=os.path.join(current_dir, "folder")
print(folder_to_zip)
```

    /home/wsuser/work/folder



```python
def get_list_of_files(directory):
    # Path
    path = directory+"/*"
    print("Looking files in the path:",directory)
    files_to_zip=glob.glob(path)
    return  files_to_zip
```


```python
from zipfile import ZipFile
from io import BytesIO
def create_zip(folder_to_zip):
    """
    returns: zip archive
    """    
    files_to_zip=get_list_of_files(folder_to_zip)
    archive = BytesIO()
    with ZipFile(archive, 'w') as zip_archive:
        for file in files_to_zip:
            print(file)
            name=file[len(folder_to_zip)+1:]
            print(name)
            # Create n files on zip archive
            with zip_archive.open(name, 'w') as files:
                with open(file, 'rb') as file_data:
                    bytes_content = file_data.read()
                    files.write(bytes_content)
    return archive
```


```python
archive = create_zip(folder_to_zip)
```

    Looking files in the path: /home/wsuser/work/folder
    /home/wsuser/work/folder/findspark-2.0.1-py2.py3-none-any.whl
    findspark-2.0.1-py2.py3-none-any.whl



```python
# Flush archive stream to a file on disk
with open('data.zip', 'wb') as f:
    f.write(archive.getbuffer())
```


```python
type(archive)
```


    _io.BytesIO


```python
archive.close()
```


```python
if os.name == 'nt':
    print('I am Windows')
    !dir *.zip
else:
    print('I am on Unix')
    !ls *.zip -ltr
```

    I am on Unix
    -rw-rw---- 1 wsuser wscommon 4616 Feb 26 19:40 data.zip


We have the file zip that we want to convert insert into the dataframe


```python
from zipfile import ZipFile
with ZipFile('data.zip') as zip_archive:
    for item in zip_archive.filelist:
        print(item)
    print(f'\nThere are {len(zip_archive.filelist)} ZipInfo objects present in archive')
```

    <ZipInfo filename='findspark-2.0.1-py2.py3-none-any.whl' filemode='?rw-------' file_size=4446>
    
    There are 1 ZipInfo objects present in archive


## Read all files in .zip archive in python


```python
archive = ZipFile('data.zip', 'r')
```


```python
files = archive.namelist()
```


```python
files
```


    ['findspark-2.0.1-py2.py3-none-any.whl']


```python
type(archive)
```


    zipfile.ZipFile


```python
import zipfile
z = zipfile.ZipFile("data.zip", "r")
for filename in z.namelist(  ):
    print('File:', filename),
    byt = z.read(filename)
    print(type(byt))
    print ('has',len(byt),'bytes')
```

    File: findspark-2.0.1-py2.py3-none-any.whl
    <class 'bytes'>
    has 4446 bytes

```python
print(byt[:10])
type(byt)
```

    b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    bytes


```python
import base64
```


```python
base64_encoded_data = base64.b64encode(byt)
print(base64_encoded_data[:10])
type(base64_encoded_data)
```

    b'UEsDBBQAAA'

    bytes


```python
base64_message = base64_encoded_data.decode('utf-8')
```


```python
print(base64_message[:10])
type(base64_message)
```

    UEsDBBQAAA
    
    str

Line 3: We encode string, cast to byte object.

Line 5: We use the decode() method with utf8 encoding scheme to transform from encoded values to a string object.

Line 7: We print decoded values


```python
# String of encoded codes
# For word EDPRESSO
bytes= b'\x45\x44\x50\x52\x45\x53\x53\x4f'
# Using encoding scheme: UTF8
bytes= bytes.decode('utf8')
# Show results
print ("Decoded bytes: " + bytes)
```

    Decoded bytes: EDPRESSO


There different ways to encode the zip file

##  Converting zip files to bytes and encode


```python
import base64
with open("data.zip", 'rb') as f:
    data = f.read()
    print(type(data))
    #print(data)
    encoded = base64.b64encode(data)
```

    <class 'bytes'>


Let us print the first 10 characters of our zip data


```python
print('Undecoded zip data: ',data[:10])
```

    Undecoded zip data:  b'PK\x03\x04\x14\x00\x00\x00\x00\x00'

```python
print('Encoded zip data: ',encoded[:10])
```

    Encoded zip data:  b'UEsDBBQAAA'

```python
type(data)
```


    bytes


```python
type(encoded)
```


    bytes

##  Converting bytes to string - from encoding


```python
# Program for converting bytes to string using decode()
data =encoded
 
# display input
print('\nInput:')
print(data[:10])
print(type(data))
 
# converting
output = str(data, 'UTF-8')
 
# display output
print('\nOutput:')
print(output[:10])
print(type(output))
```


    Input:
    b'UEsDBBQAAA'
    <class 'bytes'>
    
    Output:
    UEsDBBQAAA
    <class 'str'>

```python
import pandas as pd
dic = {'encoded' : [encoded]}
df = pd.DataFrame(data=dic)
x = df['encoded'].str.decode("utf-8")
#df['decoded']=x
print(x)
```

    0    UEsDBBQAAAAAAAAAIQAZwvYbXhEAAF4RAAAkAAAAZmluZH...
    Name: encoded, dtype: object



```python
df.dtypes
```




    encoded    object
    dtype: object




```python
df.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b'UEsDBBQAAAAAAAAAIQAZwvYbXhEAAF4RAAAkAAAAZmlu...</td>
    </tr>
  </tbody>
</table>





```python
df.memory_usage(deep=True)
```




    Index       128
    encoded    6197
    dtype: int64




```python
df.info(memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1 entries, 0 to 0
    Data columns (total 1 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   encoded  1 non-null      object
    dtypes: object(1)
    memory usage: 6.2 KB



```python
df.to_csv('df_zip.csv')  
```

# Reading zipped dataframe


```python
df_new=pd.read_csv('df_zip.csv')  
```


```python
df_new.dtypes
```




    Unnamed: 0     int64
    encoded       object
    dtype: object




```python
df_new.memory_usage(deep=True)
```




    Index          128
    Unnamed: 0       8
    encoded       6216
    dtype: int64




```python
df_new.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>b'UEsDBBQAAAAAAAAAIQAZwvYbXhEAAF4RAAAkAAAAZmlu...</td>
    </tr>
  </tbody>
</table>






# Unzip zipped dataframe

## Encoded bytes to zip file 


```python
# String of encoded codes
string_bytes=df_new['encoded'][0]
```


```python
type(string_bytes)
```




    str




```python
size=len(string_bytes)
```


```python
string=string_bytes[2:size]
```


```python
string[:10]
```


    'UEsDBBQAAA'

We compare with the original


```python
string_bytes_original=df['encoded'].str.decode("utf-8")[0]
```


```python
type(str(string_bytes_original))
```


    str


```python
string_bytes_original[:10]
```


    'UEsDBBQAAA'



We decode our encoded string


```python
byte_decoded=base64.b64decode(string)
```


```python
bin_data=byte_decoded #Whatever binary data you have store in a variable
```


```python
binary_file_path = 'new_file.zip' #Name for new zip file you want to regenerate
with open(binary_file_path, 'wb') as f:
    f.write(bin_data)
```

checking our new_file.zip


```python
from zipfile import ZipFile
with ZipFile('new_file.zip') as zip_archive:
  for item in zip_archive.filelist:
    print(item)
  print(f'\nThere are {len(zip_archive.filelist)} ZipInfo objects present in archive')
```

    <ZipInfo filename='findspark-2.0.1-py2.py3-none-any.whl' filemode='?rw-------' file_size=4446>
    
    There are 1 ZipInfo objects present in archive

```python
if os.name == 'nt':
    print('I am Windows')
    !dir
else:
    print('I am on Unix')
    !ls -ltrh
```

    I am on Unix
    total 28K
    drwxrwx--- 2 wsuser wscommon 4.0K Feb 26 19:40 folder
    -rw-rw---- 1 wsuser wscommon 4.6K Feb 26 19:40 data.zip
    -rw-rw---- 1 wsuser wscommon 6.1K Feb 26 19:40 df_zip.csv
    -rw-rw---- 1 wsuser wscommon 4.6K Feb 26 19:40 new_file.zip

You can download this notebook [here](https://github.com/ruslanmv/How-to-convert-zip-to-dataframe/blob/master/Zip_to_Dataframe.ipynb).



**Congratulations!** We have practice how to add zip files into a Dataframe as a field.