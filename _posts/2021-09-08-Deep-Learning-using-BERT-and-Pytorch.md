---
title: "Sentiment analysis of a Twitter dataset with BERT and Pytorch "
excerpt: "Deep Learning using BERT and Pytorch"

header:
  image: "../assets/images/posts/2021-09-08-Deep-Learning-using-BERT-and-Pytorch/computer3.jpg"
  teaser: "../assets/images/posts/2021-09-08-Deep-Learning-using-BERT-and-Pytorch/computer2.jpg"
  caption: "Twitter Splash screen @ Joshua Hoehne"
  
---

In this blog post, we are going to build a sentiment analysis of a Twitter dataset that uses BERT by using Python with Pytorch with Anaconda.

### What is BERT

BERT is a large-scale transformer-based Language Model that can be finetuned for a variety of tasks.

For more information, the original paper can be found [here](https://arxiv.org/abs/1810.04805).  [HuggingFace documentation](https://huggingface.co/transformers/model_doc/bert.html)

[Bert documentation](https://characters.fandom.com/wiki/Bert_(Sesame_Street) ;)

First we are going to setup the  python environment with anaconda.



###  Install Anaconda

The first step is to install Anaconda such that you can create different environments for different applications. Note the different applications may require different libraries. For example, some may require OpenCV 3 and some require OpenCV 4. So, it is better to create different environments for different applications.

Please click [[here](https://www.anaconda.com/products/individual#windows)] to go to the official website of Anaconda. Then click “**Download**” as shown below.



![img](../assets/images/posts/2021-09-08-Deep-Learning-using-BERT-and-Pytorch/1hxRmBAEX5Larw4KMlbadkA-16303239464182.png)

Figure 1. Official website of Anaconda. [[here](https://www.anaconda.com/products/individual#windows)] 

Select the installer based on your OS. Assume that your OS is Windows 10 64-Bit.

<img src="../assets/images/posts/2021-09-08-Deep-Learning-using-BERT-and-Pytorch/ana1.jpg" style="zoom:50%;" />



Figure 2. Anaconda Installers selection page from the official website of Anaconda. Captured from [[here](https://www.anaconda.com/products/individual#windows)] by author

Start to download the EXE of the installer and then follow the instructions to install Anaconda to your OS. Detailed instructions with screen captures are available at [[here](https://docs.anaconda.com/anaconda/install/windows/)].

###  Install CUDA Toolkit (if you have GPU(s))

If you have GPU(s) on your computer and you want to use GPU(s) to speed up your applications, you have to install CUDA Toolkit. Please download CUDA Toolkit [[here](https://developer.nvidia.com/cuda-downloads)].

Select your Operating System, Architecture, Version, and Installer Type as shown below.



<img src="../assets/images/posts/2021-09-08-Deep-Learning-using-BERT-and-Pytorch/cuda1.jpg" style="zoom:70%;" />

Figure 3. Select Installer for CUDA Toolkit 11.1. Captured from [[here](https://developer.nvidia.com/cuda-downloads)] 

Click the “**Download**” button as shown in Figure 3 above and then install the CUDA Toolkit. The newest version of CUDA Toolkit is 11.1 at the time of writing this installation guide. Note that you have to check which GPU you are using and which version of CUDA Toolkit is applicable.

### Create Conda environment for PyTorch

If you have finished Step 1 and 2, you have successfully installed Anaconda and CUDA Toolkit to your OS.

Please open your Command Prompt by searching ‘cmd’ as shown below.

By typing this line, you are creating a Conda environment called ‘bert’

```
conda create --name bert python=3.7
```

```
conda install ipykernel
```

```
python -m ipykernel install --user --name bert --display-name "Python (Bert)"
```

```
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

```
pip install torch
```

```
pip install pandas
```

```
pip install tqd
```

```
pip install scipy
```

```
pip install joblib
```

```
pip install transformers
```

```
pip install ipywidgets
```

If youw are interested to use images in your project you can  install OpenCV library for image pre/post-processing

```
conda install -c conda-forge opencv
```

and nstall Pillow library for reading and writing images

```
conda install -c anaconda pillow
```

later we can create the a folder and there we activate our enviroment

```
conda activate bert
```

```
jupyter notebook&
```

and we create the a jupyter notebook

##  Exploratory Data Analysis and Preprocessing

We will use the SMILE Twitter dataset.

_Wang, Bo; Tsakalidis, Adam; Liakata, Maria; Zubiaga, Arkaitz; Procter, Rob; Jensen, Eric (2016): SMILE Twitter Emotion dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.3187909.v2_

we create a folder called  data you can download [here](https://github.com/ruslanmv/Deep-Learning-using-BERT/blob/main/Data/smile-annotations-final.csv)

```
mkdir data
```

we can download the dataset by using bash with wget command

```
bash
```

```
wget https://github.com/ruslanmv/Deep-Learning-using-BERT/raw/main/Data/smile-annotations-final.csv
```

```
exit
```

then in the Jupiter notebook we can type the following:


```python
import torch
import pandas as pd
from tqdm.notebook import trange, tqdm
```


```python
# TDQ is a A Fast, Extensible Progress Bar for Python and CLI
```


```python

for i in trange(10):
    print(i)
```


      0%|          | 0/10 [00:00<?, ?it/s]


    0
    1
    2
    3
    4
    5
    6
    7
    8
    9

We check if the GPUs are available

```python
torch.cuda.is_available()
```




    True




```python
df = pd.read_csv('Data/smile-annotations-final.csv', 
                 names =['id', 'text', 'category'  ])

```


```python
df.set_index('id', inplace=True)
```


```python
df.text.iloc[0]
```




    '@aandraous @britishmuseum @AndrewsAntonio Merci pour le partage! @openwinemap'




```python
df.category.value_counts()
```




    nocode               1572
    happy                1137
    not-relevant          214
    angry                  57
    surprise               35
    sad                    32
    happy|surprise         11
    happy|sad               9
    disgust|angry           7
    disgust                 6
    sad|disgust             2
    sad|angry               2
    sad|disgust|angry       1
    Name: category, dtype: int64




```python
df = df[~df.category.str.contains('\|')]
```


```python
df = df[df.category!= 'nocode']
```


```python
df.category.value_counts()
```




    happy           1137
    not-relevant     214
    angry             57
    surprise          35
    sad               32
    disgust            6
    Name: category, dtype: int64



It is unbalanced


```python
possible_labels = df.category.unique()
```


```python
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label]= index
```


```python
label_dict
```




    {'happy': 0,
     'not-relevant': 1,
     'angry': 2,
     'disgust': 3,
     'sad': 4,
     'surprise': 5}




```python
df['label'] = df.category.replace(label_dict)
df.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
      <th>label</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>614484565059596288</th>
      <td>Dorian Gray with Rainbow Scarf #LoveWins (from...</td>
      <td>happy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>614746522043973632</th>
      <td>@SelectShowcase @Tate_StIves ... Replace with ...</td>
      <td>happy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>614877582664835073</th>
      <td>@Sofabsports thank you for following me back. ...</td>
      <td>happy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>611932373039644672</th>
      <td>@britishmuseum @TudorHistory What a beautiful ...</td>
      <td>happy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>611570404268883969</th>
      <td>@NationalGallery @ThePoldarkian I have always ...</td>
      <td>happy</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
df['text'].iloc[0]
```




    'Dorian Gray with Rainbow Scarf #LoveWins (from @britishmuseum http://t.co/Q4XSwL0esu) http://t.co/h0evbTBWRq'

## Training/Validation Split


```python
from sklearn.model_selection import train_test_split
```


```python
df.index.values
```




    array([614484565059596288, 614746522043973632, 614877582664835073, ...,
           613678555935973376, 615246897670922240, 613016084371914753],
          dtype=int64)


```python
df.label.values
```


    array([0, 0, 0, ..., 0, 0, 1], dtype=int64)


```python
X_train, X_val, y_train, y_val = train_test_split(
    df.index.values,
    df.label.values,
    test_size=0.15,
    random_state=17,
    stratify=df.label.values
)
```


```python
X_train
```




    array([614767094345936896, 610755488372948992, 610609791073931266, ...,
           613744184495894529, 610873494910443520, 610741907426267136],
          dtype=int64)




```python
y_train
```




    array([0, 0, 0, ..., 0, 0, 2], dtype=int64)




```python
df['data_type']= ['no_set']*df.shape[0]
```


```python
X_train
```




    array([614767094345936896, 610755488372948992, 610609791073931266, ...,
           613744184495894529, 610873494910443520, 610741907426267136],
          dtype=int64)




```python
df.loc[X_train, 'data_type']='train'
df.loc[X_val, 'data_type']='val'
```


```python
df.groupby(['category','label','data_type']).count()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>text</th>
    </tr>
    <tr>
      <th>category</th>
      <th>label</th>
      <th>data_type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">angry</th>
      <th rowspan="2" valign="top">2</th>
      <th>train</th>
      <td>48</td>
    </tr>
    <tr>
      <th>val</th>
      <td>9</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">disgust</th>
      <th rowspan="2" valign="top">3</th>
      <th>train</th>
      <td>5</td>
    </tr>
    <tr>
      <th>val</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">happy</th>
      <th rowspan="2" valign="top">0</th>
      <th>train</th>
      <td>966</td>
    </tr>
    <tr>
      <th>val</th>
      <td>171</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">not-relevant</th>
      <th rowspan="2" valign="top">1</th>
      <th>train</th>
      <td>182</td>
    </tr>
    <tr>
      <th>val</th>
      <td>32</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sad</th>
      <th rowspan="2" valign="top">4</th>
      <th>train</th>
      <td>27</td>
    </tr>
    <tr>
      <th>val</th>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">surprise</th>
      <th rowspan="2" valign="top">5</th>
      <th>train</th>
      <td>30</td>
    </tr>
    <tr>
      <th>val</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
##  Loading Tokenizer and Encoding our Data


```python
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
```


```python
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)
```


```python
df.data_type=='train'
```




    id
    614484565059596288    True
    614746522043973632    True
    614877582664835073    True
    611932373039644672    True
    611570404268883969    True
                          ... 
    611258135270060033    True
    612214539468279808    True
    613678555935973376    True
    615246897670922240    True
    613016084371914753    True
    Name: data_type, Length: 1481, dtype: bool


```python
df[df.data_type=='train']
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
      <th>label</th>
      <th>data_type</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>614484565059596288</th>
      <td>Dorian Gray with Rainbow Scarf #LoveWins (from...</td>
      <td>happy</td>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>614746522043973632</th>
      <td>@SelectShowcase @Tate_StIves ... Replace with ...</td>
      <td>happy</td>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>614877582664835073</th>
      <td>@Sofabsports thank you for following me back. ...</td>
      <td>happy</td>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>611932373039644672</th>
      <td>@britishmuseum @TudorHistory What a beautiful ...</td>
      <td>happy</td>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>611570404268883969</th>
      <td>@NationalGallery @ThePoldarkian I have always ...</td>
      <td>happy</td>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>611258135270060033</th>
      <td>@_TheWhitechapel @Campaignforwool @SlowTextile...</td>
      <td>not-relevant</td>
      <td>1</td>
      <td>train</td>
    </tr>
    <tr>
      <th>612214539468279808</th>
      <td>“@britishmuseum: Thanks for ranking us #1 in @...</td>
      <td>happy</td>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>613678555935973376</th>
      <td>MT @AliHaggett: Looking forward to our public ...</td>
      <td>happy</td>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>615246897670922240</th>
      <td>@MrStuchbery @britishmuseum Mesmerising.</td>
      <td>happy</td>
      <td>0</td>
      <td>train</td>
    </tr>
    <tr>
      <th>613016084371914753</th>
      <td>@NationalGallery The 2nd GENOCIDE against #Bia...</td>
      <td>not-relevant</td>
      <td>1</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
<p>1258 rows × 4 columns</p>



```python
df[df.data_type=='train'].text.values
```

We have to encode the texts by using [tokenizer.batch_encode_plus](https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus)


```python
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    #pad_to_max_length=True,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors='pt'
)
```


```python
encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
   # pad_to_max_length=True,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors='pt'
)
```


```python
 
```

For the train


```python
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values) 
```

For the validation


```python
input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values) 
```

It is created the TensorDataset adapted to Bert for the train and validation


```python
dataset_train = TensorDataset(
    input_ids_train,
    attention_masks_train,
    labels_train
)
```


```python
dataset_val = TensorDataset(input_ids_val,
                            attention_masks_val,
                            labels_val
)
```


```python
len(dataset_train)
```


    1258


```python
len(dataset_val)
```


    223

## Setting up BERT Pretrained Model


```python
from transformers import BertForSequenceClassification
```


```python
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
     num_labels=len(label_dict),
     output_attentions=False,
     output_hidden_states=False)
```

## Creating Data Loaders


```python
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
```


```python
#In Google Colab -- GPU Instance (k80)
#batch_size =32
#epoch =10
```


```python
batch_size = 4 #32
dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size=batch_size
)


dataloader_val = DataLoader(
    dataset_val,
    sampler=SequentialSampler(dataset_val),
    batch_size=batch_size 
)
```

## Setting Up Optimizer and Scheduler


```python
from transformers import AdamW, get_linear_schedule_with_warmup
```


```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-5, #2e-5 > 5e-5
    eps=1e-8
)
```


```python
epochs = 10

scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train)*epochs
)
```

## Defining our Performance Metrics

Accuracy metric approach originally used in accuracy function in [this tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/#41-bertforsequenceclassification).


```python
import numpy as np
```


```python
from sklearn.metrics import f1_score
```


```python
#preds=[0.9 0.05 0.05 0 0 0]
#preds = [1 0 0 0 0]
```


```python
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis =1 ).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')
```


```python
def accuracy_per_class(preds, labels):
    label_dict_inverse={v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis =1 ).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_pred = preds_flat[labels_flat== label]
        y_true = labels_flat[labels_flat== label]
        print(f'Class:{label_dict_inverse[label]}')
        print(f'Accuracy:{len(y_pred[y_pred==label])}/{len(y_true)}\n')
```

##  Creating our Training Loop

Approach adapted from an older version of HuggingFace's `run_glue.py` script. Accessible [here](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128).


```python
import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)
```

    cuda


Assuming valX is a tensor with the complete validation data, 
The usual approach would be to wrap it in a Dataset and DataLoader and get the predictions for each batch. 

Also, to save memory during evaluation and test, you could wrap the validation and test code into a with torch.no_grad() block.

 for evaluation and test set the code should be:
```python

with torch.no_grad():
    model.eval()
    y_pred = model(valX)
    val_loss = criterion(y_pred, valY)
```

and
```python

with torch.no_grad():
    model.eval()
    y_pred = model(test)
    test_loss = criterion(y_pred, testY)
```


```python
def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in tqdm(dataloader_val):
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

```


```python
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0
    
    progress_bar = tqdm(dataloader_train, 
                        desc='Epoch {:1d}'.format(epoch),
                        leave=False,
                        disable=False)
    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs ={
            'input_ids'    :batch[0],
            'attention_mask':batch[1],
            'labels'        :batch[2]
        }
        outputs = model(**inputs)
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
    
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix(
            {'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        
    #torch.save(model.state_dict(),f'Models/BERT_ft_epoch{epoch}.model')
    tqdm.write('\nEpoch {epoch}')
    
    loss_train_avg= loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss:{loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1= f1_score_func(predictions,true_vals)
    tqdm.write(f'Validation{val_loss}')
    tqdm.write(f'F1 Score (weigthed): {val_f1}')
torch.save(model.state_dict(),f'Models/BERT_ft_epoch{epoch}.model')       
```


      0%|          | 0/10 [00:00<?, ?it/s]



    Epoch 1:   0%|          | 0/315 [00:00<?, ?it/s]


​    
​    Epoch {epoch}
​    Training loss:0.7975574296973055



      0%|          | 0/56 [00:00<?, ?it/s]


    Validation0.6059155837699238
    F1 Score (weigthed): 0.762975916339145



    Epoch 2:   0%|          | 0/315 [00:00<?, ?it/s]


​    
​    Epoch {epoch}
​    Training loss:0.4435813750036889



      0%|          | 0/56 [00:00<?, ?it/s]


    Validation0.5948948868151221
    F1 Score (weigthed): 0.8381931883679935



    Epoch 3:   0%|          | 0/315 [00:00<?, ?it/s]


​    
​    Epoch {epoch}
​    Training loss:0.2983576275445225



      0%|          | 0/56 [00:00<?, ?it/s]


    Validation0.5538139296175879
    F1 Score (weigthed): 0.8431542233760785



    Epoch 4:   0%|          | 0/315 [00:00<?, ?it/s]


​    
​    Epoch {epoch}
​    Training loss:0.18270550284845133



      0%|          | 0/56 [00:00<?, ?it/s]


    Validation0.5259785884367635
    F1 Score (weigthed): 0.8662434969638529



    Epoch 5:   0%|          | 0/315 [00:00<?, ?it/s]


​    
​    Epoch {epoch}
​    Training loss:0.11218177344158499



      0%|          | 0/56 [00:00<?, ?it/s]


    Validation0.6620622687145702
    F1 Score (weigthed): 0.8535941751228626



    Epoch 6:   0%|          | 0/315 [00:00<?, ?it/s]


​    
​    Epoch {epoch}
​    Training loss:0.05948421741458809



      0%|          | 0/56 [00:00<?, ?it/s]


    Validation0.6958074946513599
    F1 Score (weigthed): 0.8637082897172584



    Epoch 7:   0%|          | 0/315 [00:00<?, ?it/s]


​    
​    Epoch {epoch}
​    Training loss:0.04381906234674037



      0%|          | 0/56 [00:00<?, ?it/s]


    Validation0.7028247105024222
    F1 Score (weigthed): 0.8623919844487555



    Epoch 8:   0%|          | 0/315 [00:00<?, ?it/s]


​    
​    Epoch {epoch}
​    Training loss:0.028735312574546753



      0%|          | 0/56 [00:00<?, ?it/s]


    Validation0.7281462288156035
    F1 Score (weigthed): 0.8645974679453454



    Epoch 9:   0%|          | 0/315 [00:00<?, ?it/s]


​    
​    Epoch {epoch}
​    Training loss:0.02450285246531065



      0%|          | 0/56 [00:00<?, ?it/s]


    Validation0.7339509774070133
    F1 Score (weigthed): 0.8674038975615305



    Epoch 10:   0%|          | 0/315 [00:00<?, ?it/s]


​    
​    Epoch {epoch}
​    Training loss:0.021128401538405183



      0%|          | 0/56 [00:00<?, ?it/s]


    Validation0.7412530884700702
    F1 Score (weigthed): 0.8648362667790782


When saving a general checkpoint, to be used for either inference or resuming training, you must save more than just the model’s state_dict. It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are updated as the model trains. Other items that you may want to save are the epoch you left off on, the latest recorded training loss, external torch.nn.Embedding layers, etc. As a result, such a checkpoint is often 2~3 times larger than the model alone.

To save multiple components, organize them in a dictionary and use torch.save() to serialize the dictionary. A common PyTorch convention is to save these checkpoints using the .tar file extension.

##  Loading and Evaluating our Model


```python
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False)
```


When we are loading the bert-base-cased checkpoint (which is a checkpoint that was trained using a similar architecture to BertForPreTraining) in a BertForSequenceClassification model.

This means that:

The layers that BertForPreTraining has, but BertForSequenceClassification does not have will be discarded
The layers that BertForSequenceClassification has but BertForPreTraining does not have will be randomly initialized.
This is expected, and tells you that you won't have good performance with your BertForSequenceClassification model before you fine-tune it 🙂.

This warning means that during your training, you're not using the pooler in order to compute the loss. I don't know how you're finetuning your model, but if you're not using the pooler layer then there's no need to worry about that warning.


```python
len(label_dict)
```


    6

In PyTorch, the learnable parameters (i.e. weights and biases) of an torch.nn.Module model are contained in the model’s parameters (accessed with model.parameters()). A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.


```python
# Print model's state_dict
#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
```


```python
# Print optimizer's state_dict
#print("Optimizer's state_dict:")
#for var_name in optimizer.state_dict():
#    print(var_name, "\t", optimizer.state_dict()[var_name])
```


```python
device = torch.device('cuda')
pass
```


```python
model.to(device)
pass
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```


```python
PATH='./Models/BERT_ft_epoch10.model'
```


```python
model.load_state_dict(torch.load(PATH, 
                                 map_location=torch.device('cuda:0')))
```


    <All keys matched successfully>

When loading a model on a GPU that was trained and saved on GPU, simply convert the initialized model to a CUDA optimized model using model.to(torch.device('cuda')). Also, be sure to use the .to(torch.device('cuda')) function on all model inputs to prepare the data for the model. Note that calling my_tensor.to(device) returns a new copy of my_tensor on GPU. It does NOT overwrite my_tensor. Therefore, remember to manually overwrite tensors: my_tensor = my_tensor.to(torch.device('cuda')).


```python
_, predictions, true_vals = evaluate(dataloader_val)
```


      0%|          | 0/56 [00:00<?, ?it/s]

```python
accuracy_per_class(predictions, true_vals)
```

    Class:happy
    Accuracy:161/171
    
    Class:not-relevant
    Accuracy:20/32
    
    Class:angry
    Accuracy:8/9
    
    Class:disgust
    Accuracy:0/1
    
    Class:sad
    Accuracy:2/5
    
    Class:surprise
    Accuracy:2/5

 You can download this notebook [here](https://github.com/ruslanmv/Deep-Learning-using-BERT/blob/main/Deep-Learning-using-BERT.ipynb)

**Congratulations!** we were able to build a **Deep Learning** with **Pytorch** and **BERT.**

