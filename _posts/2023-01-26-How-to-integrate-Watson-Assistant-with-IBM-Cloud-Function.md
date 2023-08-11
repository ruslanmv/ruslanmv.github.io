---
title: "How to integrate Watson Assistant with IBM Cloud Function"
excerpt: "How to connect a Chatbot with Serverless Function"

header:
  image: "../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/kvistholt-photography-oZPwn40zCK4-unsplash.jpg"
  teaser: "../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/kvistholt-photography-oZPwn40zCK4-unsplash.jpg"
  caption: "The cloud services companies of all sizes…The cloud is for everyone. The cloud is a democracy.– Marc Benioff, CEO"
  
---

Hello everyone, today we are going to integrate Watson Assistant by using IBM Cloud functions
This is very interesting way to give power to you Chatbot with a fully serveless system in IBM Cloud. We are interested to  connect a Chatbot with Serverless Function.



## Step 1 - Login to your IBM Cloud.

First you need to login to your IBM Cloud here 
[https://cloud.ibm.com](https://cloud.ibm.com)

## Step 2 - Create an Cloud Function

Type functions in the search bar type `function`
![](../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230804171027.png)

Then you select Start Creating 
<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230804174536.png" style="zoom:50%;" />then you create Namespace, then  you select Actions 

and click **Create**
![](../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230804171404.png)
then you have created your serverless
For this example, instead use node.js we use Python
![](../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230804171511.png)

we copy the following code and replace the current one

```
import sys
import requests
import json
def main(params):
    # Used to identify the specific task being called from Watson Assistant
    # URL used for API call
    print(params)
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + str(params['object_of_interest'])+ "?redirect=true"
    # Set headers
    headers = {'accept': 'application/json'}
    # Make API call
    r = requests.get(url,headers)
    # Process failed API call
    if r.status_code != 200:
        return {
            'statusCode': r.status_code,
            'headers': { 'Content-Type': 'application/json'},
            'body': {'message': 'Error processing your request'}
        }
    # Process successful API call 
    else:
        res = json.loads(r.content)
        extract = res['extract']
        return {
            'statusCode': 200,
            'headers': { 'Content-Type': 'application/json'},
            'extract': {"extract":extract}
        }

```

to test we can click on Invoke with parameters

```
{
    "object_of_interest":"gravity"
  }
```

and then we click `Invoke`

we got the following
![](../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807103416.png)


## Step 4 - Enable Endpoint

Before create API we go to Endpoints and  we click on `Enable as Web Action` and click `save`
![](../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230804173610.png)
and then we copy the HTTP method

![](../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807091607.png)

now it works our cloud function.

## Step 5 - Call Cloud Function from Watson Assistant

In the menu of Watson Assistant we select the tab `Assistant Settings`
we scroll down and we click `Activate dialog`
![](../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230805012230.png)
At the Home menu we click `Dialog`
![](../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230805012439.png)

#Step 5 - Create Entity
We create am entitiy
<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807092400.png" style="zoom:33%;" />`@object_of_interest`

#Step 6 - Create Intent
Then we create an intent,
<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807091453.png" style="zoom:50%;" />,
for example we type
for the Intent name
`#tell_me_about`
and in the user example we add different queries that countains the **nouns** that you will look on, or will be used as arguments.
eg. `what is the definition of gravity`
then you e  click on `Annotate entities` and click with the mouse gravity
<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807092127.png" style="zoom:50%;" />and click the `@object_of_interest`
<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807092550.png" style="zoom:50%;" />you will got something like
<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807092708.png" style="zoom:75%;" />

# Step 6 - Weebhook setup

We paste our edpoint copied here,
and we add the extension **.json** at the end of the URL.
<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807093343.png" style="zoom:75%;" />Here we do not need add, extra Header, or autorization.

## Step 7- Create Dialog

We go the menu Dialog and we `Create dialog`
<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807095849.png" style="zoom:33%;" />

We click on `Add node` and in the
section If assistant recognizes, we add the condition
`#tell_me_about` and `@object_of_interest`

Then in the part of **Parameters**
we add the following parameters

then we click on **Customize**

<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807100151.png" style="zoom:50%;" />and we enable `Call out to webhook/actions` and select `Call a webhook` click apply
<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807100536.png" style="zoom:50%;" />in the part of Assistat responds you will have 

**If assistant recognizes**:

```
$webhook_result_1
```

the you click the gear button of 
**Respond with** and you add the following in the Text responds

```
I am defining @object_of_interest:<? $webhook_result_1.extract.extract ?>
```

and click save

<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807101635.png" style="zoom:50%;" />then you will get something like

<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230807100847.png" style="zoom:50%;" />

with 
![](../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230811093326.png)
Now finally we can try it and  you can ask questions

<img src="../assets/images/posts/2023-01-26-How-to-integrate-Watson-Assistant-with-IBM-Cloud-Function/20230811093024.png" style="zoom:50%;" />

**Congratulations!** You have created a  Chatbot with Serverless  Cloud Function.