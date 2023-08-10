---
title: "How to Connect Watson Assistant with WatsonX and Watson Discovery"
excerpt: "How do I connect my Watson assistant to WatsonX and Watson Discovery?"

header:
  image: "../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/ai.png"
  teaser: "../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/ai.png"
  caption: "A year spent in artificial intelligence is enough to make one believe in God. —Alan Perlis"
  
---

Today we are going to setup **Watson Assistant** with **Watson Discovery** and **WatsonX.**
We are going to use the standard toolkit of **IBM Developer.**

![im-778762](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/im-778762.png)

**Generative artificial intelligence** is artificial intelligence capable of generating text, images, or other media. Generative AI models learn the patterns and structure of their input training data and then generate new data that has similar characteristics

# Step 1 - Create services in IBM Cloud.

First we need to login to IBM Cloud

[https://cloud.ibm.com/](https://cloud.ibm.com/)

We need to create the following services

1. [WatsonX](https://www.ibm.com/watsonx?)
2. [Watson Discovery](https://www.ibm.com/products/watson-discovery)
3. [Watson Assistant](https://cloud.ibm.com/catalog/services/watson-assistant)

After you have created those services, we have to setup them.

# Step 2. Setup of WatsonX

After you have created your instance in WatsonX
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808211802.png)
we open a simple prompt lab
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808212344.png)

we click over view code
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808212502.png)
and click create personal API key
and we create
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808213210.png)
and we download the `API key` that we will use.

Then we return back to our Prompt Lab and in the view code we copy our 
`project_id`.
So we should have the following two numbers, for exaple:

1. **API Key**
2. **project_id**

# Step 3 - Setup of Watson Discovery

In the menu of IBM cloud we go to resource list, and in the section of
`AI / Machine Learning`

![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808214104.png)
we go click to Watson Discovery service and then
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808214243.png)
we have two additional numbers to conserve here

1. **API Key**
2. **URL**

Notice that this API key is different to our previous case. This is for this service.

Now lunch this service and we create New project
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808214539.png)

In this example we are goint to analize Bitcoins, so the project will be named `Bitcoin`
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808214617.png)
then we select`Conversational Search`
we have different options to retreive the data, we choose `Web crawl` and click `Next`
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808214827.png)
We choose a crawling each month

![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808215007.png)

and we chose the url to crawl
https://bitcoin.org/en/faq
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808215030.png)
and finally click finish.
After few minutes. You can get sometihng like
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808215134.png)
then you go to your menu and click `Integrate and Deploy`and you will get another `project_id`
<img src="../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808215249.png" style="zoom:75%;" />so we will have our third important number to keeep for the Watson Assistant.

3. **project_id**

# Step 4 - Setup Watson Assistant.

Let us return back to our IBM cloud and there in resource list, let lunch Watson Assistant and create a new assitant
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808220116.png) and you will get something like

![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808220241.png)

then in the menu click integrations, we will install two integrations.
Click on `Build extensions`
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808220353.png)

1. Watson Discovery Extension
   First, we name it as Watson Discovery Extension
   ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808220502.png)
   then you have to download the openapi of this extension here
     [https://github.com/ruslanmv/How-to-create-a-Chatbot-with-WatsonX-and-Watson-Discovery/blob/master/watson-discovery-query-openapi.json](https://github.com/ruslanmv/How-to-create-a-Chatbot-with-WatsonX-and-Watson-Discovery/blob/master/watson-discovery-query-openapi.json)

  ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808221657.png)
  and then click Finish

  ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808221723.png)
  then we click add and then we paste our **API Key** and **URL** from Watson Discovery setup.
 <img src="../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808222046.png" style="zoom:75%;" /> and click finish
 ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808222241.png)


2. WatsonX Extension
   We repeat the same, we go to Extensions and Build a new custom extension

   ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808222348.png)
   which we call WatsonX extension

   ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808222420.png)
   then we download the openapi of WatsonX from here
   [https://github.com/ruslanmv/How-to-create-a-Chatbot-with-WatsonX-and-Watson-Discovery/raw/master/watson-discovery-query-openapi.json](https://github.com/ruslanmv/How-to-create-a-Chatbot-with-WatsonX-and-Watson-Discovery/raw/master/watson-discovery-query-openapi.json)


   ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808222534.png)
   and we click Finish

   ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808222949.png)
   On the Extensions we click `Add`

   ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808223023.png)
   we paste our API Key that was created in the WatsonX setup
   ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808223206.png)
   we click `Next` and then `Finish`

   ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808223234.png)

In the Integrations menu you should have the active two integrations

![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808223324.png)
with the word `Open`

# Step 5 - Creation of Applications fo Watson Assitant.

In this part we are going to build some standard applications.
Go to `Actions` then Click on the button `Global Settings`
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808223632.png)
On the menu, go to the last tab called `Upload/Download`

![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808223735.png)
and then download this file from here
[https://github.com/ruslanmv/How-to-create-a-Chatbot-with-WatsonX-and-Watson-Discovery/raw/master/discovery-watsonx-actions.json](https://github.com/ruslanmv/How-to-create-a-Chatbot-with-WatsonX-and-Watson-Discovery/raw/master/discovery-watsonx-actions.json)

and upload it
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808223806.png)


click upload and replace

![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808224016.png)
go to  `Variables ` then in the tab `Created by You`
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808224204.png)
 You have to edit the following variables:

 1. **discovery_project_id** -  From the Discovery Setup  you can copy the Project ID
    ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808224359.png)

    <img src="../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808224428.png" style="zoom:75%;" />
    2.**watsonx_project_id** - From you WatsonX Setup you copy your **project_id**
    ![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808224624.png)

After you have updated all variables. You can test by click `Try`
and you can ask like
`How are bitcoin transactions?`

![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808231106.png)
`What is bitcoin?`

![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808231150.png)
and finally you can click in preview
and give some questions.
![](../assets/images/posts/2023-08-07-How-to-connect-Watson-Assistant-to-WatsonX-and-Watson-Discovery/20230808231527.png)

**Congratulations** You have created a simple chatbot with Watson Assistant with WatsonX and Watson Discovery.

