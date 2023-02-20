---
title: "GCP Machine Learning Practice Questions"
excerpt: "Machine Learning questions in Google Cloud Platform."

header:
  image: "../assets/images/posts/2020-07-29-GCP-Machine-Learning-Practice-Questions/daniel-korpai-HyTwtsk8XqA-unsplash.jpg"
  teaser: "../assets/images/posts/2020-07-29-GCP-Machine-Learning-Practice-Questions/daniel-korpai-HyTwtsk8XqA-unsplash.jpg"
  caption: "Data scientists are involved with gathering data, massaging it into a tractable form, making it tell its story, and presenting that story to others. – Mike Loukides"

---

Hello everyone, today we are going to practice some  Machine Learning  Questions by using Google Cloud Platform.
In the following sections we are going to answer several questions that are important in Data Science projects in the IT industry by using Google Cloud Platform.

Google Cloud Platform, offered by Google, is a suite of cloud computing services that runs on the same infrastructure that Google uses internally for its end-user products, such as Google Search, Gmail, Google Drive, and YouTube.

We have spited the questions and answers into six parts:[ 1 ](#part-1),   [ 2 ](#part-2),   [ 3 ](#part-3),   [ 4 ](#part-4),   [ 5 ](#part-5)

# Part 1

<iframe src="https://player.rss.com/ruslanmv/833524" style="width: 100%" title="Machine Learning and Data Science - Podcast" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen><a href="https://rss.com/podcasts/ruslanmv/833524/">Machine Learning in GCP - Part 1 | RSS.com</a></iframe>
## Question 1.

You are experimenting with a built-in distributed XGBoost model in Vertex AI Workbench user-managed notebooks. You use BigQuery to split your data into training and validation sets. After training the model, you achieve an area under the receiver operating characteristic curve (AUC ROC)
value of 0.8, but after deploying the model to production, you notice that your model performance has dropped to an AUC ROC value of 0.65. What problem is most likely occurring?

### Answer:
There is training-serving skew in your production environment.

## Question 2.
You work for a biotech startup that is experimenting with deep learning ML models based on properties of
biological organisms. Your team frequently works on early-stage experiments with new architectures of ML
models, and writes custom TensorFlow ops in C++. You train your models on large datasets and large batch
sizes. Your typical batch size has 1024 examples, and each example is about 1 MB in size. The average size of
a network with all weights and embeddings is 20 GB. What hardware should you choose for your models?

### Answer
A cluster with 2 a2-megagpu-16g machines, each with 16 NVIDIA Tesla A100 GPUs (640 GB GPU
memory in total), 96 vCPUs, and 1.4 TB RAM

## Question 3.
You are an ML engineer at a global car manufacturer. You need to build an ML model to predict car sales in
different cities around the world. Which features or feature crosses should you use to train city-specific
relationships between car type and number of sales?

### Answer:

One feature obtained as an element-wise product between binned latitude, binned longitude, and one-hot
encoded car type

## Question 4.
You work on a growing team of more than 50 data scientists who all use AI Platform. You are designing a
strategy to organize your jobs, models, and versions in a clean and scalable way. Which strategy should you
choose?

### Answer:
Use labels to organize resources into descriptive categories. Apply a label to each created resource so
that users can filter the results by label when viewing or monitoring the resources.



## Question 4b.
You are training a deep learning model for semantic image segmentation with reduced training time. While
using a Deep Learning VM Image, you receive the following error: The resource
'projects/deeplearning-platforn/zones/europe-west4-c/acceleratorTypes/nvidia-tesla-k80' was not found. What
should you do?
### Answer:
Ensure that you have GPU quota in the selected region.


## Question 5.
Your team has been tasked with creating an ML solution in Google Cloud to classify support requests for one
of your platforms. You analyzed the requirements and decided to use TensorFlow to build the classifier so that
you have full control of the model's code, serving, and deployment. You will use Kubeflow pipelines for the
ML platform. To save time, you want to build on existing resources and use managed services instead of
building a completely new model. How should you build the classifier?
### Answer:
Use an established text classification model on Al Platform to perform transfer learning
Explanation
the model cannot work as-is as the classes to predict will likely not be the same; we need to use transfer
learning to retrain the last layer and adapt it to the classes we need

## Question  6.

You work for a gaming company that has millions of customers around the world. All games offer a chat
feature that allows players to communicate with each other in real time. Messages can be typed in more than
20 languages and are translated in real time using the Cloud Translation API. You have been asked to build an
ML system to moderate the chat in real time while assuring that the performance is uniform across the various
languages and without changing the serving infrastructure.
### Answer:

Remove moderation for languages for which the false positive rate is too high.
## Question 7.
You work for a company that is developing a new video streaming platform. You have been asked to create a
recommendation system that will suggest the next video for a user to watch. After a review by an AI Ethics
team, you are approved to start development. Each video asset in your company’s catalog has useful metadata
(e.g., content type, release date, country), but you do not have any historical user event data. How should you
build the recommendation system for the first version of the product?

### Answer:
Launch the product with machine learning. Use a publicly available dataset such as MovieLens to train a
model using the Recommendations AI, and then apply this trained model to your data.

## Question 8
You are developing an ML model to predict house prices. While preparing the data, you discover that an
important predictor variable, distance from the closest school, is often missing and does not have high
variance. Every instance (row) in your data is important. How should you handle the missing data?

Predict the missing values using linear regression.

## Question 9.
You built a custom ML model using scikit-learn. Training time is taking longer than expected. You decide to
migrate your model to Vertex AI Training, and you want to improve the model’s training time. What should
you try out first?
### Answer:
Train your model with DLVM images on Vertex AI, and ensure that your code utilizes NumPy and
SciPy internal methods whenever possible

## Question 10.
You manage a team of data scientists who use a cloud-based backend system to submit training jobs. This
system has become very difficult to administer, and you want to use a managed service instead. The data
scientists you work with use many different frameworks, including Keras, PyTorch, theano, scikit-learn, and
custom libraries. What should you do?
### Answer:
Set up Slurm workload manager to receive jobs that can be scheduled to run on your cloud
infrastructure.



## Question 11.
You lead a data science team at a large international corporation. Most of the models your team trains are
large-scale models using high-level TensorFlow APIs on AI Platform with GPUs. Your team usually
takes a few weeks or months to iterate on a new version of a model. You were recently asked to review your
team’s spending. How should you reduce your Google Cloud compute costs without impacting the model’s
performance?
### Answer:
Migrate to training with Kuberflow on Google Kubernetes Engine, and use preemptible VMs without
checkpoints.


## Question 12.
You work for a magazine distributor and need to build a model that predicts which customers will renew their
subscriptions for the upcoming year. Using your company’s historical data as your training set, you created a
TensorFlow model and deployed it to AI Platform. You need to determine which customer attribute has the
most predictive power for each prediction served by the model. What should you do?
### Answer:
Use the What-If tool in Google Cloud to determine how your model will perform when individual
features are excluded. Rank the feature importance in order of those that caused the most significant
performance drop when removed from the model

## Question 13.
You work for an advertising company and want to understand the effectiveness of your company's latest
advertising campaign. You have streamed 500 MB of campaign data into BigQuery. You want to query the
table, and then manipulate the results of that query with a pandas dataframe in an Al Platform notebook. What
should you do?
### Answer:
Use Al Platform Notebooks' BigQuery cell magic to query the data, and ingest the results as a pandas
dataframe

Explanation
Refer to this link for details: https://cloud.google.com/bigquery/docs/bigquery-storage-python-pandas
First 2 points talks about querying the data.
Download query results to a pandas DataFrame by using the BigQuery Storage API from the IPython magics
for BigQuery in a Jupyter notebook.
Download query results to a pandas DataFrame by using the BigQuery client library for Python.
Download BigQuery table data to a pandas DataFrame by using the BigQuery client library for Python.
Download BigQuery table data to a pandas DataFrame by using the BigQuery Storage API client library for
Python.


## Question 14.
You work for a gaming company that manages a popular online multiplayer game where teams with 6 players
play against each other in 5-minute battles. There are many new players every day. You need to build a model
that automatically assigns available players to teams in real time. User research indicates that the game is more
enjoyable when battles have players with similar skill levels. Which business metrics should you track to
measure your model’s performance?
### Answer:

User engagement as measured by the number of battles played daily per user

## Question 15.
You developed an ML model with Al Platform, and you want to move it to production. You serve a few
thousand queries per second and are experiencing latency issues. Incoming requests are served by a load
balancer that distributes them across multiple Kubeflow CPU-only pods running on Google Kubernetes
Engine (GKE). Your goal is to improve the serving latency without changing the underlying infrastructure.
What should you do?
### Answer:
Recompile TensorFlow Serving using the source to support CPU-specific optimizations Instruct GKE to
choose an appropriate baseline minimum CPU platform for serving nodes

## Question 16.
Your team needs to build a model that predicts whether images contain a driver's license, passport, or credit
card. The data engineering team already built the pipeline and generated a dataset composed of 10,000 images
with driver's licenses, 1,000 images with passports, and 1,000 images with credit cards. You now have to train
a model with the following label map: ['driversjicense', 'passport', 'credit_card']. Which loss function should
you use?
### Answer:
Categorical cross-entropy
Explanation
- **Categorical entropy** is better to use when you want to **prevent the model from giving more importance
to a certain class**. Or if the **classes are very unbalanced** you will get a better result by using Categorical
entropy.
- But **Sparse Categorical Entropy** is a more optimal coice if you have a huge amount of classes, enough to
make a lot of memory usage, so since sparse categorical entropy uses less columns it **uses less memory**.


## Question 17.
Your organization wants to make its internal shuttle service route more efficient. The shuttles currently stop at
all pick-up points across the city every 30 minutes between 7 am and 10 am. The development team has
already built an application on Google Kubernetes Engine that requires users to confirm their presence and
shuttle station one day in advance. What approach should you take?
### Answer:
Define the optimal route as the shortest route that passes by all shuttle stations with confirmed
attendance at the given time under capacity constraints.
2 Dispatch an appropriately sized shuttle and indicate the required stops on the map
Explanation
This is a case where machine learning would be terrible, as it would not be 100% accurate and some
passengers would not get picked up. A simple algorith works better here, and the question confirms customers
will be indicating when they are at the stop so no ML required.

## Question 18.

You have a demand forecasting pipeline in production that uses Dataflow to preprocess raw data prior to
model training and prediction. During preprocessing, you employ Z-score normalization on data stored in
BigQuery and write it back to BigQuery. New training data is added every week. You want to make the
process more efficient by minimizing computation time and manual intervention. What should you do?

Translate the normalization algorithm into SQL for use with BigQuery

## Question 20.

You work for a public transportation company and need to build a model to estimate delay times for multiple
transportation routes. Predictions are served directly to users in an app in real time. Because different seasons
and population increases impact the data relevance, you will retrain the model every month. You want to
follow Google-recommended best practices. How should you configure the end-to-end architecture of the
predictive model?
### Answer:
Configure Kubeflow Pipelines to schedule your multi-step workflow from training to deploying your
model.

## Question 21.

You deployed an ML model into production a year ago. Every month, you collect all raw requests that were
sent to your model prediction service during the previous month. You send a subset of these requests to a
human labeling service to evaluate your model’s performance. After a year, you notice that your model's
performance sometimes degrades significantly after a month, while other times it takes several months to
notice any decrease in performance. The labeling service is costly, but you also need to avoid large
performance degradations. You want to determine how often you should retrain your model to maintain a high
level of performance while minimizing cost. What should you do?
### Answer:
Train an anomaly detection model on the training dataset, and run all incoming requests through this
model. If an anomaly is detected, send the most recent serving data to the labeling service.

## Question 22.

You have deployed multiple versions of an image classification model on Al Platform. You want to monitor
the performance of the model versions overtime. How should you perform this comparison?
### Answer:
Compare the mean average precision across the models using the Continuous Evaluation feature

## Question 23.

You work on a data science team at a bank and are creating an ML model to predict loan default risk. You
have collected and cleaned hundreds of millions of records worth of training data in a BigQuery table, and you
now want to develop and compare multiple models on this data using TensorFlow and Vertex AI. You want to
minimize any bottlenecks during the data ingestion state while considering scalability. What should you do?
### Answer
Export data to CSV files in Cloud Storage, and use tf.data.TextLineDataset() to read them.


## Question 24.

You are an ML engineer responsible for designing and implementing training pipelines for ML models. You
need to create an end-to-end training pipeline for a TensorFlow model. The TensorFlow model will be trained
on several terabytes of structured data. You need the pipeline to include data quality checks before training
and model quality checks after training but prior to deployment. You want to minimize development time and
the need for infrastructure maintenance. How should you build and orchestrate your training pipeline?
### Answer:
Create the pipeline using TensorFlow Extended (TFX) and standard TFX components. Orchestrate the
pipeline using Vertex AI Pipelines.

## Question 25.

You work for a toy manufacturer that has been experiencing a large increase in demand. You need to build an
ML model to reduce the amount of time spent by quality control inspectors checking for product defects.
Faster defect detection is a priority. The factory does not have reliable Wi-Fi. Your company wants to
implement the new ML model as soon as possible. Which model should you use?
### Answer:
AutoML Vision model

## Question 26.
You are creating a deep neural network classification model using a dataset with categorical input values.
Certain columns have a cardinality greater than 10,000 unique values. How should you encode these
categorical values as input into the model?
### Answer:
Map the categorical variables into a vector of boolean values.

## Question 27.

You have successfully deployed to production a large and complex TensorFlow model trained on tabular data.
You want to predict the lifetime value (LTV) field for each subscription stored in the BigQuery table named
subscription. subscriptionPurchase in the project named my-fortune500-company-project.
You have organized all your training code, from preprocessing data from the BigQuery table up to deploying
the validated model to the Vertex AI endpoint, into a TensorFlow Extended (TFX) pipeline. You want to
prevent prediction drift, i.e., a situation when a feature data distribution in production changes significantly
over time. What should you do?
### Answer
Add a model monitoring job where 90% of incoming predictions are sampled 24 hours.

## Question 28.

Your team is building a convolutional neural network (CNN)-based architecture from scratch. The preliminary
experiments running on your on-premises CPU-only infrastructure were encouraging, but have slow
convergence. You have been asked to speed up model training to reduce time-to-market. You want to
experiment with virtual machines (VMs) on Google Cloud to leverage more powerful hardware. Your code
does not include any manual device placement and has not been wrapped in Estimator model-level abstraction.
Which environment should you train your model on?
### Answer:
A Deep Learning VM with an n1-standard-2 machine and 1 GPU with all libraries pre-installed.


Explanation
"speed up model training" will make us biased towards GPU,TPU options by options eliminations we may
need to stay away of any manual installations , so using preconfigered deep learning will speed up time to
market


## Question 29.

You are an ML engineer at a travel company. You have been researching customers’ travel behavior for many
years, and you have deployed models that predict customers’ vacation patterns. You have observed that
customers’ vacation destinations vary based on seasonality and holidays; however, these seasonal variations
are similar across years. You want to quickly and easily store and compare the model versions and
performance statistics across years. What should you do?
### Answer:
Create versions of your models for each season per year in Vertex AI. Compare the performance
statistics across the models in the Evaluate tab of the Vertex AI UI.

## Question 30.

You are an ML engineer on an agricultural research team working on a crop disease detection tool to detect
leaf rust spots in images of crops to determine the presence of a disease. These spots, which can vary in shape
and size, are correlated to the severity of the disease. You want to develop a solution that predicts the presence
and severity of the disease with high accuracy. What should you do?

Develop an image segmentation ML model to locate the boundaries of the rust spots.

# Part 2

GCP Machine Learning Practice Questions Part 2.

[back](#part-1)

<iframe src="https://player.rss.com/ruslanmv/833525" style="width: 100%" title="Machine Learning and Data Science - Podcast" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen><a href="https://rss.com/podcasts/ruslanmv/833525/">Machine Learning in GCP - Part 2 | RSS.com</a></iframe>



## Question 31.

You have recently created a proof-of-concept (POC) deep learning model. You are satisfied with the overall
architecture, but you need to determine the value for a couple of hyperparameters. You want to perform
hyperparameter tuning on Vertex AI to determine both the appropriate embedding dimension for a categorical
feature used by your model and the optimal learning rate. You configure the following settings:
For the embedding dimension, you set the type to INTEGER with a minValue of 16 and maxValue of 64.
For the learning rate, you set the type to DOUBLE with a minValue of 10e-05 and maxValue of 10e-02.
You are using the default Bayesian optimization tuning algorithm, and you want to maximize model accuracy.
Training time is not a concern. How should you set the hyperparameter scaling for each hyperparameter and
the maxParallelTrials?

Use UNIT_LINEAR_SCALE for the embedding dimension, UNIT_LOG_SCALE for the learning rate,
and a small number of parallel trials.

## Question 32.

You have trained a model on a dataset that required computationally expensive preprocessing operations. You
need to execute the same preprocessing at prediction time. You deployed the model on Al Platform for
high-throughput online prediction. Which architecture should you use?
### Answer:
• Send incoming prediction requests to a Pub/Sub topic
• Transform the incoming data using a Dataflow job
• Submit a prediction request to Al Platform using the transformed data
• Write the predictions to an outbound Pub/Sub queue

## Question 33.

You work for a large hotel chain and have been asked to assist the marketing team in gathering predictions for
a targeted marketing strategy. You need to make predictions about user lifetime value (LTV) over the next 30
days so that marketing can be adjusted accordingly. The customer dataset is in BigQuery, and you are
preparing the tabular data for training with AutoML Tables. This data has a time signal that is spread across
multiple columns. How should you ensure that AutoML fits the best model to your data?
### Answer:

Submit the data for training without performing any manual transformations Use the columns that have
a time signal to manually split your data Ensure that the data in your validation set is from 30 days after
the data in your training set and that the data in your testing set is from 30 days after your validation set

## Question 34.

Your organization's call center has asked you to develop a model that analyzes customer sentiments in each
call. The call center receives over one million calls daily, and data is stored in Cloud Storage. The data
collected must not leave the region in which the call originated, and no Personally Identifiable Information
(Pll) can be stored or analyzed. The data science team has a third-party tool for visualization and access which
requires a SQL ANSI-2011 compliant interface. You need to select components for data processing and for
analytics. How should the data pipeline be designed?
### Answer:

1 = Dataflow, 2 = BigQuery

## Question 35.

You have been given a dataset with sales predictions based on your company’s marketing activities. The data
is structured and stored in BigQuery, and has been carefully managed by a team of data analysts. You need to
prepare a report providing insights into the predictive capabilities of the data. You were asked to run several
ML models with different levels of sophistication, including simple models and multilayered neural networks.
You only have a few hours to gather the results of your experiments. Which Google Cloud tools should you
use to complete this task in the most efficient and self-serviced way?
### Answer:

Use BigQuery ML to run several regression models, and analyze their performance.

## Question 36.

You are building a TensorFlow model for a financial institution that predicts the impact of consumer spending
on inflation globally. Due to the size and nature of the data, your model is long-running across all types of
hardware, and you have built frequent checkpointing into the training process. Your organization has asked
you to minimize cost. What hardware should you choose?
### Answer:

A Vertex AI Workbench user-managed notebooks instance running on an n1-standard-16 with an
NVIDIA P100 GPU

## Question 37.

You recently developed a deep learning model using Keras, and now you are experimenting with different
training strategies. First, you trained the model using a single GPU, but the training process was too slow.
Next, you distributed the training across 4 GPUs using tf.distribute.MirroredStrategy (with no other changes),
but you did not observe a decrease in training time. What should you do?
### Answer:
Use a TPU with tf.distribute.TPUStrategy


## Question 38.

You are an ML engineer at a large grocery retailer with stores in multiple regions. You have been asked to
create an inventory prediction model. Your models features include region, location, historical demand, and
seasonal popularity. You want the algorithm to learn from new inventory data on a daily basis. Which
algorithms should you use to build the model?
### Answer:
Recurrent Neural Networks (RNN)
Explanation
"algorithm to learn from new inventory data on a daily basis" = time series model , best option to deal with
time series is forsure RNN

## Question 39.

You work for an online travel agency that also sells advertising placements on its website to other companies.
You have been asked to predict the most relevant web banner that a user should see next. Security is
important to your company. The model latency requirements are 300ms@p99, the inventory is thousands of
web banners, and your exploratory analysis has shown that navigation context is a good predictor. You want to
Implement the simplest solution. How should you configure the prediction pipeline?
### Answer:
Embed the client on the website, deploy the gateway on App Engine, deploy the database on Cloud
Bigtable for writing and for reading the user’s navigation context, and then deploy the model on AI
Platform Prediction.

## Question 40.

You need to analyze user activity data from your company’s mobile applications. Your team will use
BigQuery for data analysis, transformation, and experimentation with ML algorithms. You need to ensure
real-time ingestion of the user activity data into BigQuery. What should you do?
### Answer:
Configure Pub/Sub to stream the data into BigQuery

## Question 41.

You recently designed and built a custom neural network that uses critical dependencies specific to your
organization's framework. You need to train the model using a managed training service on Google Cloud.
However, the ML framework and related dependencies are not supported by Al Platform Training. Also, both
your model and your data are too large to fit in memory on a single machine. Your ML framework of choice
uses the scheduler, workers, and servers distribution structure. What should you do?
### Answer:
Build your custom containers to run distributed training jobs on Al Platform Training

Explanation
"ML framework and related dependencies are not supported by Al Platform Training" use custom containers
"your model and your data are too large to fit in memory on a single machine " use distributed learning
techniques
## Question 42.

You have trained a deep neural network model on Google Cloud. The model has low loss on the training data,
but is performing worse on the validation data. You want the model to be resilient to overfitting. Which
strategy should you use when retraining the model?
### Answer:

Run a hyperparameter tuning job on Al Platform to optimize for the L2 regularization and dropout
parameters

## Question 43.

As the lead ML Engineer for your company, you are responsible for building ML models to digitize scanned
customer forms. You have developed a TensorFlow model that converts the scanned images into text and
stores them in Cloud Storage. You need to use your ML model on the aggregated data collected at the end of
each day with minimal manual intervention. What should you do?

### Answer:
Use the batch prediction functionality of Al Platform


## Question 44.
### Answer:
You have a functioning end-to-end ML pipeline that involves tuning the hyperparameters of your ML model
using Al Platform, and then using the best-tuned parameters for training. Hypertuning is taking longer than
expected and is delaying the downstream processes. You want to speed up the tuning job without significantly
compromising its effectiveness. Which actions should you take?
Set the early stopping parameter to TRUE
Decrease the maximum number of trials during subsequent training phases


## Question 45.

You manage a team of data scientists who use a cloud-based backend system to submit training jobs. This
system has become very difficult to administer, and you want to use a managed service instead. The data
scientists you work with use many different frameworks, including Keras, PyTorch, theano. Scikit-team, and
custom libraries. What should you do?
### Answer:
Use the Al Platform custom containers feature to receive training jobs using any framework

Explanation
because AI platform supported all the frameworks mentioned. And Kubeflow is not managed service in GCP.
Use the ML framework of your choice. If you can't find an AI Platform Training runtime version that supports
the ML framework you want to use, then you can build a custom container that installs your chosen framework
and use it to run jobs on AI Platform Training.

## Question 46.

You have built a model that is trained on data stored in Parquet files. You access the data through a Hive table
hosted on Google Cloud. You preprocessed these data with PySpark and exported it as a CSV file into Cloud
Storage. After preprocessing, you execute additional steps to train and evaluate your model. You want to
parametrize this model training in Kubeflow Pipelines. What should you do?
### Answer:
Deploy Apache Spark at a separate node pool in a Google Kubernetes Engine cluster. Add a
ContainerOp to your pipeline that invokes a corresponding transformation job for this Spark instance.

## Question 47.

You built and manage a production system that is responsible for predicting sales numbers. Model accuracy is
crucial, because the production model is required to keep up with market changes. Since being deployed to
production, the model hasn't changed; however the accuracy of the model has steadily deteriorated. What issue
is most likely causing the steady decline in model accuracy?
### Answer:
Lack of model retraining

Explanation
Retraining is needed as the market is changing. its how the Model keep updated and predictions accuracy.


## Question 48.

You recently joined a machine learning team that will soon release a new project. As a lead on the project, you
are asked to determine the production readiness of the ML components. The team has already tested features
and data, model development, and infrastructure. Which additional readiness check should you recommend to
the team?
### Answer:
Ensure that training is reproducible

## Question 49.

Your data science team needs to rapidly experiment with various features, model architectures, and
hyperparameters. They need to track the accuracy metrics for various experiments and use an API to query the
metrics over time. What should they use to track and report their experiments while minimizing manual effort?
### Answer:
Use Kubeflow Pipelines to execute the experiments Export the metrics file, and query the results using
the Kubeflow Pipelines API.


Explanation
https:Kubeflow Pipelines (KFP)
helps solve these issues by providing a way to deploy robust, repeatable machine learning pipelines along with
monitoring, auditing, version tracking, and reproducibility. Cloud AI Pipelines makes it easy to set up a KFP
installation.

"Kubeflow Pipelines supports the export of scalar metrics. You can write a list of metrics to a local file to
describe the performance of the model. The pipeline agent uploads the local file as your run-time metrics. You
can view the uploaded metrics as a visualization in the Runs page for a particular experiment in the Kubeflow
Pipelines UI." https://www.kubeflow.org/docs/components/pipelines/sdk/pipelines-metrics/

## Question 50.

You work for a large technology company that wants to modernize their contact center. You have been asked
to develop a solution to classify incoming calls by product so that requests can be more quickly routed to the
correct support team. You have already transcribed the calls using the Speech-to-Text API. You want to
minimize data preprocessing and development time. How should you build the model?


Use AutoML Natural Language to extract custom entities for classification

## Question 51.

You are working on a system log anomaly detection model for a cybersecurity organization. You have
developed the model using TensorFlow, and you plan to use it for real-time prediction. You need to create a
Dataflow pipeline to ingest data via Pub/Sub and write the results to BigQuery. You want to minimize the
serving latency as much as possible. What should you do?

Deploy the model to a Vertex AI endpoint, and invoke this endpoint in the Dataflow job.

You have been asked to build a model using a dataset that is stored in a medium-sized (~10 GB) BigQuery
table. You need to quickly determine whether this data is suitable for model development. You want to create
a one-time report that includes both informative visualizations of data distributions and more sophisticated
statistical analyses to share with other ML engineers on your team. You require maximum flexibility to create
your report. What should you do?
### Answer:
Use the output from TensorFlow Data Validation on Dataflow to generate the report.


## Question 53.

You need to execute a batch prediction on 100 million records in a BigQuery table with a custom TensorFlow
DNN regressor model, and then store the predicted results in a BigQuery table. You want to minimize the
effort required to build this inference pipeline. What should you do?

Import the TensorFlow model with BigQuery ML, and run the ml.predict function.


## Question 54.

You are developing models to classify customer support emails. You created models with TensorFlow
Estimators using small datasets on your on-premises system, but you now need to train the models using large
datasets to ensure high performance. You will port your models to Google Cloud and want to minimize code
refactoring and infrastructure overhead for easier migration from on-prem to cloud. What should you do?

### Answer:
Use Al Platform for distributed training

Explanation
AI platform also contains kubeflow pipelines. you don't need to set up infrastructure to use it. For D you need
to set up a kubernetes cluster engine. The question asks us to minimize infrastructure overheard.





## Question 55.

You are building a linear regression model on BigQuery ML to predict a customer's likelihood of purchasing
your company's products. Your model uses a city name variable as a key predictive component. In order to
train and serve the model, your data must be organized in columns. You want to prepare your data using the
least amount of coding while maintaining the predictable variables. What should you do?
### Answer:
Use Cloud Data Fusion to assign each city to a region labeled as 1, 2, 3, 4, or 5r and then use that
number to represent the city in the model.


## Question 57
You are building an ML model to predict trends in the stock market based on a wide range of factors. While
exploring the data, you notice that some features have a large range. You want to ensure that the features with
the largest magnitude don’t overfit the model. What should you do?

Normalize the data by scaling it to have values between 0 and 1.



## Question 58.

You work for an online publisher that delivers news articles to over 50 million readers. You have built an AI
model that recommends content for the company’s weekly newsletter. A recommendation is considered
successful if the article is opened within two days of the newsletter’s published date and the user remains on
the page for at least one minute.
All the information needed to compute the success metric is available in BigQuery and is updated hourly. The
model is trained on eight weeks of data, on average its performance degrades below the acceptable baseline
after five weeks, and training time is 12 hours. You want to ensure that the model’s performance is above the
acceptable baseline while minimizing cost. How should you monitor the model to determine when retraining is
necessary?
### Answer:

Schedule a daily Dataflow job in Cloud Composer to compute the success metric.


## Question 59.


You are an ML engineer at a manufacturing company. You need to build a model that identifies defects in
products based on images of the product taken at the end of the assembly line. You want your model to
preprocess the images with lower computation to quickly extract features of defects in products. Which
approach should you use to build the model?
### Answer:
Convolutional Neural Networks (CNN)

## Question 60.

You need to train a computer vision model that predicts the type of government ID present in a given image
using a GPU-powered virtual machine on Compute Engine. You use the following parameters:
• Optimizer: SGD
• Image shape = 224x224
• Batch size = 64
• Epochs = 10
• Verbose = 2
During training you encounter the following error: ResourceExhaustedError: out of Memory (oom) when
allocating tensor. What should you do?
### Answer:
Reduce the batch size

You need to train a regression model based on a dataset containing 50,000 records that is stored in BigQuery.
The data includes a total of 20 categorical and numerical features with a target variable that can include
negative values. You need to minimize effort and training time while maximizing model performance. What
approach should you take to train this regression model?


Use BQML XGBoost regression to train the model

# Part 3

GCP Machine Learning Practice Questions Part 3.

[back](#part-1)

<iframe src="https://player.rss.com/ruslanmv/833529" style="width: 100%" title="Machine Learning and Data Science - Podcast" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen><a href="https://rss.com/podcasts/ruslanmv/833529/">Machine Learning in GCP - Part 3 | RSS.com</a></iframe>

## Question 61.

You need to train a regression model based on a dataset containing 50,000 records that is stored in BigQuery.
The data includes a total of 20 categorical and numerical features with a target variable that can include
negative values. You need to minimize effort and training time while maximizing model performance. What
approach should you take to train this regression model?
### Answer:
Use BQML XGBoost regression to train the model
## Question 62.

You were asked to investigate failures of a production line component based on sensor readings. After
receiving the dataset, you discover that less than 1% of the readings are positive examples representing failure
incidents. You have tried to train several classification models, but none of them converge. How should you
resolve the class imbalance problem?
### Answer:
Downsample the data with upweighting to create a sample with 10% positive examples
## Question 63.

You recently joined an enterprise-scale company that has thousands of datasets. You know that there are
accurate descriptions for each table in BigQuery, and you are searching for the proper BigQuery table to use
for a model you are building on AI Platform. How should you find the data that you need?
Use Data Catalog to search the BigQuery datasets by using keywords in the table description.
## Question 64.
### Answer:
Your task is classify if a company logo is present on an image. You found out that 96% of a data does not
include a logo. You are dealing with data imbalance problem. Which metric do you use to evaluate to model?
F Score with higher recall weighted than precision
## Question 65.

You want to rebuild your ML pipeline for structured data on Google Cloud. You are using PySpark to conduct
data transformations at scale, but your pipelines are taking over 12 hours to run. To speed up development and
pipeline run time, you want to use a serverless tool and SQL syntax. You have already moved your raw data
into Cloud Storage. How should you build the pipeline on Google Cloud while meeting the speed and
processing requirements?
### Answer:
Ingest your data into BigQuery using BigQuery Load, convert your PySpark commands into BigQuery
SQL queries to transform the data, and then write the transformations to a new table

## Question 67.

While performing exploratory data analysis on a dataset, you find that an important categorical feature has 5%
null values. You want to minimize the bias that could result from the missing values. How should you handle
the missing values?
### Answer:
Replace the missing values with the feature’s mean
## Question 68.

You are designing an architecture with a serverless ML system to enrich customer support tickets with
informative metadata before they are routed to a support agent. You need a set of models to predict ticket
priority, predict ticket resolution time, and perform sentiment analysis to help agents make strategic decisions
when they process support requests.
### Answer:
1 = Al Platform, 2 = Al Platform, 3 = Cloud Natural Language API
## Question 69.

-The Cloud Function calls 3 different endpoints to enrich the ticket:
-An AI Platform endpoint, where the function can predict the priority.
-An AI Platform endpoint, where the function can predict the resolution time.
-The Natural Language API to do sentiment analysis and word salience.
-For each reply, the Cloud Function updates the Firebase real-time database.
-The Cloud Function then creates a ticket into the helpdesk platform using the RESTful API
### Answer:
Use Vertex AI Workbench user-managed notebooks to build a custom model that has three times as
many examples of pictures that meet the profile photo requirements

## Question 70.
Your data science team has requested a system that supports scheduled model retraining, Docker containers,
and a service that supports autoscaling and monitoring for online prediction requests. Which platform
components should you choose for this system?
### Answer:
Kubeflow Pipelines and Al Platform Prediction

## Question 71.

You work on the data science team for a multinational beverage company. You need to develop an ML model
to predict the company’s profitability for a new line of naturally flavored bottled waters in different locations.
You are provided with historical data that includes product types, product sales volumes, expenses, and profits
for all regions. What should you use as the input and output for your model?
### Answer:
Use product type and the feature cross of latitude with longitude, followed by binning, as features. Use
profit as model output
## Question 72.

You are developing an ML model intended to classify whether X-Ray images indicate bone fracture risk. You
have trained on Api Resnet architecture on Vertex AI using a TPU as an accelerator, however you are
unsatisfied with the trainning time and use memory usage. You want to quickly iterate your training code but
make minimal changes to the code. You also want to minimize impact on the models accuracy. What should
you do?
### Answer:
Reduce the global batch size from 1024 to 256
## Question 73.

One of your models is trained using data provided by a third-party data broker. The data broker does not
reliably notify you of formatting changes in the data. You want to make your model training pipeline more
robust to issues like this. What should you do?
### Answer:
Use TensorFlow Transform to create a preprocessing component that will normalize data to the
expected distribution, and replace values that don’t match the schema with 0.

## Question 74.

You are training an object detection machine learning model on a dataset that consists of three million X-ray
images, each roughly 2 GB in size. You are using Vertex AI Training to run a custom training application on a
Compute Engine instance with 32-cores, 128 GB of RAM, and 1 NVIDIA P100 GPU. You notice that model
training is taking a very long time. You want to decrease training time without sacrificing model performance.
What should you do?
### Answer:
Enable early stopping in your Vertex AI Training job.
## Question 75.

You work at a subscription-based company. You have trained an ensemble of trees and neural networks to
predict customer churn, which is the likelihood that customers will not renew their yearly subscription. The
average prediction is a 15% churn rate, but for a particular customer the model predicts that they are 70%
likely to churn. The customer has a product usage history of 30%, is located in New York City, and became a
customer in 1997. You need to explain the difference between the actual prediction, a 70% churn rate, and the
average prediction. You want to use Vertex Explainable AI. What should you do?
### Answer:
Train local surrogate models to explain individual predictions.


Note  :76.

Google may optimize the configuration of the scale tiers for different jobs over time, based on customer
feedback and the availability of cloud resources. Each scale tier is defined in terms of its suitability for certain
types of jobs. Generally, the more advanced the tier, the more machines are allocated to the cluster, and the
more powerful the specifications of each virtual machine. As you increase the complexity of the scale tier, the
hourly cost of training jobs, measured in training units, also increases. See the pricing page to calculate the
cost of your job.

## Question 77.

You have trained a text classification model in TensorFlow using Al Platform. You want to use the trained
model for batch predictions on text data stored in BigQuery while minimizing computational overhead. What
should you do?
Export the model to BigQuery ML.

## Question 78.


You work for a global footwear retailer and need to predict when an item will be out of stock based on
historical inventory data. Customer behavior is highly dynamic since footwear demand is influenced by many
different factors. You want to serve models that are trained on all available data, but track your performance
on specific subsets of data before pushing to production. What is the most streamlined and reliable way to
perform this validation?
### Answer:
Use the TFX ModelValidator tools to specify performance metrics for production readiness



## Question 79.

You work for a credit card company and have been asked to create a custom fraud detection model based on
historical data using AutoML Tables. You need to prioritize detection of fraudulent transactions while
minimizing false positives. Which optimization objective should you use when training the model?
### Answer:
An optimization objective that maximizes the area under the precision-recall curve (AUC PR) value

## Question 80.
You are building a real-time prediction engine that streams files which may contain Personally Identifiable
Information (Pll) to Google Cloud. You want to use the Cloud Data Loss Prevention (DLP) API to scan the
files. How should you ensure that the Pll is not accessible by unauthorized individuals?
### Answer:
Stream all files to Google CloudT and then write the data to BigQuery Periodically conduct a bulk scan
of the table using the DLP API.


## Question 81.
Periodically conduct a bulk scan of that bucket using the DLP API, and move the data to either the
Sensitive or Non-Sensitive bucket
### Answer:
1. Export your data to Cloud Storage using Dataflow.
2. Submit a Vertex AI batch prediction job that uses your trained model in Cloud Storage to perform
scoring on the preprocessed data.
3. Export the batch prediction job outputs from Cloud Storage and import them into Cloud SQL.


## Question 82.

You are profiling the performance of your TensorFlow model training time and notice a performance issue
caused by inefficiencies in the input data pipeline for a single 5 terabyte CSV file dataset on Cloud Storage.
You need to optimize the input pipeline performance. Which action should you try first to increase the
efficiency of your pipeline?
### Answer:
Set the reshuffle_each_iteration parameter to true in the tf.data.Dataset.shuffle method.

## Question 83.


You work for a social media company. You need to detect whether posted images contain cars. Each training
example is a member of exactly one class. You have trained an object detection neural network and deployed
the model version to Al Platform Prediction for evaluation. Before deployment, you created an evaluation job
and attached it to the Al Platform Prediction model version. You notice that the precision is lower than your
business requirements allow. How should you adjust the model's final layer softmax threshold to increase
precision?
### Answer:
Decrease the number of false negatives


## Question 84.

You are training a TensorFlow model on a structured data set with 100 billion records stored in several CSV
files. You need to improve the input/output execution performance. What should you do?
### Answer:
Convert the CSV files into shards of TFRecords, and store the data in Cloud Storage

## Question 85.

You are building a linear model with over 100 input features, all with values between -1 and 1. You suspect
that many features are non-informative. You want to remove the non-informative features from your model
while keeping the informative ones in their original form. Which technique should you use?
### Answer:
Use L1 regularization to reduce the coefficients of uninformative features to 0.

## Question 86.

You are an ML engineer in the contact center of a large enterprise. You need to build a sentiment analysis tool
that predicts customer sentiment from recorded phone conversations. You need to identify the best approach to
building a model while ensuring that the gender, age, and cultural differences of the customers who called the
contact center do not impact any stage of the model development pipeline and results. What should you do?
### Answer:
Convert the speech to text and extract sentiments based on the sentences
## Question 87.
You have deployed a model on Vertex AI for real-time inference. During an online prediction request, you get
an “Out of Memory” error. What should you do?
### Answer:
Use base64 to encode your data before using it for prediction.

## Question 88.

You are building an ML model to detect anomalies in real-time sensor data. You will use Pub/Sub to handle
incoming requests. You want to store the results for analytics and visualization. How should you configure the
pipeline?
### Answer:
1 = Dataflow, 2 - Al Platform, 3 = BigQuery



## Question 89.

You work as an ML engineer at a social media company, and you are developing a visual filter for users’
profile photos. This requires you to train an ML model to detect bounding boxes around human faces. You
want to use this filter in your company’s iOS-based mobile phone application. You want to minimize code
development and want the model to be optimized for inference on mobile phones. What should you do?
### Answer:
Train a model using AutoML Vision and use the “export for Core ML” option.

## Question 90.

You work on an operations team at an international company that manages a large fleet of on-premises servers
located in few data centers around the world. Your team collects monitoring data from the servers, including
CPU/memory consumption. When an incident occurs on a server, your team is responsible for fixing it.
Incident data has not been properly labeled yet. Your management team wants you to build a predictive
maintenance solution that uses monitoring data from the VMs to detect potential failures and then alerts the
service desk team. What should you do first?
### Answer:
Hire a team of qualified analysts to review and label the machines’ historical performance data. Train a
model based on this manually labeled dataset.

# Part 4

GCP Machine Learning Practice Questions Part 4.

[back](#part-1)

<iframe src="https://player.rss.com/ruslanmv/833531" style="width: 100%" title="Machine Learning and Data Science - Podcast" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen><a href="https://rss.com/podcasts/ruslanmv/833531/">Machine Learning in GCP - Part 4 | RSS.com</a></iframe>


## Question 91.


You need to quickly build and train a model to predict the sentiment of customer reviews with custom
categories without writing code. You do not have enough data to train a model from scratch. The resulting
model should have high predictive performance. Which service should you use?
### Answer:
AutoML Natural Language

Note :92.

Your team is working on an NLP research project to predict political affiliation of authors based on articles
they have written.
If we just put inside the Training set , Validation set and Test set , randomly Text, Paragraph or sentences the
model will have the ability to learn specific qualities about The Author's use of language beyond just his own
articles. Therefore the model will mixed up different opinions. Rather if we divided things up a the author
level, so that given authors were only on the training data, or only in the test data or only in the validation data.
The model will find more difficult to get a high accuracy on the test validation (What is correct and have more
sense!). Because it will need to really focus in author by author articles rather than get a single political
affiliation based on a bunch of mixed articles from different authors.

## Question 93.

You have developed a
logistic regression model with BigQuery ML that predicts whether offering a promo code for free popcorn
increases the chance of a ticket purchase, and this prediction should be added to the ticket purchase process.
You want to identify the simplest way to deploy this model to production while adding minimal latency. What
should you do?
### Answer:
Run batch inference with BigQuery ML every five minutes on each new set of tickets issued
## Question 94.

You work for the AI team of an automobile company, and you are developing a visual defect detection model
using TensorFlow and Keras. To improve your model performance, you want to incorporate some image
augmentation functions such as translation, cropping, and contrast tweaking. You randomly apply these
functions to each training batch. You want to optimize your data processing pipeline for run time and compute
resources utilization. What should you do?

### Answer:
Use Dataflow to create all possible augmentations, and store them as TFRecords.

## Question 95.


You work for a magazine publisher and have been tasked with predicting whether customers will cancel their
annual subscription. In your exploratory data analysis, you find that 90% of individuals renew their
subscription every year, and only 10% of individuals cancel their subscription. After training a NN Classifier,
your model predicts those who cancel their subscription with 99% accuracy and predicts those who renew
their subscription with 82% accuracy. How should you interpret these results?


This is a good result because predicting those who cancel their subscription is more difficult, since there
is less data for this group.

## Question 96.


You need to build classification workflows over several structured datasets currently stored in BigQuery.
Because you will be performing the classification several times, you want to complete the following steps
without writing code: exploratory data analysis, feature selection, model building, training, and
hyperparameter tuning and serving. What should you do?
### Answer:
Configure AutoML Tables to perform the classification task

## Question 97.
You work for a retailer that sells clothes to customers around the world. You have been tasked with ensuring
that ML models are built in a secure manner. Specifically, you need to protect sensitive customer data that
might be used in the models. You have identified four fields containing sensitive data that are being used by
your data science team: AGE, IS_EXISTING_CUSTOMER, LATITUDE_LONGITUDE, and SHIRT_SIZE.
What should you do with the data before it is made available to the data science team for training purposes?
### Answer:
Tokenize all of the fields using hashed dummy values to replace the real values.


## Question 98.
You work for a bank and are building a random forest model for fraud detection. You have a dataset that
includes transactions, of which 1% are identified as fraudulent. Which data transformation strategy would
likely improve the performance of your classifier?
### Answer:
Oversample the fraudulent transaction 10 times.
## Question 99.

You work for a company that provides an anti-spam service that flags and hides spam posts on social media
platforms. Your company currently uses a list of 200,000 keywords to identify suspected spam posts. If a post
contains more than a few of these keywords, the post is identified as spam. You want to start using machine
learning to flag spam posts for human review. What is the main advantage of implementing machine learning
for this business case?
### Answer:

Posts can be compared to the keyword list much more quickly.
## Question 100.
You are a data scientist at an industrial equipment manufacturing company. You are developing a regression
model to estimate the power consumption in the company’s manufacturing plants based on sensor data
collected from all of the plants. The sensors collect tens of millions of records every day. You need to schedule
daily training runs for your model that use all the data collected up to the current date. You want your model to
scale smoothly and require minimal development work. What should you do?
### Answer:
Train a regression model using AutoML Tables.

## Question 101.
You work on a growing team of more than 50 data scientists who all use Al Platform. You are designing a
strategy to organize your jobs, models, and versions in a clean and scalable way. Which strategy should you
choose?
### Answer:
Use labels to organize resources into descriptive categories. Apply a label to each created resource so
that users can filter the results by label when viewing or monitoring the resources


## Question 102.

You need to train a natural language model to perform text classification on product descriptions that contain
millions of examples and 100,000 unique words. You want to preprocess the words individually so that they
can be fed into a recurrent neural network. What should you do?
### Answer:
Identify word embeddings from a pre-trained model, and use the embeddings in your model.


## Question 103.
You have a large corpus of written support cases that can be classified into 3 separate categories: Technical
Support, Billing Support, or Other Issues. You need to quickly build, test, and deploy a service that will
automatically classify future written requests into one of the categories. How should you configure the
pipeline?
Use AutoML Natural Language to build and test a classifier. Deploy the model as a REST API.
### Answer:

## Question 104.

You are training a Resnet model on Al Platform using TPUs to visually categorize types of defects in
automobile engines. You capture the training profile using the Cloud TPU profiler plugin and observe that it is
highly input-bound. You want to reduce the bottleneck and speed up your model training process. Which modifications should you make to the tf .data dataset?
Choose 2 ### Answers
Set the prefetch option equal to the training batch size
Decrease the batch size argument in your transformation
### Answer:
## Question 105.
Your team is building an application for a global bank that will be used by millions of customers. You built a
forecasting model that predicts customers1 account balances 3 days in the future. Your team will use the
results in a new feature that will notify users when their account balance is likely to drop below $25. How
should you serve your predictions?
1 Build a notification system on Firebase
2. Register each user with a user ID on the Firebase Cloud Messaging server, which sends a notification when your model predicts that a user's account balance will drop below the $25 threshold
### Answer:
## Question 106.

You work for an online retail company that is creating a visual search engine. You have set up an end-to-end
ML pipeline on Google Cloud to classify whether an image contains your company's product. Expecting the
release of new products in the near future, you configured a retraining functionality in the pipeline so that new
data can be fed into your ML models. You also want to use Al Platform's continuous evaluation service to
ensure that the models have high accuracy on your test data set. What should you do?
### Answer:
Extend your test dataset with images of the newer products when they are introduced to retraining



## Question 107.
You are the Director of Data Science at a large company, and your Data Science team has recently begun
using the Kubeflow Pipelines SDK to orchestrate their training pipelines. Your team is struggling to integrate
their custom Python code into the Kubeflow Pipelines SDK. How should you instruct them to proceed in order
to quickly integrate their code with the Kubeflow Pipelines SDK?
### Answer:
Deploy the custom Python code to Cloud Functions, and use Kubeflow Pipelines to trigger the Cloud
Function.
## Question 108.

You are working on a Neural Network-based project. The dataset provided to you has columns with different
ranges. While preparing the data for model training, you discover that gradient optimization is having
difficulty moving weights to a good solution. What should you do?
### Answer:
Use the representation transformation (normalization) technique.
## Question 109.

Your data science team has requested a system that supports scheduled model retraining, Docker containers,
and a service that supports autoscaling and monitoring for online prediction requests. Which platform
components should you choose for this system?
### Answer:
Vertex AI Pipelines and App Engine
## Question 110.
You are working on a classification problem with time series data and achieved an area under the receiver operating characteristic curve (AUC ROC) value of 99% for training data after just a few experiments. You
haven’t explored using any sophisticated algorithms or spent any time on hyperparameter tuning. What should
your next step be to identify and fix the problem?
### Answer:
Address data leakage by applying nested cross-validation during model training.

## Question 111.
You have written unit tests for a Kubeflow Pipeline that require custom libraries. You want to automate the
execution of unit tests with each new push to your development branch in Cloud Source Repositories. What
should you do?
### Answer:
Using Cloud Build, set an automated trigger to execute the unit tests when changes are pushed to your
development branch.

## Question 112.

You are an ML engineer at an ecommerce company and have been tasked with building a model that predicts
how much inventory the logistics team should order each month. Which approach should you take?
### Answer:
Use a regression model to predict how much additional inventory should be purchased each month. Give
the results to the logistics team at the beginning of the month so they can increase inventory by the
amount predicted by the model

## Question 113.

You are an ML engineer at a mobile gaming company. A data scientist on your team recently trained a
TensorFlow model, and you are responsible for deploying this model into a mobile application. You discover
that the inference latency of the current model doesn’t meet production requirements. You need to reduce the
inference time by 50%, and you are willing to accept a small decrease in model accuracy in order to reach the
latency requirement. Without training a new model, which model optimization technique for reducing latency
should you try first?
### Answer:
Model distillation
## Question 114.
You have been asked to productionize a proof-of-concept ML model built using Keras. The model was trained
in a Jupyter notebook on a data scientist’s local machine. The notebook contains a cell that performs data
validation and a cell that performs model analysis. You need to orchestrate the steps contained in the notebook
and automate the execution of these steps for weekly retraining. You expect much more training data in the
future. You want your solution to take advantage of managed services while minimizing cost. What should
you do?
### Answer:
Write the code as a TensorFlow Extended (TFX) pipeline orchestrated with Vertex AI Pipelines. Use
standard TFX components for data validation and model analysis, and use Vertex AI Pipelines for
model retraining.


## Question 115.

You are an ML engineer at a bank. You have developed a binary classification model using AutoML Tables to
predict whether a customer will make loan payments on time. The output is used to approve or reject loan
requests. One customer’s loan request has been rejected by your model, and the bank’s risks department is
asking you to provide the reasons that contributed to the model’s decision. What should you do?
### Answer:
Use the feature importance percentages in the model evaluation page.

## Question 116.
You are training an object detection model using a Cloud TPU v2. Training time is taking longer than
expected. Based on this simplified trace obtained with a Cloud TPU profile, what action should you take to
decrease training time in a cost-efficient way?
### Answer:
Move from Cloud TPU v2 to Cloud TPU v3 and increase batch size.


## Question 117.
You need to design a customized deep neural network in Keras that will predict customer purchases based on
their purchase history. You want to explore model performance using multiple model architectures, store
training data, and be able to compare the evaluation metrics in the same dashboard. What should you do?
### Answer:
Create an experiment in Kubeflow Pipelines to organize multiple runs
## Question 118.

You work for a gaming company that develops massively multiplayer online (MMO) games. You built a
TensorFlow model that predicts whether players will make in-app purchases of more than $10 in the next two
weeks. The model’s predictions will be used to adapt each user’s game experience. User data is stored in
BigQuery. How should you serve your model while optimizing cost, user experience, and ease of
management?
### Answer:
Import the model into BigQuery ML. Make predictions using batch reading data from BigQuery, and
push the data to Cloud SQL.

## Question 119.

You have been asked to develop an input pipeline for an ML training model that processes images from
disparate sources at a low latency. You discover that your input data does not fit in memory. How should you
create a dataset following Google-recommended best practices?
### Answer:
Convert the images Into TFRecords, store the images in Cloud Storage, and then use the tf. data API to
read the images for training


## Question 120.

Your data science team is training a PyTorch model for image classification based on a pre-trained RestNet
model. You need to perform hyperparameter tuning to optimize for several parameters. What should you do?
Create a Kuberflow Pipelines instance, and run a hyperparameter tuning job on Katib.



# Part 5

GCP Machine Learning Practice Questions Part 4.

[back](#part-1)

<iframe src="https://player.rss.com/ruslanmv/833532" style="width: 100%" title="Machine Learning and Data Science - Podcast" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen><a href="https://rss.com/podcasts/ruslanmv/833532/">Machine Learning in GCP - Part 5 | RSS.com</a></iframe>

## Question 121.

You are developing an ML model that uses sliced frames from video feed and creates bounding boxes around
specific objects. You want to automate the following steps in your training pipeline: ingestion and
preprocessing of data in Cloud Storage, followed by training and hyperparameter tuning of the object model
using Vertex AI jobs, and finally deploying the model to an endpoint. You want to orchestrate the entire
pipeline with minimal cluster management. What approach should you use?
### Answer:
Use Kubeflow Pipelines on Google Kubernetes Engine.
## Question 122.


You recently built the first version of an image segmentation model for a self-driving car. After deploying the
model, you observe a decrease in the area under the curve (AUC) metric. When analyzing the video
recordings, you also discover that the model fails in highly congested traffic but works as expected when there
is less traffic. What is the most likely reason for this result?
### Answer:
Too much data representing congested areas was used for model training


## Question 123.

You are an ML engineer at a global shoe store. You manage the ML models for the company's website. You
are asked to build a model that will recommend new products to the user based on their purchase behavior and
similarity with other users. What should you do?
### Answer:
Build a collaborative-based filtering model

## Question 124.

During batch training of a neural network, you notice that there is an oscillation in the loss. How should you
adjust your model to ensure that it converges?
### Answer:
Increase the learning rate hyperparameter


## Question 125.

You are working on a binary classification ML algorithm that detects whether an image of a classified scanned
document contains a company’s logo. In the dataset, 96% of examples don’t have the logo, so the dataset is
very skewed. Which metrics would give you the most confidence in your model?
### Answer:
F-score where recall is weighed more than precision


## Question 126.

You work with a data engineering team that has developed a pipeline to clean your dataset and save it in a
Cloud Storage bucket. You have created an ML model and want to use the data to refresh your model as soon
as new data is available. As part of your CI/CD workflow, you want to automatically run a Kubeflow Pipelines
training job on Google Kubernetes Engine (GKE). How should you architect this workflow?
### Answer:
Configure a Cloud Storage trigger to send a message to a Pub/Sub topic when a new file is available in a
storage bucket. Use a Pub/Sub-triggered Cloud Function to start the training job on a GKE cluster
## Question 127.

You are an ML engineer at a regulated insurance company. You are asked to develop an insurance approval
model that accepts or rejects insurance applications from potential customers. What factors should you
consider before building the model?
### Answer:
Traceability, reproducibility, and explainability

## Question 128.

You are developing ML models with Al Platform for image segmentation on CT scans. You frequently update
your model architectures based on the newest available research papers, and have to rerun training on the same
dataset to benchmark their performance. You want to minimize computation costs and manual intervention
while having version control for your code. What should you do?
### Answer:
Use Cloud Build linked with Cloud Source Repositories to trigger retraining when new code is pushed
to the repository

## Question 129.

You are developing a Kubeflow pipeline on Google Kubernetes Engine. The first step in the pipeline is to
issue a query against BigQuery. You plan to use the results of that query as the input to the next step in your
pipeline. You want to achieve this in the easiest way possible. What should you do?
### Answer:
Locate the Kubeflow Pipelines repository on GitHub Find the BigQuery Query Component, copy that
component's URL, and use it to load the component into your pipeline. Use the component to execute
queries against BigQuery



## Question 130.
Your company manages an application that aggregates news articles from many different online sources and
sends them to users. You need to build a recommendation model that will suggest articles to readers that are
similar to the articles they are currently reading. Which approach should you use?

### Answer:
Create a collaborative filtering system that recommends articles to a user based on the user’s past
behavior.

## Question 131.
You are building a model to predict daily temperatures. You split the data randomly and then transformed the
training and test datasets. Temperature data for model training is uploaded hourly. During testing, your model
performed with 97% accuracy; however, after deploying to production, the model's accuracy dropped to 66%.
How can you make your production model more accurate?

### Answer:
Split the training and test data based on time rather than a random split to avoid leakage


## Question 132.
You are a lead ML engineer at a retail company. You want to track and manage ML metadata in a centralized
way so that your team can have reproducible experiments by generating artifacts. Which management solution
should you recommend to your team?

### Answer:
Store all ML metadata in Google Cloud’s operations suite.


## Question 133.
Your team trained and tested a DNN regression model with good results. Six months after deployment, the
model is performing poorly due to a change in the distribution of the input data. How should you address the
input differences in production?

### Answer:
Create alerts to monitor for skew, and retrain the model.

## Question 134.
You work for a large social network service provider whose users post articles and discuss news. Millions of
comments are posted online each day, and more than 200 human moderators constantly review comments and
flag those that are inappropriate. Your team is building an ML model to help human moderators check content
on the platform. The model scores each comment and flags suspicious comments to be reviewed by a human.
Which metric(s) should you use to monitor the model’s performance?

### Answer:
Number of messages flagged by the model per minute confirmed as being inappropriate by humans.

## Question 135.
You are responsible for building a unified analytics environment across a variety of on-premises data marts.
Your company is experiencing data quality and security challenges when integrating data across the servers,
caused by the use of a wide range of disconnected tools and temporary solutions. You need a fully managed,
cloud-native data integration service that will lower the total cost of work and reduce repetitive work. Some
members on your team prefer a codeless interface for building Extract, Transform, Load (ETL) process.
Which service should you use?

### Answer:
Cloud Data Fusion


## Question 136.

You are an ML engineer at a bank that has a mobile application. Management has asked you to build an
ML-based biometric authentication for the app that verifies a customer's identity based on their fingerprint.
Fingerprints are considered highly sensitive personal information and cannot be downloaded and stored into the bank databases. Which learning strategy should you recommend to train and deploy this ML model?

### Answer:
Federated learning
## Question 137.
Your company manages a video sharing website where users can watch and upload videos. You need to
create an ML model to predict which newly uploaded videos will be the most popular so that those videos can
be prioritized on your company’s website. Which result should you use to determine whether the model is
successful?

### Answer:
The model predicts 95% of the most popular videos measured by watch time within 30 days of being
uploaded.


## Question 138.
You are designing an ML recommendation model for shoppers on your company's ecommerce website. You
will use Recommendations Al to build, test, and deploy your system. How should you develop
recommendations that increase revenue while following best practices?

### Answer:
Use the "Frequently Bought Together' recommendation type to increase the shopping cart size for each
order.

  You've reached the end of another episode of the Data Science.
  **Thanks for reading us!**  Thanks again, and I’ll see you next time!
