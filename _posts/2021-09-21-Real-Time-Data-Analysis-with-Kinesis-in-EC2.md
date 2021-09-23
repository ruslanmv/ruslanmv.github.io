---
title: "How to perform Real-Time Data Analysis with Kinesis in EC2"
excerpt: "Real-Time Data Analysis with Kinesis in EC2"

header:
  image: "../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/photo2.jpg"
  teaser: "../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/photo2.jpg"
  caption: "Business & Work"
  
---

We are going  to analyze the user ratings of a TV  streaming movies. We are interested to know   what are most popular captains characters  and identify any data anomalies.  Kinesis Data Analytics is a perfect pair for this task.

## Introduction

 Amazon Kinesis can be used as  service for real-time application monitoring, fraud detection, and live leader-boards. 

We can  ingest streaming data using Kinesis Data Streams, process it using Kinesis Data Analytics, and emit the results to any data store or application using Kinesis Data Streams with millisecond end-to-end latency. 

With Amazon Kinesis services, we can perform real-time analytics on data that has been traditionally analyzed using batch processing. Common streaming use cases include sharing data between different applications, streaming extract-transform-load, and real-time analytics. For example, you can use Kinesis Data Firehose to continuously load streaming data into your S3 data lake or analytics services



We can collect  and store the Data with **Kinesis Data Streams** then Process with Kinesis **Data Firehose** and the analize with **Kinesis Data Analytics**



![](../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/workd.png)

In this blog post we are going to perfom the following tasks:

1. Create a Kinesis Data Firehose Stream
2. Send Data to the Stream
3. Find Averages of the Data Per Minute
4. Find Anomalies in the Data



<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2012.55.09.png" alt="Screenshot 2021-09-22 at 12.55.09" style="zoom:50%;" />



### Create a Kinesis Data Firehose Stream

1. Log in to the AWS console  [https://aws.amazon.com/it/console/](https://aws.amazon.com/it/console/).
2. Please make sure you are in the `us-east-1` (N. Virginia) region when in the AWS console.
3. We are going to send data to S3,  so we need a way to receive the user data.
4. In the search bar on top, enter "firehose".<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2012.52.49.png" alt="Screenshot 2021-09-22 at 12.52.49" style="zoom:50%;" />

Select **Kinesis Data Firehose** from the search results.

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2012.53.20.png" alt="Screenshot 2021-09-22 at 12.53.20" style="zoom:50%;" />

1. Click **Create Delivery Stream**.
2. Under *Delivery stream name*, enter "captains-kfh".
3. Select **Direct PUT or other sources**.



<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2012.55.56.png" alt="Screenshot 2021-09-22 at 12.55.56" style="zoom:50%;" />

- Under *S3 Bucket*, click **Create New**.

In *S3 bucket name*, enter "captains-data" plus some random characters for uniqueness.

- Click **Create S3 bucket**.

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2012.58.11.png" alt="Screenshot 2021-09-22 at 12.58.11" style="zoom:50%;" />

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2012.59.18.png" alt="Screenshot 2021-09-22 at 12.59.18" style="zoom:50%;" />



- Under *Buffer size*, enter "1".
- Under *Buffer interval*, enter "60".

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2013.01.25.png" alt="Screenshot 2021-09-22 at 13.01.25" style="zoom:50%;" />



Click **Create delivery stream**.



<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2013.02.06.png" alt="Screenshot 2021-09-22 at 13.02.06" style="zoom:50%;" />



<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2013.02.40.png" alt="Screenshot 2021-09-22 at 13.02.40" style="zoom:50%;" />

## Ingestion of Fake Data to simulate the Streaming



In this part we will setup and EC2 instance to send Data to the Stream

Open the Amazon EC2 console at [https://console.aws.amazon.com/ec2/.](https://console.aws.amazon.com/ec2/.)

![Screenshot 2021-09-22 at 13.05.07](../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2013.05.07.png)

Choose **Launch Instance**.

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2013.05.43.png" alt="Screenshot 2021-09-22 at 13.05.43" style="zoom:50%;" />

Choose an Amazon Machine Image (AMI)**, find an Amazon Linux 2 AMI at the top of the list and choose **Select**.

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2013.06.06.png" alt="Screenshot 2021-09-22 at 13.06.06" style="zoom:50%;" />

Choose an Instance Type**, choose **Next: Configure Instance Details**.

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2013.06.47.png" alt="Screenshot 2021-09-22 at 13.06.47" style="zoom:50%;" />

 Configure Instance Details**, provide the following information:

1. Leave **Number of instances** at one.
2. Leave **Purchasing option** at the default setting.
3. For **Network**, choose the entry for the same VPC that you noted when you created your EFS file system in [ Create your Amazon EFS file system](https://docs.aws.amazon.com/efs/latest/ug/gs-step-two-create-efs-resources.html).
4. For **Subnet**, choose a default subnet in any Availability Zone.
5. For **File systems**, make sure that the EFS file system that you created in [ Create your Amazon EFS file system](https://docs.aws.amazon.com/efs/latest/ug/gs-step-two-create-efs-resources.html) is selected. The path shown next to the file system ID is the mount point that the EC2 instance will use, which you can change.
6. The **User data** automatically includes the commands for mounting your Amazon EFS file system.

Choose **Next: Add Storage**.

Choose **Next: Add Tags**.

Name your instance and choose **Next: Configure Security Group**.

 Configure Security Group**, set **Assign a security group** to **Select an existing security group**. Choose the default security group to make sure that it can access your EFS file system.

You can't access your EC2 instance by Secure Shell (SSH) using this security group. SSH access isn't required for this exercise. To add access by SSH later, you can edit the default security and add a rule to allow SSH. Or you can create a new security group that allows SSH. You can use the following settings to add SSH access:

- **Type:** SSH
- **Protocol:** TCP
- **Port Range:** 22
- **Source:** Anywhere 0.0.0.0/0

1. Choose **Review and Launch**.

2. Choose **Launch**.

3. Select the check box for the key pair that you created, and then choose **Launch Instances**.

Additionally we can create a group and select the policies to be attached to the group. Using groups is a best-practice way to manage users' permissions by job functions, AWS service access, or your custom permissions.

Policy name:

**AmazonS3FullAccess**

**AmazonKinesisFullAccess**

**AmazonEC2FullAccess**

During the creation of the user of the EC2 instance we can download the .csv file of credentials that you will use that will be used later.

### Send Data to the Stream

We simulate user data being sent to the stream and verify that it is output to S3

1. Open an SSH client.

2. Locate your private key file. The key used to launch this instance is kinesis.pem

3. Run this command, if necessary, to ensure your key is not publicly viewable.

   ```
   chmod 400 kinesis.pem
   ```

   

4. Connect to your instance using its Public DNS, for example , ec2-34-202-159-123.compute-1.amazonaws.com



```
 ssh -i "kinesis.pem" ec2-user@ec2-34-202-159-123.compute-1.amazonaws.com
```



When prompted, enter "y" for yes.

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2014.04.58.png" alt="Screenshot 2021-09-22 at 14.04.58" style="zoom:50%;" />

### Installing the Generator of Data with Python 3  with the Boto 3

 Perform a yum **check-update** to refresh the package index. The check-update also looks for available updates. 

```
sudo yum update -y
```

 Run **list installed** to determine if Python 3 is already installed on the host.

Python 3 already installed output example:

```plainText
[ec2-user ~]$ yum list installed | grep -i python3
```

```
python3.x86_64                        3.7.10-1.amzn2.0.1             installed  
python3-daemon.noarch                 2.2.3-8.amzn2.0.2              installed  
python3-docutils.noarch               0.14-1.amzn2.0.2               installed  
python3-libs.x86_64                   3.7.10-1.amzn2.0.1             installed  
python3-lockfile.noarch               1:0.11.0-17.amzn2.0.2          installed  
python3-pip.noarch                    20.2.2-1.amzn2.0.3             installed  
python3-pystache.noarch               0.5.4-12.amzn2.0.1             installed  
python3-setuptools.noarch             49.1.3-1.amzn2.0.2             installed  
python3-simplejson.x86_64             3.2.0-1.amzn2.0.2              installed  
```

If is not installed

```
sudo yum install python3 -y
```

**Create a virtual environment under the ec2-user home directory**

The following command creates the app directory with the virtual environment inside of it. You can change **my_app** to another name. If you change **my_app**, make sure that you reference the new name in the remaining resolution steps.

```plainText
python3 -m venv my_app/env
```

```
source ~/my_app/env/bin/activate
```

Next we make sure that you have the latest pip module installed within your environment.

```plainText
pip install pip --upgrade
```

Use the **pip** command to install the Boto 3 library within our virtual environment.

```plainText
pip install boto3
```

```
pip install faker
```

```
pip install json
```

Use the **deactivate** command to exit the virtual environment.

```plainText
deactivate
```

 To activate the virtual environment automatically when you log in, add it to the **~/.bashrc** file.

```plainText
 echo "source ${HOME}/my_app/env/bin/activate" >> ${HOME}/.bashrc
```

Source the **~/.bashrc** file in your home directory to reload your environment's bash environment. Reloading automatically activates your virtual environment. The prompt reflects the change (**env**). This change also applies to any future SSH sessions.

```
source ~/.bashrc
```

### AWS Command Line Interface (AWS CLI) Setup

For general use, the `aws configure` command is the fastest way to set up your AWS CLI installation. When you enter this command, the AWS CLI prompts you for four pieces of information:



The AWS CLI stores this information in a *profile* (a collection of settings) named `default` in the `credentials` file. By default, the information in this profile is used when you run an AWS CLI command that doesn't explicitly specify a profile to use. For more information on the `credentials` file, see [Configuration and credential file settings](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)

The following example shows sample values. Replace them with your own values as described in the following sections.

```
$ aws configure
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-west-2
Default output format [None]: json
```

**Create a generator of data by using Python**

We download first the application

```
curl -OL https://github.com/ruslanmv/Performing-Real-Time-Data-Analysis-with-Kinesis/raw/master/send_captains_to_cloud.py
```

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2014.48.08.png" alt="Screenshot 2021-09-22 at 14.48.08" style="zoom:50%;" />

Open up the `send_captains_to_cloud_` file to view the code:

```
nano send_captains_to_cloud.py
```

![](../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/image.png)

Clear the screen:

```
ctrl + x 
```

Run the script:

```
python3 send_captains_to_cloud.py
```

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.06.43.png" alt="Screenshot 2021-09-22 at 16.06.43" style="zoom:40%;" />

1. Go back to the Kinesis Firehose tab. Our delivery stream should now have a status of Active.

2. Click **captains-kfh**.

3. Click the **Monitoring** tab.

4. Switch to a one-hour view by clicking **1h**. The graphs may take a few minutes to populate with data points.

   <img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.08.56.png" alt="Screenshot 2021-09-22 at 16.08.56" style="zoom:50%;" />

5. Go back to the terminal and stop the script by pressing any key.

6. Go back to the Kinesis Firehose tab.

7. Click on **Kinesis Firehouse delivery streams** and then click on **captains-kfh**. Alternatively, you can just click on the **Details** tab. 

8. Under *S3 bucket* in *Amazon S3 destination*, click on the link to the bucket to view the S3 bucket.

9. Go back to the terminal to pull some of the data down.

10. Find the bucket name:

    ```
    aws s3 ls
    ```

    

11. Copy the bucket name.

12. View the bucket contents:

    ```
    aws s3 sync s3://<INSERT_BUCKET_NAME> .
    ```

13. View the folder structure:

    ```
    ll
    ```

14. Change the working directory to the folder for the current year, month, day, and hour:

    ```
    cd <YEAR>/<MONTH>/<DAY>/<HOUR>/
    ```

15. View the files in the folder:

    ```
    ll
    ```

16. Copy one of the file names.

17. View the first 200 characters of the file:

    ```
    cut -c -200 <INSERT_FILE_NAME>
    ```

    ![Screenshot 2021-09-22 at 16.18.28](../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.18.28.png)

18. Return to the home directory:

    ```
    cd
    ```

19. Start the data-generating stream again:

    ```
    python3 send_captains_to_cloud.py
    ```

### Find the Average Captain Ratings

We analyze the stream data to find the minute-by-minute popularity of our

users favorite  captains.

1. Go back to the Kinesis Firehose tab.
2. In the menu on the left, click **Data Analytics**.
3. Click **Create application**.![Screenshot 2021-09-22 at 16.20.09](../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.20.09.png)
4. Under *Application name*, enter "popular-captains".
5. Under *Description*, enter "Our users' favorite captains".



Click **Create application**.

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.23.44.png" alt="Screenshot 2021-09-22 at 16.23.44" style="zoom:50%;" />

Click **Connect streaming data**.

Select **Kinesis Firehose delivery stream**.

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.25.17.png" alt="Screenshot 2021-09-22 at 16.25.17" style="zoom:50%;" />

1. In the *Kinesis Firehose delivery stream* dropdown menu, select the **captains-kfh** stream.

   <img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.25.45.png" alt="Screenshot 2021-09-22 at 16.25.45" style="zoom:40%;" />

2. Select **Choose from IAM roles that Kinesis Analytics can assume**.

3. In the *IAM role* dropdown menu, select the provided IAM role for this lab.

   <img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.27.02.png" alt="Screenshot 2021-09-22 at 16.27.02" style="zoom:50%;" />

4. Click **Discover schema**.

   <img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.27.39.png" alt="Screenshot 2021-09-22 at 16.27.39" style="zoom:50%;" />

5. Click **Save and continue**.

   <img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.28.17.png" alt="Screenshot 2021-09-22 at 16.28.17" style="zoom:50%;" />

6. Under *Real time analytics*, click **Run**.

7. Click **Yes, start application**.

<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.29.57.png" alt="Screenshot 2021-09-22 at 16.29.57" style="zoom:40%;" />





<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.30.12.png" alt="Screenshot 2021-09-22 at 16.30.12" style="zoom:40%;" />



1. Create a query using the SQL Editor to show the average rating and total rating of each captain per minute, At the end of the query, change the last line to order the average rating by descending order:

   ```sql
   CREATE OR REPLACE STREAM "CAPTAIN_SCORES" ("favoritecaptain" VARCHAR(32), average_rating DOUBLE, total_rating INTEGER);
   
   CREATE OR REPLACE PUMP "STREAM_PUMP" AS
   INSERT INTO "CAPTAIN_SCORES"
   SELECT STREAM "favoritecaptain", avg("rating") as average_rating, sum("rating") as total_rating
   FROM "SOURCE_SQL_STREAM_001"
   GROUP BY "favoritecaptain", STEP("SOURCE_SQL_STREAM_001".ROWTIME BY INTERVAL '1' MINUTE)
   ORDER BY STEP("SOURCE_SQL_STREAM_001".ROWTIME BY INTERVAL '1' MINUTE), avg("rating") DESC;
   ```



Click **Save and run application** . The table will need to be recreated, which will take a few minutes before it is ready to view.



![Screenshot 2021-09-22 at 16.51.50](../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.51.50.png)





<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.52.08.png" alt="Screenshot 2021-09-22 at 16.52.08" style="zoom:50%;" />

### Find Anomalous Captain Ratings

Here, there might some anomalous values in the stream. We create an analysis

to find this data in real time.

1. Delete the SQL code for popular captains.

2. Create a new query that will rank the incoming captain ratings by how anomalous the rating is, displaying the most anomalous values first:

   ```
   CREATE OR REPLACE STREAM "RAW_ANOMALY_STREAM" (
      "favoritecaptain" VARCHAR(32),
      "rating"          INTEGER,
      "ANOMALY_SCORE"   DOUBLE);
   
   CREATE OR REPLACE PUMP "RAW_PUMP" AS INSERT INTO "RAW_ANOMALY_STREAM"
   SELECT STREAM "favoritecaptain", "rating", "ANOMALY_SCORE" FROM
   TABLE(RANDOM_CUT_FOREST(
     CURSOR(SELECT STREAM "favoritecaptain", "rating" FROM "SOURCE_SQL_STREAM_001")
   ));
   
   CREATE OR REPLACE STREAM "ORDERED_ANOMALY_STREAM" (
      "favoritecaptain" VARCHAR(32),
      "rating"          INTEGER,
      "ANOMALY_SCORE"   DOUBLE);
   
   -- Sort records by descending anomaly score, insert into output stream
   CREATE OR REPLACE PUMP "ORDERED_PUMP" AS INSERT INTO "ORDERED_ANOMALY_STREAM"
   SELECT STREAM * FROM "RAW_ANOMALY_STREAM"
   ORDER BY FLOOR("RAW_ANOMALY_STREAM".ROWTIME TO SECOND), "ANOMALY_SCORE" DESC;
   ```

3. Click **Save and run application**. The table will take a few minutes to populate.



<img src="../assets/images/posts/2021-09-21-Real-Time-Data-Analysis-with-Kinesis-in-EC2/Screenshot%202021-09-22%20at%2016.55.45.png" alt="Screenshot 2021-09-22 at 16.55.45" style="zoom:50%;" />



**Congratulations !**  We have created a Kinesis Data Firehose Stream we sent Data to the Stream and found Averages of the Data and  Anomalies in the Data.

