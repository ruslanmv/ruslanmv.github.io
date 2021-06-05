---
title: "Step Functions API Gateway, Lambda and S3 in AWS"
excerpt: " Building a Serverless Application Using Step Functions API Gateway, Lambda and S3 in AWS "

header:
  image: "../assets/images/posts/2021-05-18-Apache%20Spark%20and%20Scala%20in%20AWS/web_mg_9912_1.jpg"
  teaser: "../assets/images/posts/2021-05-18-Apache%20Spark%20and%20Scala%20in%20AWS/web_mg_9912_1.jpg"
  caption: "Amazon Headquarters, Milan"
  
---



Hello, today I am going to  follow the tutorial by Linux Academy to  Building a Serverless Application Using Step Functions API Gateway, Lambda and S3 in AWS .



What we want do perform is 



![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/0.jpg)



All of the resources needed to complete this lab are available from [this GitHub repo](https://github.com/julielkinsfembotit/LALabs).

## Create the `email` Lambda Function

1. Navigate to Lambda in the AWS console, and click **Create function**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/1.jpg)

1. Make sure **Author from scratch** is selected.
2. Set the *Function name* to **email**, and the runtime to **python 3.7**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/2.jpg)

1. Expand **Choose or create an execution role**, select **Use an existing role**, and pick our **LambdaRuntimeRole** from the dropdown.

2. Click **Create function**.

3. Scroll down to the function code section, and replace the code in the lambda_function area and remplace with

   

   ```python
     
   import boto3
   
   VERIFIED_EMAIL = 'YOUR_SES_VERIFIED_EMAIL'
   
   ses = boto3.client('ses')
   
   def lambda_handler(event, context):
       ses.send_email(
           Source=VERIFIED_EMAIL,
           Destination={
               'ToAddresses': [event['email']]  # Also a verified email
           },
           Message={
               'Subject': {'Data': 'A reminder from your reminder service!'},
               'Body': {'Text': {'Data': event['message']}}
           }
       )
       return 'Success!'
   ```

   

   

   

   

   - To get the code, navigate to that file in the repository, then click the **Raw** button. Copy the text on the next screen, then paste it back in the Lambda window (in the *Function code* section) we've got open.

4. Open up a new browser tab, and leave this one open. We'll be back in a minute.

### Set up an Email Address

1. Navigate to Simple Email Service.

2. Click

    

   Email Addresses

   , then

    

   Verify a New Email Address

   . Enter an email address to use, then click

    

   Verify this address

   .

   - Check the email address, open up the verification email that AWS sent, and click the link. We should land on a *Congratulations!* page telling us that the email address has been verified.

### Finish the Lambda Function Setup

1. Change `YOUR_SES_VERIFIED_EMAIL` to the email address that just got verified.
2. Click **Save**.

## Create the `sms` Lambda Function

1. Navigate again to the main Lambda page in the AWS console, and click **Create function**.

2. Make sure **Author from scratch** is selected.

3. Set the *Function name* to **sms**, and the runtime to **python 3.7**.

4. Expand **Choose or create an execution role**, select **Use an existing role**, and pick our **LambdaRuntimeRole** from the dropdown.

5. Click **Create function**.

6. Scroll down to the

    

   Function code

    

   section, and replace the code in the

    

   lambda_function

    

   area with what's in the

    

   ```
   sms_reminder.py
   ```

    

   ```python
   import boto3
   
   sns = boto3.client('sns')
   
   def lambda_handler(event, context):
       sns.publish(PhoneNumber=event['phone'], Message=event['message'])
       return 'Success!'
   ```

   file in the GitHub repository.

   - To get the code, navigate to that file in the repository, then click the **Raw** button. Copy the text on the next screen, then paste it back in the Lambda window (*Function code* section) we've got open.

7. Click **Save**.

## Create the `api_handler` Lambda Function

1. Navigate again to the main Lambda page in the AWS console, and click **Create function**.

2. Make sure **Author from scratch** is selected.

3. Set the *Function name* to **api_handler**, and the runtime to **python 3.7**.

4. Expand **Choose or create an execution role**, select **Use an existing role**, and pick our **LambdaRuntimeRole** from the dropdown.

5. Click **Create function**.

6. Scroll down to the

    

   Function code

    

   section, and replace the code in the

    

   lambda_function

    

   area with what's in the

    

   ```
   api_handler.py
   ```

    

   ```python
   import boto3
   import json
   import os
   import decimal
   
   SFN_ARN = 'STEP_FUNCTION_ARN'
   
   sfn = boto3.client('stepfunctions')
   
   def lambda_handler(event, context):
       print('EVENT:')
       print(event)
       data = json.loads(event['body'])
       data['waitSeconds'] = int(data['waitSeconds'])
       
       # Validation Checks
       checks = []
       checks.append('waitSeconds' in data)
       checks.append(type(data['waitSeconds']) == int)
       checks.append('preference' in data)
       checks.append('message' in data)
       if data.get('preference') == 'sms':
           checks.append('phone' in data)
       if data.get('preference') == 'email':
           checks.append('email' in data)
   
       # Check for any errors in validation checks
       if False in checks:
           response = {
               "statusCode": 400,
               "headers": {"Access-Control-Allow-Origin":"*"},
               "body": json.dumps(
                   {
                       "Status": "Success", 
                       "Reason": "Input failed validation"
                   },
                   cls=DecimalEncoder
               )
           }
       # If none, run the state machine and return a 200 code saying this is fine :)
       else: 
           sfn.start_execution(
               stateMachineArn=SFN_ARN,
               input=json.dumps(data, cls=DecimalEncoder)
           )
           response = {
               "statusCode": 200,
               "headers": {"Access-Control-Allow-Origin":"*"},
               "body": json.dumps(
                   {"Status": "Success"},
                   cls=DecimalEncoder
               )
           }
       return response
   
   # This is a workaround for: http://bugs.python.org/issue16535
   class DecimalEncoder(json.JSONEncoder):
       def default(self, obj):
           if isinstance(obj, decimal.Decimal):
               return int(obj)
           return super(DecimalEncoder, self).default(obj)
   ```

   file in the GitHub repository.

   - To get the code, navigate to that file in the repository, then click the **Raw** button. Copy the text on the next screen, then paste it back in the Lambda window (in the *Function code* section) we've got open.

7. Click

    

   Save

   .

   - Note that we'll have to come back and edit this a bit, changing the `SFN_ARN`.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/7.jpg)





## Create a Step Function State Machine

1. In a new browser tab, navigate to the Step Functions console. Click the **here** in *You can also skip and access more functionality here* up at the top of the page.
2. Make sure **Author with code snippets** is selected, and leave the *Type* set to **Standard**.
3. In our GitHub browser tab, navigate to the `step-function-template.json` file, click the **Raw** button, and copy the next on the resulting screen.



```json
{
  "Comment": "An example of the Amazon States Language using a choice state.",
  "StartAt": "SendReminder",
  "States": {
    "SendReminder": {
      "Type": "Wait",
      "SecondsPath": "$.waitSeconds",
      "Next": "ChoiceState"
    },
    "ChoiceState": {
      "Type" : "Choice",
      "Choices": [
        {
          "Variable": "$.preference",
          "StringEquals": "email",
          "Next": "EmailReminder"
        },
        {
          "Variable": "$.preference",
          "StringEquals": "sms",
          "Next": "TextReminder"
        },
        {
          "Variable": "$.preference",
          "StringEquals": "both",
          "Next": "BothReminders"
        }
      ],
      "Default": "DefaultState"
    },

    "EmailReminder": {
      "Type" : "Task",
      "Resource": "EMAIL_REMINDER_ARN",
      "Next": "NextState"
    },

    "TextReminder": {
      "Type" : "Task",
      "Resource": "TEXT_REMINDER_ARN",
      "Next": "NextState"
    },
    
    "BothReminders": {
      "Type": "Parallel",
      "Branches": [
        {
          "StartAt": "EmailReminderPar",
          "States": {
            "EmailReminderPar": {
              "Type" : "Task",
              "Resource": "EMAIL_REMINDER_ARN",
              "End": true
            }
          }
        },
        {
          "StartAt": "TextReminderPar",
          "States": {
            "TextReminderPar": {
              "Type" : "Task",
              "Resource": "TEXT_REMINDER_ARN",
              "End": true
            }
          }
        }
      ],
      "Next": "NextState"
    },
    
    "DefaultState": {
      "Type": "Fail",
      "Error": "DefaultStateError",
      "Cause": "No Matches!"
    },

    "NextState": {
      "Type": "Pass",
      "End": true
    }
  }
}
```



1. Paste that code into the *Definition* section in the Step Functions window we have open.

2. Replace any occurrences of

    

   ```
   EMAIL_REMINDER_ARN
   ```

    

   with the ARN of the

    

   ```
   email
   ```

    

   Lambda function, and any occurrences of

    

   ```
   TEXT_REMINDER_ARN
   ```

    

   with the ARN of the

    

   ```
   sms
   ```

    

   Lambda function.

   - You'll have to navigate into each of those functions, in the Lambda console, to get their ARNs. Just get into each function's configuration page, and the ARN is in the upper right of the screen.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/10.jpg)





![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/11.jpg)







1. Click the refresh button, to the right of the code pane, to update the diagram.



![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/12.jpg)



1. Click **Next**
2. On the next form, **MyStateMachine** is fine for a *State machine name*. Select **Choose an existing IAM role** in the *Permissions* section, and choose **RoleForStepFunction** from the dropdown.
3. Click **Create state machine**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/13.jpg)

1. Copy the state machine's ARN that shows on the next screen.



![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/14.jpg)

1. Back in our

    

   ```
   api_handler
   ```

    

   page (which should still be open in a browser tab), in the

    

   Function code

    

   section, we need to replace

    

   ```
   STEP_FUNCTION_ARN
   ```

    

   with the actual ARN we just copied.

   - It should read like this: `SFN_ARN = WhatWeJustCopied`

2. Click **Save**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/15.jpg)

## Create the API Gateway

1. In the AWS console, navigate to API Gateway, click **Get Started**, and then click **OK** on any welcome message that might pop up.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/16.jpg)

1. Select  on REST API  **Buid** and select **REST** as a *Protocol*, and **New API** right below that.
2. Set the *API name* and *Description* to **reminders**, and set the *Endpoint Type* to **Regional**.
3. Click **Create API**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/17.jpg)

1. Click the **Actions** dropdown and select **Create Resource**.
2. ![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/18.jpg)
3. Call it **reminders**, and check the box to enable **Enable API Gateway CORS**.
4. Click **Create Resource**.



![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/19.jpg)

1. Select **/reminders**, and click **Actions** > **Create method**.

   ![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/20.jpg)

2. In the dropdown under */reminders* (in the *Resources* pane), select **POST** and then click the checkmark next to it.

3. With

    

   POST

    

   selected, set the form to the right like this:

   - *Integration type:* **Lambda Function**
   - *Use Lambda Proxy Integration:* Checked
   - *Lambda Region:* **us-east-1**
   - *Lambda Function:* **api_handler**

4. Click **Save**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/21.jpg)

1. Click **OK** to allow API Gateway permission to invoke the `api_handler` function.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/22.jpg)

1. Select **Actions** and **Deploy API**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/23.jpg)

1. Set *Deployment stage* to **New Stage**.
2. Give it a *Stage name* and *Stage description* of **prod**.
3. Click **Deploy**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/24.jpg)

1. 
2. Note the **Invoke URL** that is now at the top of the page. Leave this page open, because we will need this soon.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/25.jpg)

## Create the Static S3 Website

### Configure the Static Website

1. Open the S3 console.
2. Click **Create bucket**.
3. Give the bucket a unique name, and ensure the region is **US East (N. Virginia)**.
4. Click **Next**.
5. Click **Next** again.
6. Ensure *no* checkmarks are selected on the block public access screen. This includes individual ones and any group check mark. In the window that pops up, check the box to acknowledge that these settings may make this bucket and its objects public.
7. Click **Next** and then **Create bucket**.'

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/28.jpg)

1. Open the bucket.

2. Click

    

   Upload

   - If you haven't already, download the GitHub repository.

   https://github.com/Cloud-Data-Science/lab-building-a-serverless-application-using-step-functions-api-gateway-lambda-and-s3-in-aws/tree/master/static_website

   

   - Then click **Add files**, select the all of the files from the `static_website` folder, and click **Next**.

   ![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/29.jpg)

3. Change the *Manage public permissions* dropdown to **Grant public read access to this object(s)**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/30.jpg)

1. Click **Next**, then **Next** again, and finally **Upload**.
2. Now click the **+ Create folder** button, give it a name of **images**, and click **Save**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/31.jpg)

1. Click on the newly created `images` folder and click **Upload**.
2. Browse to the downloaded GitHub repository and select `cat.png`.
3. Change the *Manage public permissions* dropdown to **Grant public read access to this object(s)**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/32.jpg)

1. Click **Next**, then **Next** again, and finally **Upload**.
2. Navigate back out to the main bucket screen, and click the **Properties** tab.
3. Click **Static website hosting**.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/33.jpg)

1. Select **Use this bucket to host a website**.
2. Enter `index.html` in the Index Document box, and `error.html` in the Error Document box.



![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/34.jpg)

1. Click **Save**.
2. Select **Static website hosting** again and click on the *Endpoint* URL.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/35.jpg)



![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/36.jpg)





1. Test it out:
   - Fill out the form underneath the cat picture:
     - *Seconds to wait:* **1**
     - *Message:* **hello**
     - *Email address:* Just enter a valid email here
     - *Telephone:* We can leave this blank
     - Click the **email** button below the form.
   - We'll get an error, because we didn't update `formlogic.js`

### Updating `formlogic.js`

1. Back in the API Gateway browser tab that we should still have open, copy that *Invoke URL*.
2. With a text editor, find `formlogic.js` in the downloaded GitHub repository.
3. Replace the `YOUR_API_ENDPOINT_URL` placeholder text with the invoke URL we just copied, and add `/reminders` to the end of it.

![](../assets/images/posts/2021-05-24-%20Step%20Functions%20API%20Gatway,%20Lambda%20and%20S3%20in%20AWS/37.jpg)

1. Upload it to the S3 bucket like we did before (replacing the existing `formlogic.js` that's already up there).

2. Get back into the browser tab we had open to the *Endpoint* URL, and fill out the form again, using the same values we did before. This time when we click the **email** button, we should get both a *Looks ok* and a *{"Status":"Success"}* message (one above and one below the form).

3. Now let's check our email, and we should have one sitting there with a subject line of *A reminder from your reminder service!*. If we click into the email, we'll see our *hello* message.

4. If we'd chosen

    

   sms

    

   or

    

   both

    

   when choosing which method to use for notification, then we need to make sure the phone number we type includes a

    

   +1

    

   and area code before the rest of the phone number, like this:

   - **+19998675309**

## Seeing What Happened

If we want to see the process that occurred, navigate to Step Functions in the AWS console, then **State machines**, and click on **MyStateMachine**. Down in the *Executions* section, we should see one with a status of *Succeeded*. Click on that event's name, and we can look at the *Visual workflow* diagram. It will show the order of events that occurred when we submitted the form.

## Conclusion

We set up a little application uses several AWS components to send us reminder messages based on what we enter into a web form. Congratulations!























**Congratulation** we have  practiced Scala and  Apache Spark with **IntelliJ** 











