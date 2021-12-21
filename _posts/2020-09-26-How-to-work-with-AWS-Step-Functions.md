---
title: "How to work with AWS Step Functions"
excerpt: "How to work with AWS Step Functions"

header:
  image: "/assets/images/computer3.jpeg"
  teaser: "/assets/images/computer3.jpeg"
  caption: "We are all now connected by the Internet, like neurons in a giant brain. Stephen Hawking"
  
---

# How to work with AWS Step Functions

In this blog post, I will discuss how to get start with **AWS Step Functions**. Why use it and how to use it.

**AWS Step Functions** is a low-code, visual workflow service that developers use to build distributed applications, automate IT and business processes, and build data and machine learning pipelines using AWS services.

Step Functions are actually an extension of the Lambda function, allowing you to combine several Lambda functions to call each other.

## Getting started

To effectively design and implement workflows in AWS Step Functions first let us practice, let us login to our  AWS web console

 [https://aws.amazon.com/console/](https://aws.amazon.com/console/)

 and let us create a step function.

<img src="/assets/images/posts/2020-09-26-How-to-work-with-AWS-Step-Functions/1.jpg" style="zoom:50%;" />

Then we create a **state machine**



<img src="/assets/images/posts/2020-09-26-How-to-work-with-AWS-Step-Functions/2.jpg" style="zoom:50%;" />



We select **write workflow code**

<img src="/assets/images/posts/2020-09-26-How-to-work-with-AWS-Step-Functions/3.jpg" alt="3" style="zoom:50%;" />

and let us just continue with the **Hello World** example

![](/assets/images/posts/2020-09-26-How-to-work-with-AWS-Step-Functions/4.jpg)

we choose create with the default settings and create.

 We select our **State machine** and press **Start Execution**

![](/assets/images/posts/2020-09-26-How-to-work-with-AWS-Step-Functions/5.jpg)

We  keep the default **JSON** input and click **Start Execution**

![](/assets/images/posts/2020-09-26-How-to-work-with-AWS-Step-Functions/6.jpg)

 A Step Functions execution receives a **JSON** text as input and passes that input to the first state in the workflow.    Individual states receive **JSON** as input and usually pass **JSON** as output to the next state.

<img src="/assets/images/posts/2020-09-26-How-to-work-with-AWS-Step-Functions/7.jpg" style="zoom:50%;" />



After the execution you can see that was **Succeeded** and the deployed picture as below

<img src="/assets/images/posts/2020-09-26-How-to-work-with-AWS-Step-Functions/8.jpg" alt="8" style="zoom:50%;" />





**Good!** Now we have created and executed our Hello World  Step function. Let us now analyze in detail how this information flows from state to state, and learning how to filter and manipulate this data.

### States

A state can be any string but must be unique within the scope of the entire state machine. It does the following functions:

- Performs some work in the state machine (a Task state).
- Makes a choice between branches of execution (a [Choice](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-choice-state.html) state).
- Stops execution with failure or success (a [Fail](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-fail-state.html) or [Succeed](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-succeed-state.html) state).
- Simply passes its input to its output or injects some fixed data (a [Pass](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-pass-state.html) state).
- Provides a delay for a certain amount of time or until a specified time/date (a [Wait](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-wait-state.html) state).
- Begins parallel branches of execution (a [Parallel](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-parallel-state.html) state).

From our Hello World **JSON**  example for our step function definition   ,

we have a `Comment` of the step fuction, `StartAt`, we select where starts

and settings of the states in `States`, the result in `Result` and the second state

with `Next` following with the word **World**. This name should be used as the name of the next state otherwise is wrong the definition, an finally finish with `End`

```
{
  "Comment": "A Hello World example of the Amazon States Language using Pass states",
  "StartAt": "Hello",
  "States": {
    "Hello": {
      "Type": "Pass",
      "Result": "Hello",
      "Next": "World"
    },
    "World": {
      "Type": "Pass",
      "Result": "World",
      "End": true
    }
  }
}
```

and our execution input is our `State-input`

```
{
  "Comment": "Insert your JSON here"
}
```

### Task type

An example of a state definition for Task type:

```
"States": {
"FirstState": {
"Type": "Task",
"Resource": "arn:aws:lambda:ap-southeast-2:710187714096:function:DivideNumbers",
"Next": "ChoiceState"
}
```

A Task is the basic unit of work in Step Functions. It represents a single unit of work performed by a state machine

You can define a Task by setting a state to “Type”: “Task” and providing the **Amazon Resource Name (ARN)** of the activity or Lambda function the Task should invoke.



## How to pass data to AWS Step Functions 

There are **two ways** to pass arguments through state machine. Via `InputPath` and `Parameters`. For differences please look [here](https://docs.aws.amazon.com/step-functions/latest/dg/input-output-inputpath-params.html).

If you do not have any static values that you want to pass to lambda, I would do the following. Passed all parameters to step function in **JSON** format. For example an **Input JSON** for state machine

```py
{
    "foo": 123,
    "bar": ["a", "b", "c"],
    "car": {
        "cdr": true
    }
    "TableName": "table_example"
}
```



 A path is a string beginning with $ within the Amazon States Language, and you can use it to identify components found inside the JSON text.

With reference paths will return:

```
$.foo => 123
$.bar =>["a","b","c"]
$.car.cdr= true
```

In step function you are passing entire **JSON** explicitly to lambda using `"InputPath": "$"`, except for a first step where it is passed implicitly. 

You can specify a path that will allow you access to subsets of the input when you’re determining the values for *InputPath, ResultPath*, and the *OutputPath*.

For more about `$` path syntax please look [here](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-paths.html). You also need to take care of task result, with one of [multiple approaches](https://docs.aws.amazon.com/step-functions/latest/dg/input-output-resultpath.html) using `ResultPath`. 

For most of cases the safest solution is to keep task result in special variable `"ResultPath": "$.taskresult"`

```py
{
  "Comment": "A Hello World example of the Amazon States Language using an AWS Lambda function",
  "StartAt": "HelloWorld",
  "States": {
    "HelloWorld": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:ap-southeast-2:XXXXXXX:function:fields_sync",
      "Next": "HelloWorld2"
    },
    "HelloWorld2": {
      "Type": "Task",
      "InputPath": "$",
      "ResultPath": "$.taskresult"
      "Resource": "arn:aws:lambda:ap-southeast-2:XXXXXXX:function:fields_sync_2",
      "End": true
    }
  }
}
```

Which in lambda became event variable and can be access as python dictionary

```py
def lambda_handler(event, context):
    table_example = event["TableName"]
    a = event["bar"][0]
    cdr_value = event["car"]["cdr"]
    # taskresult will not exist as event key 
    # only on lambda triggered by first state
    # in the rest of subsequent states
    # it will hold a task result of last executed state
    taskresult = event["taskresult"]
```

With this approach you can use multiple step functions and different lambdas and still keep both of them **clean and small** by **moving all the logic in lambdas**. Also it is easier to debug because all events variables will be the same in all lambdas, so via simple `print(event)` you can see all parameters needed for entire state machine and what possibly went wrong.

### Input and Output

In the **Amazon States Language**, these fields filter and control the flow of **JSON** from state to state:

- **InputPath** selects which parts of the **JSON** input to pass to the task of the `Task` state . 
- **OutputPath**  can filter the **JSON** output to further limit the information that's passed to the output.
- **ResultPath** then selects what combination of the state input and the task result to pass to the output. 
- **InputPath**, **Parameters**, **ResultSelector**, **ResultPath**, and **OutputPath** each manipulate **JSON** as it moves through each state in your workflow.

The following diagram shows how **JSON** information moves through a task state. 

![       Input and output processing     ](/assets/images/posts/2020-09-26-How-to-work-with-AWS-Step-Functions/input-output-processing.png)



# InputPath

It would be best if you used *InputPath* so you could **select a segment of the state input**

Let us take an example,where we assign a  Lambda execution,  within  Input  like below.

```
{
    "comment": "exam results.",
    "result": {
        "math": 80,
        "eng": 93
    },
    "extra": "foo",
    "lambda": {
        "who": "AWS Step Functions"
    }
}
```

 That input is bound to the symbol $ and passed on as the input to the first state in the state machine. By default, the output of each state would be bound to $ and becomes the input of the next state. The state machine definition is as below:

```
{
    "Comment": "A Hello World example of the Amazon States Language using an AWS Lambda function",
    "StartAt": "ExamResults",
    "States": {
        "ExamResults": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1: 3123456789012:function:HelloFunction",
	    	"InputPath": "$.lambda",
            "ResultPath": "$.result.total",
            "OutputPath": "$.result",
            "End": true
        }
    }
}
```

In Each state, we have *InputPath*, *ResultPath* and *OutputPath* attributes which filters the input and provide the final output. In the above scenario, “*ExamResults*” state is filtering “lambda” node, appending the result of a state execution to “*results*” node and final output is just “*result*” node rather than the whole JSON object.

In other words, given `$` as an input to the state machine

```
{
    "comment": "Exam Results.",
    "result": {
        "math": 0,
        "eng": 93
    },
    "extra": "foo",
    "Lambda": {
        "who": "AWS Step Functions"
    }
}
```

by using `$.lambda` in  `"InputPath" : "$.lambda"` will filter the Input **JSON** of state machine and pass only  `lambda`  node to the `"ExamResults"` state as an input

```
{
"who": "AWS Step Functions"
}
```

by using  `$.result.total`  in `"ResultPath" : "$.result.total"` takes whole `$` output
json, and adds the output of state task execution with "total" attribute inside "result" node.

```
{
    "comment": "exam results.",
    "result": {
        "math": 80,
        "eng": 93,
        "total": 173
    },
    "extra": "foo",
    "lambda": {
        "who": "AWS Step Functions"
    }
}
```

Hence, the final output  with `$.result` the `"OutputPath" : "$.result"` filters the `$` output json and pass only `"result"` as a final output of the state

```
{
    "math": 80,
    "eng": 93,
    "total": 173
}
```



# Parameters

Using the Parameters field will help you **create a collection of key-value pairs that are all passed as input**. These values can be selected from an input or context object with a path, or they can be static values that you’ve included within your state machine definition. **The key name must end in \*.$\*** for key-value pairs whose value was selected **using a path**.

Take a look at the following input example:

```
{
    "coment": "Example for Parameters.",
    "product": {
        "details": {
            "color": "blue",
            "size": "small",
            "material": "cotton"
        },
        "availability": "in stock",
        "sku": "2317",
        "cost": "$23"
    }
}
```

Specifying these parameters within your state machine definition will enable you to select some of the information.

```
"Parameters": {
    "comment": "Selecting what I care about.",
    "MyDetails": {
        "size.$": "$.product.details.size",
        "exists.$": "$.product.availability",
        "StaticValue": "foo"
    }
},
```

Considering the previous input along with the Parameters field, this is the JSON that’ll pass:

```
{
    "comment: "Selecting what I care about.",
    "MyDetails": {
        "size": "small",
        "exists": "in stock",
        "StaticValue": "foo"
    }
},
```

In addition to the provided input, you’ll easily access a special JSON object that’s known as “context object.” This object includes all information regarding your state machine execution