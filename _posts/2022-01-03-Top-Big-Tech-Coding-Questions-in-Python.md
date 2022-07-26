---
title: "Top Big Tech Coding Questions in Python"
excerpt: "Coding questions in Python for the  largest Big Tech companies "

header:
  image: "../assets/images/posts/2022-01-03-Top-Big-Tech-Coding-Questions-in-Python/joshua-reddekopp.jpg"
  teaser: "../assets/images/posts/2022-01-03-Top-Big-Tech-Coding-Questions-in-Python/joshua-reddekopp.jpg"
  caption: "“First, solve the problem. Then, write the code. – John Johnson"
---

Hello everyone, today I will discuss some interesting Easy Python exercises.  You can solve them in less than 3 minutes for each of them. 

I have collected some basic questions in Python for the  largest Big Tech companies. 

- Microsoft
- Apple
- Amazon

Usually FAANG is an acronym used to describe some of the most prominent companies in the tech sector.  FANG is a group of high performing technology stocks that includes Facebook, Amazon, Netflix, and Google (Alphabet). Investors then added Apple into the list to form the acronym FAANG.

<img src="../assets/images/posts/2022-01-03-Top-Big-Tech-Coding-Questions-in-Python/FANG.png" style="zoom:50%;" />



Big Tech, also known as the Tech Giants, Big Four, or Big Five, is a name given to the presently four or five largest, most dominant, and most prestigious companies in the information technology industry of the United States.

![](../assets/images/posts/2022-01-03-Top-Big-Tech-Coding-Questions-in-Python/FANG2.png)



# Coding Questions

The questions  contains an example of the first rows of the datasets to deal with.

## **Microsoft**

Find the total number of records that belong to each variety in the dataset. Output the variety along with the corresponding number of records. Order records by the variety in ascending order.

### iris

| sepal_length | sepal_width | petal_length | petal_width | variety |
| ------------ | ----------- | ------------ | ----------- | ------- |
| 5.1          | 3.5         | 1.4          | 0.2         | Setosa  |
| 4.9          | 3           | 1.4          | 0.2         | Setosa  |
| 4.7          | 3.2         | 1.3          | 0.2         | Setosa  |
| 4.6          | 3.1         | 1.5          | 0.2         | Setosa  |
| 5            | 3.6         | 1.4          | 0.2         | Setosa  |

```python
# Import your libraries
import pandas as pd

# Start writing code
iris.groupby(['variety']).size().to_frame('n_variety').reset_index().sort_values('variety')
```



Write a query that returns the number of unique users per client per month.

### fact_events

| id   | time_id             | user_id    | customer_id | client_id | event_type          | event_id |
| ---- | ------------------- | ---------- | ----------- | --------- | ------------------- | -------- |
| 1    | 2020-02-28 00:00:00 | 3668-QPYBK | Sendit      | desktop   | message sent        | 3        |
| 2    | 2020-02-28 00:00:00 | 7892-POOKP | Connectix   | mobile    | file received       | 2        |
| 3    | 2020-04-03 00:00:00 | 9763-GRSKD | Zoomit      | desktop   | video call received | 7        |
| 4    | 2020-04-02 00:00:00 | 9763-GRSKD | Connectix   | desktop   | video call received | 7        |
| 5    | 2020-02-06 00:00:00 | 9237-HQITU | Sendit      | desktop   | video call received | 7        |



```python
# Import your libraries
import pandas as pd

# Start writing code
df=fact_events
df['month'] = pd.to_datetime(df['time_id']).dt.month
df.groupby(['month','client_id'])['user_id'].nunique().reset_index()
```



We have a table with employees and their salaries, however, some of the records are old and contain outdated salary information. Find the current salary of each employee assuming that salaries increase each year. Output their id, first name, last name, department ID, and current salary. Order your list by employee ID in ascending order.

### ms_employee_salary

| id   | first_name | last_name | salary | department_id |
| ---- | ---------- | --------- | ------ | ------------- |
| 1    | Todd       | Wilson    | 110000 | 1006          |
| 1    | Todd       | Wilson    | 106119 | 1006          |
| 2    | Justin     | Simon     | 128922 | 1005          |
| 2    | Justin     | Simon     | 130000 | 1005          |
| 3    | Kelly      | Rosario   | 42689  | 1002          |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
df = ms_employee_salary
df.groupby('id').max().reset_index()
```



Given a list of projects and employees mapped to each project, calculate by the amount of project budget allocated to each employee . The output should include the **project title** and the **project budget** per employee **rounded to the closest integer**. Order your list by projects with the highest budget per employee first.

Preview of firsts rows 

**ms_projects**

| id   | title    | budget |
| ---- | -------- | ------ |
| 1    | Project1 | 29498  |
| 2    | Project2 | 32487  |
| 3    | Project3 | 43909  |
| 4    | Project4 | 15776  |
| 5    | Project5 | 36268  |

**ms_emp_projects**

| emp_id | project_id |
| ------ | ---------- |
| 10592  | 1          |
| 10593  | 2          |
| 10594  | 3          |
| 10595  | 4          |
| 10596  | 5          |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
df_grp = ms_emp_projects.groupby('project_id')['emp_id'].count().reset_index()

df_merge = pd.merge(ms_projects , df_grp , left_on = 'id'
                    , right_on = 'project_id' 
                    , how = 'inner')
df_merge['budget_emp_ratio'] = df_merge['budget'] / df_merge['emp_id']
df_merge[['title','budget_emp_ratio']]
```

Given a list of projects and employees mapped to each project, calculate by the amount of project budget allocated to each employee . The output should include the project title and the project budget per employee rounded to the closest integer. Order your list by projects with the highest budget per employee first.

### ms_projects

| id   | title    | budget |
| ---- | -------- | ------ |
| 1    | Project1 | 29498  |
| 2    | Project2 | 32487  |
| 3    | Project3 | 43909  |
| 4    | Project4 | 15776  |
| 5    | Project5 | 36268  |



### ms_emp_projects

| emp_id | project_id |
| ------ | ---------- |
| 10592  | 1          |
| 10593  | 2          |
| 10594  | 3          |
| 10595  | 4          |
| 10596  | 5          |

**Solution**

```python
import pandas as pd
import numpy as np

df=pd.merge(ms_projects, ms_emp_projects, how = 'inner',left_on = ['id'], right_on=['project_id'])
df1=df.groupby(['title','budget'])['emp_id'].size().reset_index()
df1['budget_emp_ratio'] = (df1['budget']/df1['emp_id']).round(0)
df2=df1.sort_values(by='budget_emp_ratio',ascending=False)
result = df2[["title","budget_emp_ratio"]]

```





Write a query that returns the company (customer id column) with highest number of users that use desktop only.



### fact_events

| id   | time_id             | user_id    | customer_id | client_id | event_type          | event_id |
| ---- | ------------------- | ---------- | ----------- | --------- | ------------------- | -------- |
| 1    | 2020-02-28 00:00:00 | 3668-QPYBK | Sendit      | desktop   | message sent        | 3        |
| 2    | 2020-02-28 00:00:00 | 7892-POOKP | Connectix   | mobile    | file received       | 2        |
| 3    | 2020-04-03 00:00:00 | 9763-GRSKD | Zoomit      | desktop   | video call received | 7        |
| 4    | 2020-04-02 00:00:00 | 9763-GRSKD | Connectix   | desktop   | video call received | 7        |
| 5    | 2020-02-06 00:00:00 | 9237-HQITU | Sendit      | desktop   | video call received | 7        |

**Solution**

```python
import pandas as pd

one_client_users = fact_events.groupby('user_id')['client_id'].nunique().reset_index()
one_client_users = one_client_users[one_client_users['client_id']==1]
desktop_users = fact_events[fact_events['client_id']=='desktop']

result = fact_events[(fact_events['user_id'].isin(desktop_users['user_id'])) & (fact_events['user_id'].isin(one_client_users['user_id']))].groupby('customer_id')['user_id'].nunique().to_frame('num_users').reset_index()
result = result[result['num_users']==result['num_users'].max()][['customer_id']]

```

Write a query that returns the company (customer id column) with highest number of users that use desktop only.



| id   | time_id             | user_id    | customer_id | client_id | event_type          | event_id |
| ---- | ------------------- | ---------- | ----------- | --------- | ------------------- | -------- |
| 1    | 2020-02-28 00:00:00 | 3668-QPYBK | Sendit      | desktop   | message sent        | 3        |
| 2    | 2020-02-28 00:00:00 | 7892-POOKP | Connectix   | mobile    | file received       | 2        |
| 3    | 2020-04-03 00:00:00 | 9763-GRSKD | Zoomit      | desktop   | video call received | 7        |
| 4    | 2020-04-02 00:00:00 | 9763-GRSKD | Connectix   | desktop   | video call received | 7        |
| 5    | 2020-02-06 00:00:00 | 9237-HQITU | Sendit      |           |                     |          |

**Solution**

```python
import pandas as pd

one_client_users = fact_events.groupby('user_id')['client_id'].nunique().reset_index()
one_client_users = one_client_users[one_client_users['client_id']==1]
desktop_users = fact_events[fact_events['client_id']=='desktop']

result = fact_events[(fact_events['user_id'].isin(desktop_users['user_id'])) & (fact_events['user_id'].isin(one_client_users['user_id']))].groupby('customer_id')['user_id'].nunique().to_frame('num_users').reset_index()
result = result[result['num_users']==result['num_users'].max()][['customer_id']]

```



Write a query that returns a number of users who are exclusive to only one client. Output the client_id and number of exclusive users.

### fact_events

| id   | time_id             | user_id    | customer_id | client_id | event_type          | event_id |
| ---- | ------------------- | ---------- | ----------- | --------- | ------------------- | -------- |
| 1    | 2020-02-28 00:00:00 | 3668-QPYBK | Sendit      | desktop   | message sent        | 3        |
| 2    | 2020-02-28 00:00:00 | 7892-POOKP | Connectix   | mobile    | file received       | 2        |
| 3    | 2020-04-03 00:00:00 | 9763-GRSKD | Zoomit      | desktop   | video call received | 7        |
| 4    | 2020-04-02 00:00:00 | 9763-GRSKD | Connectix   | desktop   | video call received | 7        |
| 5    | 2020-02-06 00:00:00 | 9237-HQITU | Sendit      | desktop   | video call received | 7        |

**Solution**

```python
import pandas as pd

fact_events['n_clients'] = fact_events.groupby('user_id')['client_id'].transform('nunique')
result = fact_events[fact_events['n_clients']==1].groupby(['client_id'])['user_id'].nunique().reset_index()

```



Write a query that returns a list of the bottom 2 companies by mobile usage. Company is defined in the customer_id column. Mobile usage is defined as the number of events registered on a client_id == 'mobile'. Order the result by the number of events ascending. In the case where there are multiple companies tied for the bottom ranks (rank 1 or 2), return all the companies. Output the customer_id and number of events.

### fact_events

| id   | time_id             | user_id    | customer_id | client_id | event_type          | event_id |
| ---- | ------------------- | ---------- | ----------- | --------- | ------------------- | -------- |
| 1    | 2020-02-28 00:00:00 | 3668-QPYBK | Sendit      | desktop   | message sent        | 3        |
| 2    | 2020-02-28 00:00:00 | 7892-POOKP | Connectix   | mobile    | file received       | 2        |
| 3    | 2020-04-03 00:00:00 | 9763-GRSKD | Zoomit      | desktop   | video call received | 7        |
| 4    | 2020-04-02 00:00:00 | 9763-GRSKD | Connectix   | desktop   | video call received | 7        |
| 5    | 2020-02-06 00:00:00 | 9237-HQITU | Sendit      | desktop   | video call received | 7        |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
fact_events.head()

agg_df = fact_events\
    .query('client_id == "mobile"')\
    .groupby('customer_id')['event_id']\
    .count()\
    .reset_index()\
    .rename(columns={'event_id': 'n_events'})
    
agg_df[agg_df['n_events'].isin(agg_df['n_events'].drop_duplicates().nsmallest(2))]\
    .sort_values('n_events', ascending=True)
```



Find the number of employees who received the bonus and who didn't. Output an indication of whether the bonus was received or not along with the corresponding number of employees. ex: if the bonus was received: 1, if not: 0.

### employee

| id   | first_name | last_name | age  | sex  | employee_title | department | salary | target | bonus | email                | city       | address              | manager_id |
| ---- | ---------- | --------- | ---- | ---- | -------------- | ---------- | ------ | ------ | ----- | -------------------- | ---------- | -------------------- | ---------- |
| 5    | Max        | George    | 26   | M    | Sales          | Sales      | 1300   | 200    | 150   | Max@company.com      | California | 2638 Richards Avenue | 1          |
| 13   | Katty      | Bond      | 56   | F    | Manager        | Management | 150000 | 0      | 300   | Katty@company.com    | Arizona    |                      | 1          |
| 11   | Richerd    | Gear      | 57   | M    | Manager        | Management | 250000 | 0      | 300   | Richerd@company.com  | Alabama    |                      | 1          |
| 10   | Jennifer   | Dion      | 34   | F    | Sales          | Sales      | 1000   | 200    | 150   | Jennifer@company.com | Alabama    |                      | 13         |
| 19   | George     | Joe       | 50   | M    | Manager        | Management | 100000 | 0      | 300   | George@company.com   | Florida    | 1003 Wyatt Street    | 1          |



### bonus

| worker_ref_id | bonus_amount | bonus_date          |
| ------------- | ------------ | ------------------- |
| 1             | 5000         | 2020-02-16 00:00:00 |
| 2             | 3000         | 2011-06-16 00:00:00 |
| 3             | 4000         | 2020-02-16 00:00:00 |
| 1             | 4500         | 2020-02-16 00:00:00 |
| 2             | 3500         | 2011-06-16 00:00:00 |

**Solution**

```python
import pandas as pd
import numpy as np

merged_df = pd.merge(employee,bonus,left_on='id',right_on='worker_ref_id',how='left')
merged_df['has_bonus'] = (merged_df.bonus_date.notnull()).astype(int)
result = merged_df.groupby(['has_bonus'])['id'].nunique()

```

Find the number of transactions that occurred for each product. Output the product name along with the corresponding number of transactions and order records by the product id in ascending order. You can ignore products without transactions.

### excel_sql_inventory_data

| product_id | product_name | product_type | unit | price_unit | wholesale | current_inventory |
| ---------- | ------------ | ------------ | ---- | ---------- | --------- | ----------------- |
| 1          | strawberry   | produce      | lb   | 3.28       | 1.77      | 13                |
| 2          | apple_fuji   | produce      | lb   | 1.44       | 0.43      | 2                 |
| 3          | orange       | produce      | lb   | 1.02       | 0.37      | 2                 |
| 4          | clementines  | produce      | lb   | 1.19       | 0.44      | 44                |
| 5          | blood_orange | produce      | lb   | 3.86       | 1.66      | 19                |



### excel_sql_transaction_data

| transaction_id | time                | product_id |
| -------------- | ------------------- | ---------- |
| 1              | 2016-01-08 17:46:17 | 3          |
| 4              | 2016-01-06 17:57:42 | 4          |
| 8              | 2016-01-07 09:35:40 | 8          |
| 9              | 2016-01-03 09:06:20 | 12         |
| 9              | 2016-01-03 09:06:20 | 14         |

**Solution**

```python
import pandas as pd
import numpy as np

transaction = excel_sql_transaction_data[excel_sql_transaction_data['product_id'].notnull()]

merged = pd.merge(excel_sql_inventory_data,transaction, on = 'product_id', how = 'inner')

merged['count'] = merged.groupby(['product_name','product_id'])['transaction_id'].transform('count')

merged = merged.drop_duplicates(subset = 'product_id')

result = merged.sort_values('product_id')[['product_name', 'count']]

```

Select the most popular client_id based on a count of the number of users who have at least 50% of their events from the following list: 'video call received', 'video call sent', 'voice call received', 'voice call sent'.

### fact_events

| id   | time_id             | user_id    | customer_id | client_id | event_type          | event_id |
| ---- | ------------------- | ---------- | ----------- | --------- | ------------------- | -------- |
| 1    | 2020-02-28 00:00:00 | 3668-QPYBK | Sendit      | desktop   | message sent        | 3        |
| 2    | 2020-02-28 00:00:00 | 7892-POOKP | Connectix   | mobile    | file received       | 2        |
| 3    | 2020-04-03 00:00:00 | 9763-GRSKD | Zoomit      | desktop   | video call received | 7        |
| 4    | 2020-04-02 00:00:00 | 9763-GRSKD | Connectix   | desktop   | video call received | 7        |
| 5    | 2020-02-06 00:00:00 | 9237-HQITU | Sendit      | desktop   | video call received | 7        |

**Solution**

```python
import pandas as pd

events_list = ['video call received', 'video call sent', 'voice call received', 'voice call sent']
fact_events['event_check'] = fact_events['event_type'].apply(lambda x: 1 if x in events_list else 0)
fact_events['event_check_mean'] = fact_events.groupby('user_id')['event_check'].transform('mean')
result = fact_events[fact_events['event_check_mean']>=0.5].groupby('client_id')['id'].count().reset_index()
result['ranking'] = result['id'].rank(ascending=False)
result = result[result.ranking == 1][['client_id']]

```



Calculate the share of new and existing users for each month in the table. Output the month, share of new users, and share of existing users as a ratio. New users are defined as users who started using services in the current month (there is no usage history in previous months). Existing users are users who used services in current month, but they also used services in any previous month. Assume that the dates are all from the year 2020.



### fact_events

| id   | time_id             | user_id    | customer_id | client_id | event_type          | event_id |
| ---- | ------------------- | ---------- | ----------- | --------- | ------------------- | -------- |
| 1    | 2020-02-28 00:00:00 | 3668-QPYBK | Sendit      | desktop   | message sent        | 3        |
| 2    | 2020-02-28 00:00:00 | 7892-POOKP | Connectix   | mobile    | file received       | 2        |
| 3    | 2020-04-03 00:00:00 | 9763-GRSKD | Zoomit      | desktop   | video call received | 7        |
| 4    | 2020-04-02 00:00:00 | 9763-GRSKD | Connectix   | desktop   | video call received | 7        |
| 5    | 2020-02-06 00:00:00 | 9237-HQITU | Sendit      | desktop   | video call received | 7        |

**Solution**

```python
import pandas as pd
import datetime as dt 

result = fact_events.groupby(pd.to_datetime(fact_events['time_id']).dt.month)['user_id'].nunique().to_frame('n_users')
df1 = fact_events.groupby('user_id')['time_id'].min().to_frame('month_min').reset_index()
df1['month_min'] = pd.to_datetime(df1['month_min']).dt.month
df1 = df1.groupby('month_min')['user_id'].nunique().to_frame('n_new_users')
result = result.join(df1)
result['share_of_new_users'] = result['n_new_users'] /  result['n_users']
result['share_of_old_users'] = 1 - result['share_of_new_users']
result = result[['share_of_new_users', 'share_of_old_users']].reset_index()

```

Find the total number of downloads for paying and non-paying users by date. Include only records where non-paying customers have more downloads than paying customers. The output should be sorted by earliest date first and contain 3 columns date, non-paying downloads, paying downloads.

### ms_user_dimension

| user_id | acc_id |
| ------- | ------ |
| 1       | 716    |
| 2       | 749    |
| 3       | 713    |
| 4       | 744    |
| 5       | 726    |

### ms_acc_dimension

| acc_id | paying_customer |
| ------ | --------------- |
| 700    | no              |
| 701    | no              |
| 702    | no              |
| 703    | no              |
| 704    | no              |

### ms_download_facts

View the output in a separate browser tab

| date                | user_id | downloads |
| ------------------- | ------- | --------- |
| 2020-08-24 00:00:00 | 1       | 6         |
| 2020-08-22 00:00:00 | 2       | 6         |
| 2020-08-18 00:00:00 | 3       | 2         |
| 2020-08-24 00:00:00 | 4       | 4         |
| 2020-08-19 00:00:00 | 5       | 7         |

**Solution**

```python
import pandas as pd
import numpy as np

df=pd.merge(ms_user_dimension, ms_acc_dimension, how = 'inner',left_on = ['acc_id'], right_on=['acc_id'])
df1= pd.merge(df, ms_download_facts, how = 'inner',left_on = ['user_id'], right_on=['user_id'])
x=df1.pivot_table(index=['date'],columns=['paying_customer'],values=['downloads'],aggfunc='sum')
df2=pd.DataFrame(x.to_records())
df2.columns = df2.columns.str.replace("[()]","").str.replace("[' ']","").str.replace("[,]","").str.replace("downloads","")
df3=df2.fillna(0)
df3['diff'] = df3['no']-df3['yes']
df4 = df3[df3["diff"] > 0]
result = df4[["date","no","yes"]].sort_values("date")

```

## **Interview Questions in Apple**



Count the number of user events performed by MacBookPro users. Output the result along with the event name. Sort the result based on the event count in the descending order.

### playbook_events

| user_id | occurred_at         | event_type  | event_name | location      | device              |
| ------- | ------------------- | ----------- | ---------- | ------------- | ------------------- |
| 6991    | 2014-06-09 18:26:54 | engagement  | home_page  | United States | iphone 5            |
| 18851   | 2014-08-29 13:18:38 | signup_flow | enter_info | Russia        | asus chromebook     |
| 14998   | 2014-07-01 12:47:56 | engagement  | login      | France        | hp pavilion desktop |
| 8186    | 2014-05-23 10:44:16 | engagement  | home_page  | Italy         | macbook pro         |
| 9626    | 2014-07-31 17:15:14 | engagement  | login      | Russia        | nexus 7             |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
df = playbook_events
df.loc[df.device == 'macbook pro', 'event_name'].value_counts().reset_index()
```

Find customers who appear in the orders table more than three times.

### orders

| id   | cust_id | order_date          | order_details | total_order_cost |
| ---- | ------- | ------------------- | ------------- | ---------------- |
| 1    | 3       | 2019-03-04 00:00:00 | Coat          | 100              |
| 2    | 3       | 2019-03-01 00:00:00 | Shoes         | 80               |
| 3    | 3       | 2019-03-07 00:00:00 | Skirt         | 30               |
| 4    | 7       | 2019-02-01 00:00:00 | Coat          | 25               |
| 5    | 7       | 2019-03-10 00:00:00 | Shoes         | 80               |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
grp = orders.groupby('cust_id').size().reset_index(name='ct')
grp[grp.ct > 3]['cust_id'].values

```

Find customers who have never made an order. Output the first name of the customer.

### customers

| id   | first_name | last_name | city          | address | phone_number |
| ---- | ---------- | --------- | ------------- | ------- | ------------ |
| 8    | John       | Joseph    | San Francisco |         | 928-386-8164 |
| 7    | Jill       | Michael   | Austin        |         | 813-297-0692 |
| 4    | William    | Daniel    | Denver        |         | 813-368-1200 |
| 5    | Henry      | Jackson   | Miami         |         | 808-601-7513 |
| 13   | Emma       | Isaac     | Miami         |         | 808-690-5201 |



### orders

| id   | cust_id | order_date          | order_details | total_order_cost |
| ---- | ------- | ------------------- | ------------- | ---------------- |
| 1    | 3       | 2019-03-04 00:00:00 | Coat          | 100              |
| 2    | 3       | 2019-03-01 00:00:00 | Shoes         | 80               |
| 3    | 3       | 2019-03-07 00:00:00 | Skirt         | 30               |
| 4    | 7       | 2019-02-01 00:00:00 | Coat          | 25               |
| 5    | 7       | 2019-03-10 00:00:00 | Shoes         | 80               |

**Solution**

```python
import pandas as pd
import numpy as np

merge = pd.merge(customers,orders, left_on = 'id', right_on = 'cust_id', how = 'left')
result = merge[merge['cust_id'].isnull()][['first_name']]


```



For each platform (e.g. Windows, iPhone, iPad etc.), calculate the number of users. Consider unique users and not individual sessions. Output the name of the platform with the corresponding number of users.



### user_sessions

| session_id | user_id | session_starttime   | session_endtime     | platform |
| ---------- | ------- | ------------------- | ------------------- | -------- |
| 1          | U1      | 2020-01-01 12:14:28 | 2020-01-01 12:16:08 | Windows  |
| 2          | U1      | 2020-01-01 18:23:50 | 2020-01-01 18:24:00 | Windows  |
| 3          | U1      | 2020-01-01 08:15:00 | 2020-01-01 08:20:00 | IPhone   |
| 4          | U2      | 2020-01-01 10:53:10 | 2020-01-01 10:53:30 | IPhone   |
| 5          | U2      | 2020-01-01 18:25:14 | 2020-01-01 18:27:53 | IPhone   |

**Solution**

```python
grouped_df = user_sessions.groupby("platform")
grouped_df = grouped_df['user_id'].nunique()
result = grouped_df.reset_index()
```



Find the number of male and female employees per department and also their corresponding total salaries. Output department names along with the corresponding number of female employees, the total salary of female employees, the number of male employees, and the total salary of male employees.



### employee

| id   | first_name | last_name | age  | sex  | employee_title | department | salary | target | bonus | email                | city       | address              | manager_id |
| ---- | ---------- | --------- | ---- | ---- | -------------- | ---------- | ------ | ------ | ----- | -------------------- | ---------- | -------------------- | ---------- |
| 5    | Max        | George    | 26   | M    | Sales          | Sales      | 1300   | 200    | 150   | Max@company.com      | California | 2638 Richards Avenue | 1          |
| 13   | Katty      | Bond      | 56   | F    | Manager        | Management | 150000 | 0      | 300   | Katty@company.com    | Arizona    |                      | 1          |
| 11   | Richerd    | Gear      | 57   | M    | Manager        | Management | 250000 | 0      | 300   | Richerd@company.com  | Alabama    |                      | 1          |
| 10   | Jennifer   | Dion      | 34   | F    | Sales          | Sales      | 1000   | 200    | 150   | Jennifer@company.com | Alabama    |                      | 13         |
| 19   | George     | Joe       | 50   | M    | Manager        | Management | 100000 | 0      | 300   |                      |            |                      |            |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
df = employee
sex_df = df.groupby(['department', 'sex'])['id'].nunique().reset_index()
salary_df = df.groupby(['department', 'sex'])['salary'].sum().reset_index()
pd.merge(sex_df, salary_df, on = ['department', 'sex'])
```



## **Interview Questions in Amazon**

Find the details of each customer regardless of whether the customer made an order. Output the customer's **first name**, **last name**, and the **city** along with the **order details.** You may have duplicate rows in your results due to a customer ordering several of the same items. Sort records based on the customer's **first name** and the order details in **ascending order**.

**Sample of header:**

**customers**

| id   | first_name | last_name | city          | address | phone_number |
| ---- | ---------- | --------- | ------------- | ------- | ------------ |
| 8    | John       | Joseph    | San Francisco |         | 928-386-8164 |
| 7    | Jill       | Michael   | Austin        |         | 813-297-0692 |
| 4    | William    | Daniel    | Denver        |         | 813-368-1200 |
| 5    | Henry      | Jackson   | Miami         |         | 808-601-7513 |
| 13   | Emma       | Isaac     | Miami         |         | 808-690-5201 |

​	**orders**

| id   | cust_id | order_date          | order_details | total_order_cost |
| ---- | ------- | ------------------- | ------------- | ---------------- |
| 1    | 3       | 2019-03-04 00:00:00 | Coat          | 100              |
| 2    | 3       | 2019-03-01 00:00:00 | Shoes         | 80               |
| 3    | 3       | 2019-03-07 00:00:00 | Skirt         | 30               |
| 4    | 7       | 2019-02-01 00:00:00 | Coat          | 25               |
| 5    | 7       | 2019-03-10 00:00:00 | Shoes         | 80               |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
merged = pd.merge(customers, orders, left_on = 'id', right_on = 'cust_id', how = 'left')

merged[['first_name', 'last_name', 'city', 'order_details']].sort_values(['first_name', 'order_details'])
```

 

Find the second highest salary without using ORDER BY.

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 |                     |            |

```python
import pandas as pd
import numpy as np

distinct_salary = worker[['salary']].drop_duplicates()
top2_highest = distinct_salary.nlargest(2,'salary')[['salary']]
result = top2_highest.nsmallest(1,'salary')[['salary']]

```

Find the 5 highest salaries. Order records based on salary in descending order.

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 | 2014-06-11          |            |

**Solution**

```
import pandas as pd
worker.salary.drop_duplicates().nlargest(5)
```

or 

```
import pandas as pd
import numpy as np
result = worker.drop_duplicates(['salary'])[['salary']].sort_values(['salary'], ascending=False).head(5)
```

Find the five highest salaries.

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 |                     |            |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
df=worker
df['salary'].drop_duplicates().nlargest(5)
```

Find the total cost of each customer's orders. Output customer's id, first name, and the total order cost. Order records by customer's first name alphabetically.

### customers

| id   | first_name | last_name | city          | address | phone_number |
| ---- | ---------- | --------- | ------------- | ------- | ------------ |
| 8    | John       | Joseph    | San Francisco |         | 928-386-8164 |
| 7    | Jill       | Michael   | Austin        |         | 813-297-0692 |
| 4    | William    | Daniel    | Denver        |         | 813-368-1200 |
| 5    | Henry      | Jackson   | Miami         |         | 808-601-7513 |
| 13   | Emma       | Isaac     | Miami         |         |              |

### orders

| id   | cust_id | order_date          | order_details | total_order_cost |
| ---- | ------- | ------------------- | ------------- | ---------------- |
| 1    | 3       | 2019-03-04 00:00:00 | Coat          | 100              |
| 2    | 3       | 2019-03-01 00:00:00 | Shoes         | 80               |
| 3    | 3       | 2019-03-07 00:00:00 | Skirt         | 30               |
| 4    | 7       | 2019-02-01 00:00:00 | Coat          | 25               |
| 5    | 7       | 2019-03-10 00:00:00 | Shoes         | 80               |

**Solution**

```python
import pandas as pd
import numpy as np

merge = pd.merge(customers, orders, left_on="id", right_on="cust_id")
merge = (
    merge.groupby(["cust_id", "first_name"])["total_order_cost"].sum().reset_index()
)
result = merge.sort_values(by="first_name", ascending=True)

```

Find the top 5 highest paid workers. Output all information about workers whose salaries are among top 5 paid ones.



### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 |                     |            |

**Solution**

```python
import pandas as pd
import numpy as np
result = worker[worker['salary'].rank(method='min', ascending=False) <= 5] 

```

Find all workers whose first name contains the letter 'a'. Output all columns for such workers.

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 |                     |            |

**Solution**

```python
import pandas as pd
import numpy as np

result = worker[worker['first_name'].str.contains("a")]

```

Find all workers who joined on February 2014.

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 |                     |            |

**Solution**

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

worker['joining_date'] = pd.to_datetime(worker['joining_date'], format="%Y-%m-%d")
result = worker[(worker['joining_date']>pd.Timestamp(2014,2,1)) & (worker['joining_date']<pd.Timestamp(2014,2,28))]

```

or 

```python
# Import your libraries
import pandas as pd

# Start writing code
worker.query("joining_date.dt.month == 2 & joining_date.dt.year == 2014")
```

Write a query that will calculate the number of shipments per month. The unique key for one shipment is a combination of shipment_id and sub_id. Output the year_month in format YYYY-MM and the number of shipments in that month.

### amazon_shipment

| shipment_id | sub_id | weight | shipment_date       | add  |
| ----------- | ------ | ------ | ------------------- | ---- |
| 101         | 1      | 10     | 2021-08-30 00:00:00 |      |
| 101         | 2      | 20     | 2021-09-01 00:00:00 |      |
| 101         | 3      | 10     | 2021-09-05 00:00:00 |      |
| 102         | 1      | 50     | 2021-09-02 00:00:00 |      |
| 103         | 1      | 25     | 2021-09-01 00:00:00 |      |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
df = amazon_shipment
df['year_month'] = pd.to_datetime(df['shipment_date']).dt.to_period('M')
df.groupby('year_month').size().to_frame('num_of_shipment').reset_index()
```

Find all workers who are also managers.
 Output the first name along with the corresponding title.

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 | 2014-06-11 09:00:00 | Admin      |

### title

| worker_ref_id | worker_title  | affected_from       |
| ------------- | ------------- | ------------------- |
| 1             | Manager       | 2016-02-20 00:00:00 |
| 2             | Executive     | 2016-06-11 00:00:00 |
| 8             | Executive     | 2016-06-11 00:00:00 |
| 5             | Manager       | 2016-06-11 00:00:00 |
| 4             | Asst. Manager | 2016-06-11 00:00:00 |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
worker.join(title[title.worker_title=='Manager'][['worker_ref_id', 'worker_title']].set_index('worker_ref_id'),
    how='inner', on='worker_id')[['first_name', 'worker_title']]

```

Find the rows where (worker title, affected from date) combination occurred more than once Output the worker title, affected_from date, and the number of times the occurrence happened.

### title

| worker_ref_id | worker_title  | affected_from       |
| ------------- | ------------- | ------------------- |
| 1             | Manager       | 2016-02-20 00:00:00 |
| 2             | Executive     | 2016-06-11 00:00:00 |
| 8             | Executive     | 2016-06-11 00:00:00 |
| 5             | Manager       | 2016-06-11 00:00:00 |
| 4             | Asst. Manager | 2016-06-11 00:00:00 |

**Solution**

```python
import pandas as pd
import numpy as np
import datetime

title['new_affected_from'] = title['affected_from']
title['new_affected_from'] =  pd.to_datetime(title['affected_from'], format='%Y-%m-%d' ).dt.strftime('%Y-%m-%d')

group = title.groupby(['worker_title','new_affected_from']).size().to_frame('n_affected').reset_index()
result = group[group['n_affected'] > 1]
```

Find the total salary of each department. Output the salary along with the corresponding department.

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 | 2014-06-11 09:00:00 | Admin      |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
worker.head()

worker.groupby("department").salary.sum().to_frame("total_salary").reset_index()
```

Find order details made by Jill and Eva. Consider the Jill and Eva as first names of customers. Output the order date, details and cost along with the first name. Order records based on the customer id in ascending order.

### customers

| id   | first_name | last_name | city          | address | phone_number |
| ---- | ---------- | --------- | ------------- | ------- | ------------ |
| 8    | John       | Joseph    | San Francisco |         | 928-386-8164 |
| 7    | Jill       | Michael   | Austin        |         | 813-297-0692 |
| 4    | William    | Daniel    | Denver        |         | 813-368-1200 |
| 5    | Henry      | Jackson   | Miami         |         | 808-601-7513 |
| 13   | Emma       | Isaac     | Miami         |         | 808-690-5201 |

### orders

| id   | cust_id | order_date          | order_details | total_order_cost |
| ---- | ------- | ------------------- | ------------- | ---------------- |
| 1    | 3       | 2019-03-04 00:00:00 | Coat          | 100              |
| 2    | 3       | 2019-03-01 00:00:00 | Shoes         | 80               |
| 3    | 3       | 2019-03-07 00:00:00 | Skirt         | 30               |
| 4    | 7       | 2019-02-01 00:00:00 | Coat          | 25               |
| 5    | 7       | 2019-03-10 00:00:00 | Shoes         | 80               |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
customers.head()

df = pd.merge(customers, orders, left_on="id", right_on="cust_id")
custs = ["Jill", "Eva"]
df1 = df[df["first_name"].isin(custs)].sort_values("cust_id")[["first_name", "order_date", "order_details", "total_order_cost"]]
```

Find all workers that work in the Admin department

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 | 2014-06-11 0        |            |

**Solution**

```python
import pandas as pd
import numpy as np

result = worker[worker['department']=='Admin']
```

Find the three lowest salaries. Order records based on the salary in ascending order.

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 | 2014-06-11 09:00:00 | Admin      |

**Solution**

```python
import pandas as pd
import numpy as np

salaries = worker[['salary']].drop_duplicates()
result = salaries.nsmallest(3,'salary')[['salary']]
```

For each video game player, find the latest date when they logged in.

### players_logins

| player_id | login_date          |
| --------- | ------------------- |
| 101       | 2021-12-14 00:00:00 |
| 101       | 2021-12-18 00:00:00 |
| 101       | 2021-12-15 00:00:00 |
| 101       | 2021-12-19 00:00:00 |
| 102       | 2021-12-31 00:00:00 |

```
result = players_logins.groupby(['player_id'])['login_date'].max().reset_index()

```

Write a query that'll identify returning active users. A returning active user is a user that has made a second purchase within 7 days of any other of their purchases. Output a list of user_ids of these returning active users.

### amazon_transactions

| id   | user_id | item    | created_at          | revenue |
| ---- | ------- | ------- | ------------------- | ------- |
| 1    | 109     | milk    | 2020-03-03 00:00:00 | 123     |
| 2    | 139     | biscuit | 2020-03-18 00:00:00 | 421     |
| 3    | 120     | milk    | 2020-03-18 00:00:00 | 176     |
| 4    | 108     | banana  | 2020-03-18 00:00:00 | 862     |
| 5    | 130     | milk    | 2020-03-28 00:00:00 | 333     |

**Solution**

```python
# Import your libraries
import pandas as pd
import numpy as np
import datetime

# Start writing code

df = amazon_transactions.sort_values(['user_id','created_at'])
df['diff'] = df.groupby('user_id')['created_at'].diff()
df[df['diff'] <= pd.Timedelta(days=7)]['user_id'].unique()
```

or

```python
import pandas as pd
import numpy as np
from datetime import datetime

amazon_transactions["created_at"] = pd.to_datetime(amazon_transactions["created_at"]).dt.strftime('%m-%d-%Y')
df = amazon_transactions.sort_values(by=['user_id', 'created_at'], ascending=[True, True])
df['prev_value'] = df.groupby('user_id')['created_at'].shift()
df['days'] = (pd.to_datetime(df['created_at']) - pd.to_datetime(df['prev_value'])).dt.days
result = df[df['days'] <= 7]['user_id'].unique()

```

Find the number of customers without an order.

### orders

View the output in a separate browser tab

| id   | cust_id | order_date          | order_details | total_order_cost |
| ---- | ------- | ------------------- | ------------- | ---------------- |
| 1    | 3       | 2019-03-04 00:00:00 | Coat          | 100              |
| 2    | 3       | 2019-03-01 00:00:00 | Shoes         | 80               |
| 3    | 3       | 2019-03-07 00:00:00 | Skirt         | 30               |
| 4    | 7       | 2019-02-01 00:00:00 | Coat          | 25               |
| 5    | 7       | 2019-03-10 00:00:00 | Shoes         |                  |

### customers



| id   | first_name | last_name | city          | address | phone_number |
| ---- | ---------- | --------- | ------------- | ------- | ------------ |
| 8    | John       | Joseph    | San Francisco |         | 928-386-8164 |
| 7    | Jill       | Michael   | Austin        |         | 813-297-0692 |
| 4    | William    | Daniel    | Denver        |         | 813-368-1200 |
| 5    | Henry      | Jackson   | Miami         |         | 808-601-7513 |
| 13   | Emma       | Isaac     | Miami         |         |              |

**Solution**

```python
import pandas as pd
import numpy as np

merged = pd.merge(orders,customers,left_on='cust_id',right_on='id',how='right')
null_cust = merged[merged['cust_id'].isnull()]
result = len(null_cust)

```

Find the most expensive products on Amazon for each product category. Output category, product name and the price (as a number)

### innerwear_amazon_com

| product_name                                          | mrp    | price  | pdp_url                                           | brand_name | product_category | retailer | description | rating | review_count | style_attributes | total_sizes | available_size | color |
| ----------------------------------------------------- | ------ | ------ | ------------------------------------------------- | ---------- | ---------------- | -------- | ----------- | ------ | ------------ | ---------------- | ----------- | -------------- | ----- |
| Wacoal Women's Full Figure Basic Beauty Underwire Bra | $50.00 | $50.00 | https://www.amazon.com/-/dp/B005FR9XVK?th=1&psc=1 |            |                  |          |             |        |              |                  |             |                |       |

**Solution**

```python
# Import your libraries
import pandas as pd

# Start writing code
innerwear_amazon_com.head() 

df=innerwear_amazon_com[['product_name','product_category','price']]
df.price = df.price.str.replace('[\$]','').astype('float')

dff = df.groupby(['product_category']).agg({'price':'max'}).reset_index()

result = pd.merge(df,dff,on=['product_category','price'])
```

Combine the first and last names of workers with a space inbetween.

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 500000 | 2014-06-11 09:00:00 | Admin      |

**Solution**

```python
import pandas as pd
import numpy as np

result = (worker['first_name'].map(str) + ' ' + worker['last_name'].map(str))

```

Find the percentage of shipable orders. Consider an order is shipable if the customer's address is known.

### orders

| id   | cust_id | order_date          | order_details | total_order_cost |
| ---- | ------- | ------------------- | ------------- | ---------------- |
| 1    | 3       | 2019-03-04 00:00:00 | Coat          | 100              |
| 2    | 3       | 2019-03-01 00:00:00 | Shoes         | 80               |
| 3    | 3       | 2019-03-07 00:00:00 | Skirt         | 30               |
| 4    | 7       | 2019-02-01 00:00:00 | Coat          | 25               |
| 5    | 7       | 2019-03-10 00:00:00 | Shoes         | 80               |

### customers

| id   | first_name | last_name | city          | address | phone_number |
| ---- | ---------- | --------- | ------------- | ------- | ------------ |
| 8    | John       | Joseph    | San Francisco |         | 928-386-8164 |
| 7    | Jill       | Michael   | Austin        |         | 813-297-0692 |
| 4    | William    | Daniel    | Denver        |         | 813-368-1200 |
| 5    | Henry      | Jackson   | Miami         |         | 808-601-7513 |
| 13   | Emma       | Isaac     | Miami         |         | 808-690-5201 |

```python
import pandas as pd
import numpy as np

merged_df = pd.merge(orders,customers,left_on='cust_id',right_on='id')
merged_df['is_shipable'] = (merged_df.address.notnull()).astype(int)
result = 100 * (merged_df['is_shipable'].sum()/len(merged_df))
```

Find all workers whose first name contains 6 letters and also ends with the letter 'h'.

### worker

| worker_id | first_name | last_name | salary | joining_date        | department |
| --------- | ---------- | --------- | ------ | ------------------- | ---------- |
| 1         | Monika     | Arora     | 100000 | 2014-02-20 09:00:00 | HR         |
| 2         | Niharika   | Verma     | 80000  | 2014-06-11 09:00:00 | Admin      |
| 3         | Vishal     | Singhal   | 300000 | 2014-02-20 09:00:00 | HR         |
| 4         | Amitah     | Singh     | 500000 | 2014-02-20 09:00:00 | Admin      |
| 5         | Vivek      | Bhati     | 50000  |                     |            |

**Solution**

```python
import pandas as pd
import numpy as np

six_char = worker[worker['first_name'].str.contains('[a-zA-Z]{6}$')]
result = six_char[six_char['first_name'].str.endswith("h")]
```

Find products which are exclusive to only Amazon and therefore not sold at Top Shop and Macy's. Your output should include the product name, brand name, price, and rating.

Two products are considered equal if they have the same product name and same maximum retail price (mrp column).

### innerwear_macys_com

| product_name               | mrp    | price  | pdp_url                                                      | brand_name | product_category | retailer | description | rating | review_count | style_attributes | total_sizes | available_size | color |
| -------------------------- | ------ | ------ | ------------------------------------------------------------ | ---------- | ---------------- | -------- | ----------- | ------ | ------------ | ---------------- | ----------- | -------------- | ----- |
| Cotton Tanga Mini Knickers | $20.00 | $20.00 | http://www.topshop.com/en/tsus/product/clothing-70483/lingerie-2313852/cotton-frill-bra-and-tanga-knickers-set-662234 |            |                  |          |             |        |              |                  |             |                |       |

innerwear_macys_com and innerwear_amazon_com has the same structure.

**Solution**



```python
# Import your libraries
import pandas as pd

# Start writing code
retailers = [innerwear_macys_com, innerwear_topshop_com, innerwear_amazon_com]

for r in retailers:
    r['code'] = r['mrp'].astype(str) + r['product_name']
    
df_com = pd.concat([innerwear_macys_com,innerwear_topshop_com])
products = list(df_com['code'].unique())

results = innerwear_amazon_com[~(innerwear_amazon_com['code'].isin(products))]


```

Find the best selling item for each month (no need to separate months by year) where the biggest total invoice was paid. The best selling item is calculated using the formula (unitprice * quantity). Output the description of the item along with the amount paid.

### online_retail

| invoiceno | stockcode | description                        | quantity | invoicedate         | unitprice | customerid | country        |
| --------- | --------- | ---------------------------------- | -------- | ------------------- | --------- | ---------- | -------------- |
| 544586    | 21890     | S/6 WOODEN SKITTLES IN COTTON BAG  | 3        | 2011-02-21 00:00:00 | 2.95      | 17338      | United Kingdom |
| 541104    | 84509G    | SET OF 4 FAIRY CAKE PLACEMATS      | 3        | 2011-01-13 00:00:00 | 3.29      |            | United Kingdom |
| 560772    | 22499     | WOODEN UNION JACK BUNTING          | 3        | 2011-07-20 00:00:00 | 4.96      |            | United Kingdom |
| 555150    | 22488     | NATURAL SLATE RECTANGLE CHALKBOARD | 5        | 2011-05-31 00:00:00 | 3.29      |            | United Kingdom |
| 570521    | 21625     | VINTAGE UNION JACK APRON           | 3        | 2011-10-11 00:00:00 | 6.95      | 12371      | Switzerland    |

**Solution**

```python
# month\description\unitprice * quantity
import pandas as pd

# Start writing code
df = online_retail[['invoicedate', 'description', 'quantity', 'unitprice']]
df['month'] =  df['invoicedate'].dt.month
df['total_invoice'] = df['quantity']*df['unitprice']
df = df.groupby(['month', 'description'])['total_invoice'].sum().reset_index()
df['rank'] = df.groupby(['month']).rank(ascending=False, method='dense')
df[df['rank'] == 1][['month', 'description', 'total_invoice']]
```

**Congratulations!**  We have practiced some coding questions for high tech companies.