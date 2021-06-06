---
title: "What data scientist should know? "
excerpt: "What do you learn in data science?"

header:
  image: "../assets/images/posts/2021-06-06-What-all-should-a-data-scientist-know/nexi-mod2_high.jpg"
  teaser: "../assets/images/posts/2021-06-06-What-all-should-a-data-scientist-know/nexi-mod2_low.jpg"
  caption: " AWS Solutions | Connect your business in the cloud with Reply.com"
---

In this blog post,  I will try to give you the first  **10 things** to become a **Data Scientist** .

For sure, depending of your background, you should learn many others things needed to become a great Data Scientist.

This is my personal list of the things that as data scientist should know:

### Table of Contents

- [Section 1: Python for Data Science](#section-1)
- [Section 2: Importing Data](#section-2)
- [Section 3:  Queries in SQL ](#section-3)
- [Section 4: Data Wrangling with Pandas ](#section-4)
- [Section 5:   Data Analysis with Numpy](#section-5)
- [Section 6:   Data Visualization with Matplotlib and Seaborn](#section-6)
- [Section 7:   Machine Learning with Scikit-Learn](#section-7)
- [Section 8:  Neural Networks with Keras and TensorFlow](#section-8)
- [Section 9:  SciPy ](#section-9)
- [Section 10:  PySpark ](#section-10)

I should remark that I am **missing** many other import languages that can be used in **Data Science** such as **R, Scala, Ruby , JavaScript,Go and Swift.**

I will focus only in **Python** and a little of **SQL** to manage the data from **Cloud Databases**. 

You  have also to know basis of **Data Engineering** such as in the previous post [here](https://ruslanmv.com/blog/Top-questions-Data-Engineer) and a little of **Mathematics** to understand how to solve the problems first by creating your algorithm which solves what you want to produce and analyze. If you are Physicist like me, should be very useful to you the knowledge of Wolfram Mathematica or MATLAB or Maple to use Python because the relationship with computer algebra.

 I have collected the information from different sources among them:  [Google](https://www.google.com/) , [Udemy](https://www.udemy.com/) , [Coursera](https://www.coursera.com/), [DataCamp](https://www.datacamp.com/) , [Pluralsight](https://www.pluralsight.com/) and [EdX](https://www.edx.org/).



# Section 1 
# Python for Data Science



## Python Operator Precedence

From Python documentation on [operator precedence](http://docs.python.org/reference/expressions.html) (Section 5.15)

Highest precedence at top, lowest at bottom.
Operators in the same box evaluate left to right.

| **Operator**                                     | **Description**                     |
| ------------------------------------------------ | ----------------------------------- |
| ()                                               | Parentheses (grouping)              |
| *f*(args...)                                     | Function call                       |
| *x*[index:index]                                 | Slicing                             |
| *x*[index]                                       | Subscription                        |
| *x.attribute*                                    | Attribute reference                 |
| **                                               | Exponentiation                      |
| ~*x*                                             | Bitwise not                         |
| +*x*, -*x*                                       | Positive, negative                  |
| *, /, %                                          | Multiplication, division, remainder |
| +, -                                             | Addition, subtraction               |
| <<, >>                                           | Bitwise shifts                      |
| &                                                | Bitwise AND                         |
| ^                                                | Bitwise XOR                         |
| \|                                               | Bitwise OR                          |
| in, not in, is, is not, <, <=, >, >=, <>, !=, == | Comparisons, membership, identity   |
| not *x*                                          | Boolean NOT                         |
| and                                              | Boolean AND                         |
| or                                               | Boolean OR                          |
| lambda                                           | Lambda expression                   |

##  Types and Type Conversion

```python
str()
'5', '3.45', 'True' #Variables to strings

int() 
5, 3, 1 #Variables to integers 

float() 
5.0, 1.0 #Variables to floats 

bool()  
True True True , , #Variables to boolean
```



# Section 3

# Queries in SQL 

## **Querying** data **from a table**

Query data in columns c1, c2 from a table

```sql
SELECT c1, c2 FROM t;
```

Query all rows and columns from a table

```sql
SELECT * FROM t;

```

Query data and filter rows with a condition

```sql 
SELECT c1, c2 FROM t
WHERE condition;
```

Query distinct rows from a table

```sql
SELECT DISTINCT c1 FROM t
WHERE condition;

```

Sort the result set in ascending or descending order

```sql
SELECT c1, c2 FROM t
ORDER BY c1 ASC [DESC];
```

Skip *offset* of rows and return the next n rows

```sql
SELECT c1, c2 FROM t
ORDER BY c1 
LIMIT n OFFSET offset;
```

Group rows using an aggregate function

```sql
SELECT c1, aggregate(c2)
FROM t
GROUP BY c1;

```

Filter groups using HAVING clause

```sql
SELECT c1, aggregate(c2)
FROM t
GROUP BY c1
HAVING condition;

```

## **Querying** from **multiple tables**

Inner join t1 and t2

```sql
SELECT c1, c2 
FROM t1
INNER JOIN t2 ON condition;

```

Left join t1 and t1

```sql
SELECT c1, c2 
FROM t1
LEFT JOIN t2 ON condition;

```

Right join t1 and t2

```sql
SELECT c1, c2 
FROM t1
RIGHT JOIN t2 ON condition;

```

Perform full outer join

```sql
SELECT c1, c2 
FROM t1
FULL OUTER JOIN t2 ON condition;

```

Produce a Cartesian product of rows in tables

```sql
SELECT c1, c2 
FROM t1
CROSS JOIN t2;

```

Another way to perform cross join

```sql
SELECT c1, c2 
FROM t1, t2;

```

Join t1 to itself using INNER JOIN clause

```sql
SELECT c1, c2
FROM t1 A
INNER JOIN t1 B ON condition;

```

## Using SQL Operators

Combine rows from two queries

```sql
SELECT c1, c2 FROM t1
UNION [ALL]
SELECT c1, c2 FROM t2;

```

Return the intersection of two queries

```sql
SELECT c1, c2 FROM t1
INTERSECT
SELECT c1, c2 FROM t2;

```

Subtract a result set from another result set

```sql
SELECT c1, c2 FROM t1
MINUS
SELECT c1, c2 FROM t2;

```

Query rows using pattern matching %, _

```sql
SELECT c1, c2 FROM t1
WHERE c1 [NOT] LIKE pattern;

```

Query rows in a list

```sql
SELECT c1, c2 FROM t
WHERE c1 [NOT] IN value_list;

```

Query rows between two values

```sql
SELECT c1, c2 FROM t
WHERE  c1 BETWEEN low AND high;

```

Check if values in a table is NULL or not

```sql
SELECT c1, c2 FROM t
WHERE  c1 IS [NOT] NULL;

```

## Managing tables

Create a new table with three columns

```sql
CREATE TABLE t (
     id INT PRIMARY KEY,
     name VARCHAR NOT NULL,
     price INT DEFAULT 0
);

```

Delete the table from the database

```sql
DROP TABLE t ;

```

Add a new column to the table

```sql
ALTER TABLE t ADD column;

```

Drop column c from the table

```sql
ALTER TABLE t DROP COLUMN c ;

```

Add a constraint

```sql
ALTER TABLE t ADD constraint;

```

Drop a constraint

```sql
ALTER TABLE t DROP constraint;

```

Rename a table from t1 to t2

```sql
ALTER TABLE t1 RENAME TO t2;

```

Rename column c1 to c2

```sql
ALTER TABLE t1 RENAME c1 TO c2 ;

```

Remove all data in a table

```sql
TRUNCATE TABLE t;

```

## **Using** **SQL** constraints

Set c1 and c2 as a primary key

```sql
CREATE TABLE t(
    c1 INT, c2 INT, c3 VARCHAR,
    PRIMARY KEY (c1,c2)
);

```

Set c2 column as a foreign key

```sql
CREATE TABLE t1(
    c1 INT PRIMARY KEY,  
    c2 INT,
    FOREIGN KEY (c2) REFERENCES t2(c2)
);

```

Make the values in c1 and c2 unique

```sql
CREATE TABLE t(
    c1 INT, c1 INT,
    UNIQUE(c2,c3)
);

```

Ensure c1 > 0 and values in c1 >= c2

```sql
CREATE TABLE t(
  c1 INT, c2 INT,
  CHECK(c1> 0 AND c1 >= c2)
);

```

Set values in c2 column not NULL

```sql
CREATE TABLE t(
     c1 INT PRIMARY KEY,
     c2 VARCHAR NOT NULL
);

```

## Modifying **Data**

Insert one row into a table

```sql
INSERT INTO t(column_list)
VALUES(value_list);

```

Insert multiple rows into a table

```sql
INSERT INTO t(column_list)
VALUES (value_list), 
       (value_list), …;

```

Insert rows from t2 into t1

```sql
INSERT INTO t1(column_list)
SELECT column_list
FROM t2;

```

Update new value in the column c1 for all rows

```sql
UPDATE t
SET c1 = new_value;

```

Update values in the column c1, c2 that match the condition

```sql
UPDATE t
SET c1 = new_value, 
        c2 = new_value
WHERE condition;

```

Delete all data in a table

```sql
DELETE FROM t;

```

Delete subset of rows in a table

```sql
DELETE FROM t
WHERE condition;

```

## Managing Views

Create a new view that consists of c1 and c2

```sql
CREATE VIEW v(c1,c2) 
AS
SELECT c1, c2
FROM t;

```

Create a new view with check option

```sql
CREATE VIEW v(c1,c2) 
AS
SELECT c1, c2
FROM t;
WITH [CASCADED | LOCAL] CHECK OPTION;
```

Create a recursive view

```sql
CREATE RECURSIVE VIEW v 
AS
select-statement -- anchor part
UNION [ALL]
select-statement; -- recursive part

```

Create a temporary view

```sql
CREATE TEMPORARY VIEW v 
AS
SELECT c1, c2
FROM t;

```

Delete a view

```sql
DROP VIEW view_name;

```

## **Managing indexes**

Create an index on c1 and c2 of the t table

```sql
CREATE INDEX idx_name 
ON t(c1,c2);

```

Create a unique index on c3, c4 of the t table

```sql
CREATE UNIQUE INDEX idx_name
ON t(c3,c4)

```

Drop an index

```sql
DROP INDEX idx_name;

```

## **Managing** triggers

Create or modify a trigger

```sql
CREATE OR MODIFY TRIGGER trigger_name
WHEN EVENT
ON table_name TRIGGER_TYPE
EXECUTE stored_procedure;

```

**WHEN**

- **BEFORE** – invoke before the event occurs
- **AFTER** – invoke after the event occurs

**EVENT**

- **INSERT** – invoke for INSERT
- **UPDATE** – invoke for UPDATE
- **DELETE** – invoke for DELETE

**TRIGGER_TYPE**

- **FOR EACH ROW**
- **FOR EACH STATEMENT**

Delete a specific trigger

```sql
DROP TRIGGER trigger_name;

```






