---
title: "What data scientist should know? "
excerpt: "What do you learn in data science?"

header:
  image: "../assets/images/posts/2021-06-06-What-all-should-a-data-scientist-know/nexi-mod2_low.jpg"
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

```
SELECT c1, c2 FROM t;
Code language: SQL (Structured Query Language) (sql)
```

Query all rows and columns from a table

```
SELECT * FROM t;
Code language: SQL (Structured Query Language) (sql)
```

Query data and filter rows with a condition

```
SELECT c1, c2 FROM t
WHERE condition;Code language: SQL (Structured Query Language) (sql)
```

Query distinct rows from a table

```
SELECT DISTINCT c1 FROM t
WHERE condition;
Code language: SQL (Structured Query Language) (sql)
```

Sort the result set in ascending or descending order

```
SELECT c1, c2 FROM t
ORDER BY c1 ASC [DESC];Code language: SQL (Structured Query Language) (sql)
```

Skip *offset* of rows and return the next n rows

```
SELECT c1, c2 FROM t
ORDER BY c1 
LIMIT n OFFSET offset;Code language: SQL (Structured Query Language) (sql)
```

Group rows using an aggregate function

```
SELECT c1, aggregate(c2)
FROM t
GROUP BY c1;
Code language: SQL (Structured Query Language) (sql)
```

Filter groups using HAVING clause

```
SELECT c1, aggregate(c2)
FROM t
GROUP BY c1
HAVING condition;
Code language: SQL (Structured Query Language) (sql)
```

## **Querying** from **multiple tables**

Inner join t1 and t2

```
SELECT c1, c2 
FROM t1
INNER JOIN t2 ON condition;
Code language: SQL (Structured Query Language) (sql)
```

Left join t1 and t1

```
SELECT c1, c2 
FROM t1
LEFT JOIN t2 ON condition;
Code language: SQL (Structured Query Language) (sql)
```

Right join t1 and t2

```
SELECT c1, c2 
FROM t1
RIGHT JOIN t2 ON condition;
Code language: SQL (Structured Query Language) (sql)
```

Perform full outer join

```
SELECT c1, c2 
FROM t1
FULL OUTER JOIN t2 ON condition;
Code language: SQL (Structured Query Language) (sql)
```

Produce a Cartesian product of rows in tables

```
SELECT c1, c2 
FROM t1
CROSS JOIN t2;
Code language: SQL (Structured Query Language) (sql)
```

Another way to perform cross join

```
SELECT c1, c2 
FROM t1, t2;
Code language: SQL (Structured Query Language) (sql)
```

Join t1 to itself using INNER JOIN clause

```
SELECT c1, c2
FROM t1 A
INNER JOIN t1 B ON condition;
Code language: SQL (Structured Query Language) (sql)
```

## Using SQL Operators

Combine rows from two queries

```
SELECT c1, c2 FROM t1
UNION [ALL]
SELECT c1, c2 FROM t2;
Code language: SQL (Structured Query Language) (sql)
```

Return the intersection of two queries

```
SELECT c1, c2 FROM t1
INTERSECT
SELECT c1, c2 FROM t2;
Code language: SQL (Structured Query Language) (sql)
```

Subtract a result set from another result set

```
SELECT c1, c2 FROM t1
MINUS
SELECT c1, c2 FROM t2;
Code language: SQL (Structured Query Language) (sql)
```

Query rows using pattern matching %, _

```
SELECT c1, c2 FROM t1
WHERE c1 [NOT] LIKE pattern;
Code language: SQL (Structured Query Language) (sql)
```

Query rows in a list

```
SELECT c1, c2 FROM t
WHERE c1 [NOT] IN value_list;
Code language: SQL (Structured Query Language) (sql)
```

Query rows between two values

```
SELECT c1, c2 FROM t
WHERE  c1 BETWEEN low AND high;
Code language: SQL (Structured Query Language) (sql)
```

Check if values in a table is NULL or not

```
SELECT c1, c2 FROM t
WHERE  c1 IS [NOT] NULL;
Code language: SQL (Structured Query Language) (sql)
```

## Managing tables

Create a new table with three columns

```
CREATE TABLE t (
     id INT PRIMARY KEY,
     name VARCHAR NOT NULL,
     price INT DEFAULT 0
);
Code language: SQL (Structured Query Language) (sql)
```

Delete the table from the database

```
DROP TABLE t ;
Code language: SQL (Structured Query Language) (sql)
```

Add a new column to the table

```
ALTER TABLE t ADD column;
Code language: SQL (Structured Query Language) (sql)
```

Drop column c from the table

```
ALTER TABLE t DROP COLUMN c ;
Code language: SQL (Structured Query Language) (sql)
```

Add a constraint

```
ALTER TABLE t ADD constraint;
Code language: SQL (Structured Query Language) (sql)
```

Drop a constraint

```
ALTER TABLE t DROP constraint;
Code language: SQL (Structured Query Language) (sql)
```

Rename a table from t1 to t2

```
ALTER TABLE t1 RENAME TO t2;
Code language: SQL (Structured Query Language) (sql)
```

Rename column c1 to c2

```
ALTER TABLE t1 RENAME c1 TO c2 ;
Code language: SQL (Structured Query Language) (sql)
```

Remove all data in a table

```
TRUNCATE TABLE t;
Code language: SQL (Structured Query Language) (sql)
```

## **Using** **SQL** constraints

Set c1 and c2 as a primary key

```
CREATE TABLE t(
    c1 INT, c2 INT, c3 VARCHAR,
    PRIMARY KEY (c1,c2)
);
Code language: SQL (Structured Query Language) (sql)
```

Set c2 column as a foreign key

```
CREATE TABLE t1(
    c1 INT PRIMARY KEY,  
    c2 INT,
    FOREIGN KEY (c2) REFERENCES t2(c2)
);
Code language: SQL (Structured Query Language) (sql)
```

Make the values in c1 and c2 unique

```
CREATE TABLE t(
    c1 INT, c1 INT,
    UNIQUE(c2,c3)
);
Code language: SQL (Structured Query Language) (sql)
```

Ensure c1 > 0 and values in c1 >= c2

```
CREATE TABLE t(
  c1 INT, c2 INT,
  CHECK(c1> 0 AND c1 >= c2)
);
Code language: SQL (Structured Query Language) (sql)
```

Set values in c2 column not NULL

```
CREATE TABLE t(
     c1 INT PRIMARY KEY,
     c2 VARCHAR NOT NULL
);
Code language: SQL (Structured Query Language) (sql)
```

## Modifying **Data**

Insert one row into a table

```
INSERT INTO t(column_list)
VALUES(value_list);
Code language: SQL (Structured Query Language) (sql)
```

Insert multiple rows into a table

```
INSERT INTO t(column_list)
VALUES (value_list), 
       (value_list), …;
Code language: SQL (Structured Query Language) (sql)
```

Insert rows from t2 into t1

```
INSERT INTO t1(column_list)
SELECT column_list
FROM t2;
Code language: SQL (Structured Query Language) (sql)
```

Update new value in the column c1 for all rows

```
UPDATE t
SET c1 = new_value;
Code language: SQL (Structured Query Language) (sql)
```

Update values in the column c1, c2 that match the condition

```
UPDATE t
SET c1 = new_value, 
        c2 = new_value
WHERE condition;
Code language: SQL (Structured Query Language) (sql)
```

Delete all data in a table

```
DELETE FROM t;
Code language: SQL (Structured Query Language) (sql)
```

Delete subset of rows in a table

```
DELETE FROM t
WHERE condition;
Code language: SQL (Structured Query Language) (sql)
```

## Managing Views

Create a new view that consists of c1 and c2

```
CREATE VIEW v(c1,c2) 
AS
SELECT c1, c2
FROM t;
Code language: SQL (Structured Query Language) (sql)
```

Create a new view with check option

```
CREATE VIEW v(c1,c2) 
AS
SELECT c1, c2
FROM t;
WITH [CASCADED | LOCAL] CHECK OPTION;Code language: SQL (Structured Query Language) (sql)
```

Create a recursive view

```
CREATE RECURSIVE VIEW v 
AS
select-statement -- anchor part
UNION [ALL]
select-statement; -- recursive part
Code language: SQL (Structured Query Language) (sql)
```

Create a temporary view

```
CREATE TEMPORARY VIEW v 
AS
SELECT c1, c2
FROM t;
Code language: SQL (Structured Query Language) (sql)
```

Delete a view

```
DROP VIEW view_name;
Code language: SQL (Structured Query Language) (sql)
```

## **Managing indexes**

Create an index on c1 and c2 of the t table

```
CREATE INDEX idx_name 
ON t(c1,c2);
Code language: SQL (Structured Query Language) (sql)
```

Create a unique index on c3, c4 of the t table

```
CREATE UNIQUE INDEX idx_name
ON t(c3,c4)
Code language: SQL (Structured Query Language) (sql)
```

Drop an index

```
DROP INDEX idx_name;
Code language: SQL (Structured Query Language) (sql)
```

## **Managing** triggers

Create or modify a trigger

```
CREATE OR MODIFY TRIGGER trigger_name
WHEN EVENT
ON table_name TRIGGER_TYPE
EXECUTE stored_procedure;
Code language: SQL (Structured Query Language) (sql)
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

```
DROP TRIGGER trigger_name;
Code language: SQL (Structured Query Language) (sql)
```

