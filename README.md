
# PT08.S2 (titania) Prediction with Linear Regression, Polynomial Regression

## Problem Statement

The purpose of this study is to predict the PT08.S2 (titania) based on Benzene concentration.

## Dataset

The dataset contains **9358 instances** of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level,within an Italian city. Data were recorded from **March 2004 to February 2005** (one year) representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses. (*You can access the rest of the information via the link given in the next sentence.*) Dataset is downloaded from [https://archive.ics.uci.edu/](https://archive.ics.uci.edu/ml/datasets/Air+quality#) website. Dataset has **15 columns** and **9358 rows with the header**.

## Methodology

In this project, as stated in the title, results were obtained through two different methods. These methods are as respectively listed below:

 1. Linear Regression
 2. Polynomial Regression

## Analysis

| # | Column | Non-Null Count | Dtype |
|--|--|--|--|
| 0 | Date | 9357 non-null | object
| 1 | Time | 9357 non-null | object
| 2 | CO(GT) | 9357 non-null | float64
| 3 | PT08.S1(CO) | 9357 non-null | float64
| 4 | NMHC(GT) | 9357 non-null | float64
| 5 | C6H6(GT) | 9357 non-null | float64
| 6 | PT08.S2(NMHC) | 9357 non-null | float64
| 7 | NOx(GT) | 9357 non-null | float64
| 8 | PT08.S3(NOx) | 9357 non-null | float64
| 9 | NO2(GT) | 9357 non-null | float64
| 10 | PT08.S4(NO2) | 9357 non-null | float64
| 11 | PT08.S5(O3) | 9357 non-null | float64
| 12 | T | 9357 non-null | float64
| 13 | RH | 9357 non-nulll | float64
| 14 | AH | 9357 non-null | float64
| 15 | Unnamed: 15 | 9357 non-null | float64
| 16 | Unnamed: 16 | 9357 non-null | float64

### Prediction
Titania prediction based on Benzene concentration in 20 microg/m³

 **1.** Linear regression
 
 > **1297.73220864**

 **2.** Polinomial regression (degree=2)
 
 > **1318.44340315**

 **3.** Polinomial regression (degree=4)
 
 > **1292.10353595**

### R² Values

 **1.** Linear regression
 
 > **0.9652428158082308**

 **2.** Polinomial regression (degree=2)
 
 > **0.9945846043583563**

 **3.** Polinomial regression (degree=4)
 
 > **0.9996348091872889**
 
### Mean squared error 

 **1.** Linear regression
 
 > **2363.1413780702046**

 **2.** Polinomial regression (degree=2)
 
 > **368.1928158731508**

 **3.** Polinomial regression (degree=4)
 
 > **24.829327820323684**

***Process took 1.210456371307373 seconds.***

## How to Run Code

Before running the code make sure that you have these libraries:

 - pandas 
 - matplotlib
 - seaborn
 - time
 - sklearn
    
## Contact Me

If you have something to say to me please contact me: 

 - Twitter: [Doguilmak](https://twitter.com/Doguilmak).  
 - Mail address: doguilmak@gmail.com
 
