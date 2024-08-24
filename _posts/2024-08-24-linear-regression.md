# Linear Regression in Python

Linear regression is one of the most basic and widely used predictive modeling techniques in statistics and machine learning. This blog will walk you through the basics of linear regression, and you'll see how to implement it using Python. The code snippets are ready to be executed in a web browser with the help of Jupyter notebooks or similar environments.

## What is Linear Regression?

---
Linear regression attempts to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. The simplest form, **Simple Linear Regression**, involves one independent variable:

\[
y = \beta_0 + \beta_1x + \epsilon
\]

- \( y \) is the dependent variable (the outcome we want to predict).
- \( x \) is the independent variable (the input variable).
- \( \beta_0 \) is the intercept (the value of \( y \) when \( x \) is 0).
- \( \beta_1 \) is the slope of the line (how much \( y \) changes for a unit change in \( x \)).
- \( \epsilon \) is the error term (the difference between the observed and predicted values).

![Gradien Descent](https://global.discourse-cdn.com/dlai/original/3X/b/7/b773c6bcfa1b2afce716e329f08323f262736eb4.jpeg)

## Implementing Linear Regression in Python

---

Let’s dive into some Python code to implement linear regression. We’ll use the **scikit-learn** library, a popular tool for machine learning in Python.

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px

# Create a simple dataset
data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate the independent and dependent variables
X = df[['YearsExperience']]
y = df['Salary']

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Predict using the model
predictions = model.predict(X)

# Display the results
df['PredictedSalary'] = predictions

# Plotting the data
fig = px.scatter(df, x='YearsExperience', y='Salary', title='Salary vs Years of Experience')
fig.add_scatter(x=df['YearsExperience'], y=df['PredictedSalary'], mode='lines', name='Regression Line')

# Show plot in the browser
fig.show()
```
